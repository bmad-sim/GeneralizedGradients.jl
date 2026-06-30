# ---------------------------------------------------------------------------
# field_grid_io.jl
#
# HDF5 storage for the project's native data products:
#   * field grids       -- write_field_grid (the Bmad openPMD field_grid format;
#                          a thin wrapper over field_grid_hdf5.jl, so field grids
#                          are also Bmad files). Read them with read_field_grid_hdf5.
#   * GG fit results     -- gg_load_fit (written by gg_fit_write_results)
#
# These replace the former JLD2 `load`/`save`/`jldsave` storage.  Plain HDF5 has
# no notion of Julia Dicts, so the GG-fit-result schema below stores the
# dictionary keys explicitly alongside the numeric data.
# ---------------------------------------------------------------------------

using HDF5, OffsetArrays

# ===========================================================================
# Field grid
#
# The storage format is chosen by file extension:
#   * ".h5" / ".hdf5"  -> Bmad openPMD `field_grid` HDF5 (read_field_grid_hdf5 /
#                         write_field_grid_hdf5 in field_grid_hdf5.jl), so the file
#                         is also a valid Bmad field_grid file.
#   * anything else    -> a Julia source file (like ags-snakes/wsnk_fieldmap.jl)
#                         that, when `include`d, defines `fg::FieldGridTable`.
# ===========================================================================

# True if `path` should be treated as an HDF5 file (".h5" or ".hdf5" suffix).
_is_hdf5_path(path) = lowercase(splitext(path)[2]) in (".h5", ".hdf5")

#---------------------------------------------------------------------------------------------------
"""
    write_field_grid(path, fg::FieldGridTable)

Write a `FieldGridTable`.  If `path` ends in `.h5`/`.hdf5` it is written as a
Bmad openPMD `field_grid` HDF5 file (`write_field_grid_hdf5`, readable by Bmad);
otherwise it is written as a Julia source file (like `ags-snakes/wsnk_fieldmap.jl`)
that defines `fg` when `include`d.
"""
function write_field_grid(path::AbstractString, fg::FieldGridTable)
  _is_hdf5_path(path) && return write_field_grid_hdf5(path, fg)
  ndims(fg.magnetic) == 3 ||
    error("fg.magnetic must be a 3D (ix, iy, iz) array of [Bx,By,Bz] 3-vectors.")
  open(path, "w") do io
    println(io, "# Field grid for GeneralizedGradients. `include` this file to define `fg`.")
    println(io, "# A point fg.magnetic[ix, iy, iz] is at (x, y, z) = r0 + dr .* (ix, iy, iz)")
    println(io)
    println(io, "using OffsetArrays")
    println(io, "using GeneralizedGradients")
    println(io)
    println(io, "fg = FieldGridTable()")
    println(io)
    println(io, "fg.r0           = ", fg.r0, "     # Grid origin")
    println(io, "fg.dr           = ", fg.dr, "     # Grid spacing")
    println(io, "fg.g_ref        = ", fg.g_ref, "     # curvilinear coordinates 1 / bending_radius")
    println(io, "fg.scale        = ", fg.scale, "     # field scale factor")
    println(io, "fg.RF_frequency = ", fg.RF_frequency)
    println(io, "fg.RF_phase     = ", fg.RF_phase)
    println(io, "fg.anchor_pt    = GridAnchorPt.", fg.anchor_pt)
    println(io, "fg.geometry     = GridGeometry.", fg.geometry)
    _write_field_component_jl(io, "magnetic", fg.magnetic)
    isempty(fg.electric) || _write_field_component_jl(io, "electric", fg.electric)
  end
  return path
end

# Write the `fg.<name>` OffsetArray of [Bx,By,Bz] 3-vectors as include-able Julia.
function _write_field_component_jl(io, name, field)
  ax = axes(field)
  nx, ny, nz = length.(ax)
  ox, oy, oz = first(ax[1]) - 1, first(ax[2]) - 1, first(ax[3]) - 1
  println(io)
  println(io, "temp = Array{Vector{Float64}}(undef, $nx, $ny, $nz);")
  println(io, "fg.$name = OffsetArray(temp, $ox, $oy, $oz);")
  println(io)
  for ix in ax[1], iy in ax[2], iz in ax[3]
    b = field[ix, iy, iz]
    println(io, "fg.$name[$ix, $iy, $iz] = [", b[1], ", ", b[2], ", ", b[3], "]")
  end
end

# ===========================================================================
# GG fit result
#
# HDF5 schema (written by gg_fit_write_results, read by gg_load_fit):
#   root datasets   : z_base, rms_plane, origin            (Float64[])
#   root attributes : g_ref, dz_grid (Float64); m_max, n_planes_add (Int);
#                     core_weight, outer_plane_weight (Float64)
#   groups a, b     : n (Int[]), m (Int[]), values (Float64[nkeys, nplanes])
#                     -- reconstruct Dict{(n,m) => values[i,:]}
#   group  bs       : m (Int[]), values (Float64[nkeys, nplanes])
#                     -- reconstruct Dict{m => values[i,:]}
# ===========================================================================

# Write a Dict keyed by (n,m) (or by m, if `single`) as index arrays + matrix.
function _write_coef_group(parent, name, d; single::Bool = false)
  g = create_group(parent, name)
  ks = sort(collect(keys(d)))
  nplanes = isempty(ks) ? 0 : length(d[first(ks)])
  V = Array{Float64}(undef, length(ks), nplanes)
  for (i, k) in enumerate(ks)
    V[i, :] = d[k]
  end
  if single
    g["m"] = Int[k for k in ks]
  else
    g["n"] = Int[k[1] for k in ks]
    g["m"] = Int[k[2] for k in ks]
  end
  g["values"] = V
end

function _read_coef_group(parent, name; single::Bool = false)
  g = parent[name]
  m = Int.(read(g["m"]))
  V = read(g["values"])
  if single
    return Dict{Int,Vector{Float64}}(m[i] => V[i, :] for i in eachindex(m))
  else
    n = Int.(read(g["n"]))
    return Dict{Tuple{Int,Int},Vector{Float64}}((n[i], m[i]) => V[i, :] for i in eachindex(m))
  end
end

#---------------------------------------------------------------------------------------------------
# gg_load_fit

"""
    gg_load_fit(path::AbstractString) -> (fit::GGCoefs, meta::NamedTuple)

Load a `gg_fit` result HDF5 file (written by `gg_fit_write_results`). Returns a
two-tuple whose first component is a `GGCoefs` struct holding the GG
coefficient dictionaries `a`, `b`, `bs` (and `z_base`, `m_max`, `rms_plane`),
and whose second component is a NamedTuple of the associated fit metadata
(`g_ref`, `origin`, `dz_grid`, `n_planes_add`, `core_weight`,
`outer_plane_weight`). The `params` field of the returned struct is empty (the
unknown list is not stored in the file).

```julia
fit, meta = gg_load_fit(path)
fit.a            # Dict{(n,m) => values_over_planes}
meta.g_ref       # reference curvature
```
"""
function gg_load_fit(path::AbstractString)
  h5open(path, "r") do f
    fit = GGCoefs(; z_base    = read(f["z_base"]),
                         a         = _read_coef_group(f, "a"),
                         b         = _read_coef_group(f, "b"),
                         bs        = _read_coef_group(f, "bs"; single = true),
                         m_max     = Int(read_attribute(f, "m_max")),
                         rms_plane = read(f["rms_plane"]))
    meta = (; g_ref   = read_attribute(f, "g_ref"),
              origin  = read(f["origin"]),
              dz_grid = read_attribute(f, "dz_grid"),
              # Fit-control metadata, retained for reference / reproducibility.
              n_planes_add       = read_attribute(f, "n_planes_add"),
              core_weight        = read_attribute(f, "core_weight"),
              outer_plane_weight = read_attribute(f, "outer_plane_weight"))
    return fit, meta
  end
end
