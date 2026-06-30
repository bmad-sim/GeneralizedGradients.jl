# ---------------------------------------------------------------------------
# field_grid.jl
#
# Everything for reading, writing, and exporting 3D field grids:
#   * Bmad `grid_field` export -- write_bmad_field_grid (see also the shell
#     wrapper programs/run_write_bmad_field_grid.jl).
#   * native storage I/O        -- write_field_grid / read_gg_fit / write_gg_fit.
#   * openPMD HDF5 field maps    -- read_field_grid_hdf5 / write_field_grid_hdf5.
# ---------------------------------------------------------------------------

# ===========================================================================
# Bmad grid_field export
#
# Read a 3D field grid and write it out in Bmad `grid_field` format (a lattice
# element with the field grid attached).  `write_bmad_field_grid` is the public
# function; programs/run_write_bmad_field_grid.jl is a shell wrapper.
# ===========================================================================

# Format a real for a Bmad lattice file: compact but lossless. `iszero` guard
# avoids printing a signed "-0".
_grid_num(x::Real) = iszero(x) ? "0" : @sprintf("%.15g", float(x))

# Write the plain-text field-grid block from an (ix, iy, iz) OffsetArray of
# [Bx,By,Bz] 3-vectors, using the grid's own indices (origin `r0`, spacing `dr`,
# anchor = beginning).
function _write_field_grid_text(path, mag, r0, dr, is_bend, field_scale)
  ax = axes(mag)
  open(path, "w") do io
    println(io, "{")
    println(io, "  geometry = xyz,")
    println(io, "  field_type = magnetic,")
    println(io, "  ele_anchor_pt = beginning,")
    is_bend && println(io, "  curved_ref_frame = T,")
    field_scale != 1 && println(io, "  field_scale = ", _grid_num(field_scale), ",")
    println(io, "  r0 = (", _grid_num(r0[1]), ", ", _grid_num(r0[2]), ", ", _grid_num(r0[3]), "),")
    println(io, "  dr = (", _grid_num(dr[1]), ", ", _grid_num(dr[2]), ", ", _grid_num(dr[3]), "),")
    println(io, "  {")
    for iz in ax[3], iy in ax[2], ix in ax[1]
      B = mag[ix, iy, iz]
      @printf(io, "    %d %d %d: %s %s %s,\n",
          ix, iy, iz, _grid_num(B[1]), _grid_num(B[2]), _grid_num(B[3]))
    end
    println(io, "  }")
    println(io, "}")
  end
end

#---------------------------------------------------------------------------------------------------

"""
    write_bmad_field_grid(field; ele_name, output_base, field_scale, hdf5)

Translate a field grid into Bmad `grid_field` format. Writes two files and
returns the path of the lattice-element file.

- `field` — either a `FieldGridTable` (see `read_field_grid_hdf5`) or a string
  path to a Bmad openPMD `field_grid` HDF5 file, which is read with
  `read_field_grid_hdf5`.

Keyword arguments:
- `ele_name`    — name of the Bmad lattice element. Default `"fieldmap_ele"`.
- `output_base` — base path for the two output files. Default `ele_name`.
- `field_scale` — overall field scale factor written to the field grid. Default `1`.
- `hdf5`        — if true (the default), write the field grid as an openPMD HDF5 file
  (`<output_base>_grid.h5`) instead of a plain-text block.

Two files are written: `<output_base>.bmad` (the lattice element) and the field
grid (`<output_base>_grid.h5` or `_grid.bmad`). The reference-curve bending
strength is taken from the field grid's `g_ref` (`= 1/bend_radius`): if non-zero
the element is written as an `sbend`, otherwise an `em_field`. The grid is
anchored at the entrance of the element (`ele_anchor_pt = beginning`) with length
`L = dz*(nz-1)`; the field-grid `r0` keeps the transverse offset `(x0, y0)` of the
input grid and shifts z so the first grid plane sits at the element entrance.
"""
function write_bmad_field_grid(field::Union{AbstractString,FieldGridTable};
                               ele_name::AbstractString = "fieldmap_ele",
                               output_base::AbstractString = ele_name,
                               field_scale::Real = 1.0,
                               hdf5::Bool = true)

  field isa AbstractString && (field = read_field_grid_hdf5(field))
  g_ref = field.g_ref

  mag = field.magnetic
  dr  = field.dr
  dz  = dr[3]
  zax = axes(mag, 3)
  iz_lo = first(zax)
  nz  = length(zax)
  is_bend = g_ref != 0
  L = dz * (nz - 1)                        # longitudinal span of the grid

  # Anchor the grid at the entrance of the element: shift z so the first plane
  # lands at element z = 0. The transverse position of the grid is preserved,
  # and the grid index ranges are kept as-is.
  r0_out = [field.r0[1], field.r0[2], -dz * iz_lo]

  grid_file = output_base * (hdf5 ? "_grid.h5" : "_grid.bmad")
  ele_file  = output_base * ".bmad"
  grid_name = basename(grid_file)

  # ---- Write the field grid --------------------------------------------
  if hdf5
    fg = FieldGridTable{Float64}(;
      magnetic = mag,                  # keep offset indices -> gridLowerBound preserved
      r0 = r0_out,
      dr = collect(Float64, dr),
      g_ref = float(g_ref),
      scale = float(field_scale),
      anchor_pt = GridAnchorPt.Beginning,
      geometry = GridGeometry.XYZ)
    write_field_grid_hdf5(grid_file, fg)
  else
    _write_field_grid_text(grid_file, mag, r0_out, dr, is_bend, field_scale)
  end

  # ---- Write the lattice element ---------------------------------------
  open(ele_file, "w") do io
    println(io, "! Bmad lattice element with attached field grid.")
    println(io, "! Generated from write_bmad_field_grid.")
    println(io, "!")
    if is_bend
      println(io)
      println(io, ele_name, ": sbend,")
      println(io, "  g = ", _grid_num(g_ref), ",")
    else
      println(io)
      println(io, ele_name, ": em_field,")
    end

    println(io, "  l = ", _grid_num(L), ",")
    println(io, "  field_calc = fieldmap,")
    println(io, "  tracking_method = runge_kutta,")
    println(io, "  mat6_calc_method = tracking,")
    println(io, "  grid_field = call::", grid_name)
  end

  return ele_file
end

# ===========================================================================
# Native storage I/O
#
# HDF5 storage for the project's native data products:
#   * field grids       -- write_field_grid (the Bmad openPMD field_grid format;
#                          a thin wrapper over the HDF5 routines below, so field
#                          grids are also Bmad files). Read them with read_field_grid_hdf5.
#   * GG fit results    -- read_gg_fit (written by write_gg_fit)
# ===========================================================================

using HDF5, OffsetArrays

# ===========================================================================
# Field grid
#
# The storage format is chosen by file extension:
#   * ".h5" / ".hdf5"  -> Bmad openPMD `field_grid` HDF5 (read_field_grid_hdf5 /
#                         write_field_grid_hdf5 below), so the file
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
# HDF5 schema (written by write_gg_fit, read by read_gg_fit):
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
# read_gg_fit

"""
    read_gg_fit(path::AbstractString) -> (fit::GGCoefs, meta::NamedTuple)

Load a `gg_fit` result HDF5 file (written by `write_gg_fit`). Returns a
two-tuple whose first component is a `GGCoefs` struct holding the GG
coefficient dictionaries `a`, `b`, `bs` (and `z_base`, `m_max`, `rms_plane`,
`g_ref`), and whose second component is a NamedTuple of the associated fit
metadata (`origin`, `dz_grid`, `n_planes_add`, `core_weight`,
`outer_plane_weight`). The `params` field of the returned struct is empty (the
unknown list is not stored in the file).

```julia
fit, meta = read_gg_fit(path)
fit.a            # Dict{(n,m) => values_over_planes}
fit.g_ref        # reference curvature
```
"""
function read_gg_fit(path::AbstractString)
  h5open(path, "r") do f
    fit = GGCoefs(; z_base    = read(f["z_base"]),
                         a         = _read_coef_group(f, "a"),
                         b         = _read_coef_group(f, "b"),
                         bs        = _read_coef_group(f, "bs"; single = true),
                         m_max     = Int(read_attribute(f, "m_max")),
                         rms_plane = read(f["rms_plane"]),
                         g_ref     = read_attribute(f, "g_ref"))
    meta = (; origin  = read(f["origin"]),
              dz_grid = read_attribute(f, "dz_grid"),
              # Fit-control metadata, retained for reference / reproducibility.
              n_planes_add       = read_attribute(f, "n_planes_add"),
              core_weight        = read_attribute(f, "core_weight"),
              outer_plane_weight = read_attribute(f, "outer_plane_weight"))
    return fit, meta
  end
end

# ===========================================================================
# openPMD HDF5 field maps
#
# Read and write Bmad `field_grid` field maps (as a `FieldGridTable`) in openPMD
# HDF5 format, matching Bmad's hdf5_write_grid_field.f90 / hdf5_read_grid_field.f90.
# Currently supports `geometry = xyz` (rectangular) grids.
#
# openPMD/HDF5 ordering note: each field component is written as a plain 1-based
# (nx, ny, nz) Julia array.  HDF5.jl reverses dimensions on write (column-major,
# like Fortran), giving on-disk C-dims (nz, ny, nx) -- byte-identical to Bmad's
# Fortran writer (H5Screate_simple_f with Fortran dims [nx,ny,nz]).  Bmad reads
# the size via H5LTget_dataset_info_f, which reverses back to (nx,ny,nz); combined
# with reversed `axisLabels` / `gridDataOrder = "F"` its size check and data read
# both succeed.  On read, HDF5.jl likewise hands back a (nx,ny,nz) array == the
# field, so no transpose is needed.
# ---------------------------------------------------------------------------

# openPMD complex field samples are a compound type with real ("r") and
# imaginary ("i") double members -- identical to Bmad's pmd_init_compound_complex.
struct ComplexPMD
  r::Float64
  i::Float64
end

# openPMD SI base-unit exponents (L, M, T, I, Theta, N, J) for Tesla and V/m.
const _DIM_TESLA = [0.0, 1, -2, -1, 0, 0, 0]
const _DIM_VPERM = [1.0, 1, -3, -1, 0, 0, 0]

# Write a fixed-length (null-terminated, ASCII) string-array attribute, matching
# Bmad's `hdf5_write_attribute_string` rank-1.  HDF5.jl writes String arrays as
# variable-length strings by default, which Bmad's reader cannot convert into its
# fixed `character` buffers (it aborts on `axisLabels`).
function _write_fixed_str_array(parent, name, strs::AbstractVector{<:AbstractString})
  n = maximum(length, strs)
  dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
  HDF5.API.h5t_set_size(dt, n)
  HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLTERM)
  HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
  dspace = dataspace((length(strs),))
  attr = create_attribute(parent, name, dt, dspace)
  buf = zeros(UInt8, n * length(strs))
  for (i, s) in enumerate(strs)
    cu = codeunits(s)
    copyto!(buf, (i - 1) * n + 1, cu, 1, length(cu))
  end
  HDF5.API.h5a_write(attr, dt, buf)
  close(attr); close(dspace); close(dt)
end

# Map the GridAnchorPt enum <-> the openPMD `eleAnchorPt` strings.
function _anchor_to_str(a::GridAnchorPt.T)
  a == GridAnchorPt.Beginning && return "beginning"
  a == GridAnchorPt.Center    && return "center"
  return "end"
end

function _anchor_from_str(s)
  ls = lowercase(strip(string(s)))
  ls == "beginning" && return GridAnchorPt.Beginning
  ls == "center"    && return GridAnchorPt.Center
  ls == "end"       && return GridAnchorPt.End
  error("Unrecognized eleAnchorPt: $s")
end

# Map the GridGeometry enum <-> the openPMD `gridGeometry` strings.
_geometry_to_str(::GridGeometry.T) = "rectangular"   # only XYZ supported
function _geometry_from_str(s)
  s == "rectangular" && return GridGeometry.XYZ
  error("read_field_grid_hdf5 supports only 'rectangular' (xyz) grids, got: $s")
end

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

# Lay component `c` of a (ix, iy, iz) OffsetArray of 3-vectors out as a 1-based
# (nx, ny, nz) complex array.  HDF5.jl reverses dims on write, so the dataset
# lands on disk exactly like Bmad's own Fortran writer (H5Screate_simple_f with
# Fortran dims [nx,ny,nz]): Bmad's reader gets data_dim = (nx,ny,nz) and, with
# data_order "F", reads the column-major buffer back into pt[ix,iy,iz] correctly.
function _component_dataset(field, c)
  ax = axes(field)
  nx, ny, nz = length(ax[1]), length(ax[2]), length(ax[3])
  out = Array{ComplexPMD}(undef, nx, ny, nz)
  for (a, ix) in enumerate(ax[1]), (b, iy) in enumerate(ax[2]), (k, iz) in enumerate(ax[3])
    v = field[ix, iy, iz][c]
    out[a, b, k] = ComplexPMD(real(v), imag(v))
  end
  return out
end

# Write one field group ("magneticField"/"electricField") from an (ix,iy,iz)
# OffsetArray of 3-vectors.
function _write_field_group(g1, name, field, unit_dim, unit_sym)
  grp = create_group(g1, name)
  for (c, axis) in enumerate(("x", "y", "z"))
    grp[axis] = _component_dataset(field, c)
    da = attributes(grp[axis])
    da["gridDataOrder"] = "F"           # explicit; Bmad reader honors this first
    da["localName"]     = axis
    da["unitSI"]        = [1.0]
    da["unitDimension"] = unit_dim
    da["unitSymbol"]    = unit_sym
  end
end

"""
    write_field_grid_hdf5(path, fg::FieldGridTable)

Write a `FieldGridTable` as an openPMD HDF5 `field_grid` file matching Bmad's
`hdf5_write_grid_field` (geometry = xyz).  The `fg.magnetic` and/or `fg.electric`
OffsetArrays are indexed `(ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi)` with each
element a `[Bx,By,Bz]` 3-vector; an empty array means that field type is omitted.
`gridLowerBound` is taken from the array's index ranges (the grids are not assumed
to start at zero) and `fg.r0` is written as `gridOriginOffset`, so a grid point
`(ix,iy,iz)` is at `dr .* (ix,iy,iz) + r0` relative to the anchor.  A non-zero
`fg.g_ref` sets `gridCurvatureRadius = 1/g_ref` so Bmad enables `curved_ref_frame`.
"""
function write_field_grid_hdf5(path::AbstractString, fg::FieldGridTable)
  has_mag = !isempty(fg.magnetic)
  has_elec = !isempty(fg.electric)
  (has_mag || has_elec) ||
    error("FieldGridTable has no magnetic or electric field data.")
  ref = has_mag ? fg.magnetic : fg.electric
  ax = axes(ref)
  lb = (first(ax[1]), first(ax[2]), first(ax[3]))
  nx, ny, nz = length(ax[1]), length(ax[2]), length(ax[3])

  h5open(path, "w") do f
    attributes(f)["dataType"]          = "Bmad:grid_field"
    attributes(f)["openPMD"]           = "2.0.0"
    attributes(f)["openPMDextension"]  = "BeamPhysics;SpeciesType"
    attributes(f)["externalFieldPath"] = "/ExternalFieldMesh/%T/"
    attributes(f)["software"]          = "GeneralizedGradients"
    attributes(f)["softwareVersion"]   = "1.0"
    attributes(f)["date"]              = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

    g = create_group(f, "ExternalFieldMesh")
    g1 = create_group(g, "1")
    a = attributes(g1)
    a["gridGeometry"]        = _geometry_to_str(fg.geometry)
    a["fieldScale"]          = [Float64(fg.scale)]
    a["componentFieldScale"] = [Float64(fg.scale)]
    # Fixed-length (not variable-length) so Bmad can read it; reversed -> data_order F.
    _write_fixed_str_array(g1, "axisLabels", ["z", "y", "x"])
    a["eleAnchorPt"]         = _anchor_to_str(fg.anchor_pt)
    a["gridOriginOffset"]    = Float64[fg.r0...]
    a["gridSpacing"]         = Float64[fg.dr...]
    a["harmonic"]            = Int32[0]
    a["interpolationOrder"]  = Int32[1]
    a["gridLowerBound"]      = Int32[lb...]
    a["gridSize"]            = Int32[nx, ny, nz]
    a["gridCurvatureRadius"] = fg.g_ref == 0 ? [0.0] : Float64[1 / fg.g_ref]
    if fg.RF_frequency != 0
      a["fundamentalFrequency"] = [Float64(fg.RF_frequency)]
      a["RFphase"]              = [Float64(fg.RF_phase)]
    end

    has_mag  && _write_field_group(g1, "magneticField", fg.magnetic, _DIM_TESLA, "Tesla")
    has_elec && _write_field_group(g1, "electricField", fg.electric, _DIM_VPERM, "V/m")
  end
  return path
end

# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

# Read an attribute if present, else return `default`.
_attr(obj, name, default) = haskey(attributes(obj), name) ? read_attribute(obj, name) : default

# Read a field group ("magneticField"/"electricField") into an (ix, iy, iz)
# OffsetArray of [Bx,By,Bz] 3-vectors indexed from `lb`, or `nothing` if absent.
#
# In a Bmad field_grid file each component dataset is written Fortran-order
# (logical dims [nx,ny,nz]; on-disk C-dims (nz,ny,nx)).  HDF5.jl reverses dims on
# read, so it hands back a 1-based (nx, ny, nz) array that is already the field --
# no transpose needed.
function _read_field_group(g1, name, lb, nx, ny, nz)
  haskey(g1, name) || return nothing
  grp = g1[name]
  comps = ntuple(_ -> zeros(Float64, nx, ny, nz), 3)   # one (nx,ny,nz) array per component
  for (c, axis) in enumerate(("x", "y", "z"))
    haskey(grp, axis) || continue       # missing component => zero field
    comp = read(grp[axis])
    size(comp) == (nx, ny, nz) ||
      error("field_grid dataset $name/$axis has size $(size(comp)), expected ($nx, $ny, $nz) " *
            "-- not a Bmad-format (Fortran-order) field_grid file.")
    comps[c] .= real.(comp)
  end
  field = [Float64[comps[1][a, b, k], comps[2][a, b, k], comps[3][a, b, k]]
      for a in 1:nx, b in 1:ny, k in 1:nz]
  return OffsetArray(field, lb[1]:lb[1]+nx-1, lb[2]:lb[2]+ny-1, lb[3]:lb[3]+nz-1)
end

"""
    read_field_grid_hdf5(path; index = 1) -> FieldGridTable

Read a Bmad/openPMD `field_grid` HDF5 file (as written by
`write_field_grid_hdf5`, or by Bmad itself) into a `FieldGridTable`.  The
`magnetic`/`electric` OffsetArrays are indexed `(ix_lo:ix_hi, …)` with each
element a `[Bx,By,Bz]` 3-vector; the grid index ranges come from
`gridLowerBound`/`gridSize` (the grid is not assumed to start at zero).  `r0` is
`gridOriginOffset` (so a point `(ix,iy,iz)` is at
`dr .* (ix,iy,iz) + r0` relative to the anchor).  An absent field type is left at
the struct default.  `index` selects which grid under `/ExternalFieldMesh/` to
read (default 1).
"""
function read_field_grid_hdf5(path::AbstractString; index::Integer = 1)
  h5open(path, "r") do f
    haskey(f, "ExternalFieldMesh") ||
      error("Not a Bmad field_grid HDF5 file (missing /ExternalFieldMesh): $path")
    g1 = f["ExternalFieldMesh"][string(index)]

    geometry = _geometry_from_str(_attr(g1, "gridGeometry", "rectangular"))

    lb  = Int.(read_attribute(g1, "gridLowerBound"))
    sz  = Int.(read_attribute(g1, "gridSize"))
    r0  = collect(Float64, read_attribute(g1, "gridOriginOffset"))
    dr  = collect(Float64, read_attribute(g1, "gridSpacing"))
    nx, ny, nz = sz[1], sz[2], sz[3]

    scale   = Float64(first(_attr(g1, "fieldScale", _attr(g1, "componentFieldScale", [1.0]))))
    anchor  = _anchor_from_str(_attr(g1, "eleAnchorPt", "center"))
    rho     = first(_attr(g1, "gridCurvatureRadius", [0.0]))
    rf_freq = Float64(first(_attr(g1, "fundamentalFrequency", [0.0])))
    rf_phase = Float64(first(_attr(g1, "RFphase", [0.0])))

    mag = _read_field_group(g1, "magneticField", lb, nx, ny, nz)
    elec = _read_field_group(g1, "electricField", lb, nx, ny, nz)
    (mag === nothing && elec === nothing) &&
      error("Grid has neither magneticField nor electricField: $path")

    fg = FieldGridTable{Float64}(;
      r0 = r0,                        # gridOriginOffset (grid indices kept as-is)
      dr = dr,
      g_ref = rho == 0 ? 0.0 : 1 / rho,
      scale = scale,
      RF_frequency = rf_freq,
      RF_phase = rf_phase,
      anchor_pt = anchor,
      geometry = geometry)
    mag  !== nothing && (fg.magnetic = mag)
    elec !== nothing && (fg.electric = elec)
    return fg
  end
end
