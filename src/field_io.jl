# ---------------------------------------------------------------------------
# field_io.jl
#
# HDF5 storage for the project's native data products:
#   * field grids       -- read_field_grid / write_field_grid (the Bmad openPMD
#                          grid_field format; thin aliases for the functions in
#                          hdf5_grid_field.jl, so field grids are also Bmad files)
#   * GG fit results     -- gg_load_fit / gg_save_fit
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
#   * ".h5" / ".hdf5"  -> Bmad openPMD `grid_field` HDF5 (read_field_grid_hdf5 /
#                         write_field_grid_hdf5 in hdf5_grid_field.jl), so the file
#                         is also a valid Bmad grid_field file.
#   * anything else    -> a Julia source file (like ags-snakes/wsnk_fieldmap.jl)
#                         that, when `include`d, defines `fg::FieldGridTable`.
# ===========================================================================

# True if `path` should be treated as an HDF5 file (".h5" or ".hdf5" suffix).
_is_hdf5_path(path) = lowercase(splitext(path)[2]) in (".h5", ".hdf5")

"""
    read_field_grid(path) -> FieldGridTable

Load a field grid into a [`FieldGridTable`].  If `path` ends in `.h5`/`.hdf5` it is
read as a Bmad openPMD `grid_field` HDF5 file ([`read_field_grid_hdf5`]); otherwise
it is read as a Julia source file (`include`d) that defines `fg::FieldGridTable`.
"""
function read_field_grid(path::AbstractString)
    _is_hdf5_path(path) && return read_field_grid_hdf5(path)
    m = Module(:FieldGridInclude)
    Base.include(m, abspath(path))
    isdefined(m, :fg) ||
        error("Julia field-grid file did not define `fg`: $path")
    fg = getfield(m, :fg)
    fg isa FieldGridTable ||
        error("`fg` defined in $path is not a FieldGridTable.")
    return fg
end

"""
    write_field_grid(path, fg::FieldGridTable)

Write a [`FieldGridTable`].  If `path` ends in `.h5`/`.hdf5` it is written as a
Bmad openPMD `grid_field` HDF5 file ([`write_field_grid_hdf5`], readable by Bmad);
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
# HDF5 schema:
#   root datasets   : z_base, rms_plane, origin            (Float64[])
#   root attributes : g_ref, dz_grid (Float64); m_max, n_planes_add (Int);
#                     core_weight, outer_plane_weight (Float64); input_file (String)
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

_opt_attr(f, name) = haskey(attributes(f), name) ? read_attribute(f, name) : missing

"""
    gg_save_fit(path; z_base, a, b, bs, rms_plane, m_max, g_ref, origin, dz_grid,
                n_planes_add, core_weight, outer_plane_weight, input_file)

Write a `gg_fit` result to an HDF5 file readable by [`gg_load_fit`].
"""
function gg_save_fit(path::AbstractString; z_base, a, b, bs, rms_plane, m_max, g_ref,
                     origin, dz_grid, n_planes_add, core_weight, outer_plane_weight, input_file)
    h5open(path, "w") do f
        f["z_base"]    = collect(Float64, z_base)
        f["rms_plane"] = collect(Float64, rms_plane)
        f["origin"]    = collect(Float64, origin)
        attributes(f)["m_max"]              = Int(m_max)
        attributes(f)["g_ref"]              = Float64(g_ref)
        attributes(f)["dz_grid"]            = Float64(dz_grid)
        attributes(f)["n_planes_add"]       = Int(n_planes_add)
        attributes(f)["core_weight"]        = Float64(core_weight)
        attributes(f)["outer_plane_weight"] = Float64(outer_plane_weight)
        attributes(f)["input_file"]         = String(input_file)
        _write_coef_group(f, "a", a)
        _write_coef_group(f, "b", b)
        _write_coef_group(f, "bs", bs; single = true)
    end
    return path
end

"""
    gg_load_fit(path::AbstractString) -> NamedTuple

Load a `gg_fit` result HDF5 file (see [`gg_save_fit`]) into a NamedTuple with the
GG coefficient dictionaries `a`, `b`, `bs` and the associated metadata.
"""
function gg_load_fit(path::AbstractString)
    h5open(path, "r") do f
        return (; z_base   = read(f["z_base"]),
                  a        = _read_coef_group(f, "a"),
                  b        = _read_coef_group(f, "b"),
                  bs       = _read_coef_group(f, "bs"; single = true),
                  g_ref    = read_attribute(f, "g_ref"),
                  origin   = read(f["origin"]),
                  dz_grid  = read_attribute(f, "dz_grid"),
                  m_max    = Int(read_attribute(f, "m_max")),
                  rms_plane = read(f["rms_plane"]),
                  # Fit-control metadata (absent in older files → missing).
                  n_planes_add       = _opt_attr(f, "n_planes_add"),
                  core_weight        = _opt_attr(f, "core_weight"),
                  outer_plane_weight = _opt_attr(f, "outer_plane_weight"),
                  input_file         = _opt_attr(f, "input_file"))
    end
end
