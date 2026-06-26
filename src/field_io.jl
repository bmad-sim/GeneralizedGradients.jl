# ---------------------------------------------------------------------------
# field_io.jl
#
# HDF5 storage for the project's two native data products:
#   * field grids       -- read_field_grid / write_field_grid
#   * GG fit results     -- gg_load_fit / gg_save_fit
#
# These replace the former JLD2 `load`/`save`/`jldsave` storage.  Plain HDF5 has
# no notion of Julia Dicts or OffsetArrays, so the schemas below store the index
# bounds / dictionary keys explicitly alongside the numeric data.
# ---------------------------------------------------------------------------

using HDF5, OffsetArrays

# ===========================================================================
# Field grid
#
# HDF5 schema:
#   root attributes : g_ref (Float64, bending strength = 1/rho; 0 for straight)
#   root datasets   : r0_grid     (Float64[3])      grid origin
#                     dr_grid     (Float64[3])      grid spacing
#                     lower_bound (Int[3])          (ix_lo, iy_lo, iz_lo)
#                     B           (Float64[3,nx,ny,nz])  B[:,a,b,c] = [Bx,By,Bz]
# ===========================================================================

"""
    read_field_grid(path) -> Dict

Load a field-grid HDF5 file and sanity check it. Returns a Dict with keys
`"r0_grid"`, `"dr_grid"`, `"g_ref"`, and `"pt"`, where `pt[ix,iy,iz]` is the
`[Bx,By,Bz]` 3-vector at that grid point (an OffsetArray indexed from the stored
lower bound).
"""
function read_field_grid(path::AbstractString)
    h5open(path, "r") do f
        for name in ("r0_grid", "dr_grid", "B", "lower_bound")
            haskey(f, name) || error("Field grid file is missing the \"$name\" dataset: $path")
        end
        r0_grid = collect(Float64, read(f["r0_grid"]))
        dr_grid = collect(Float64, read(f["dr_grid"]))
        lb      = Int.(read(f["lower_bound"]))
        B       = read(f["B"])                       # (3, nx, ny, nz)
        g_ref   = haskey(attributes(f), "g_ref") ? read_attribute(f, "g_ref") : 0.0

        length(r0_grid) == 3 || error("\"r0_grid\" must be a 3-vector.")
        length(dr_grid) == 3 || error("\"dr_grid\" must be a 3-vector.")
        (ndims(B) == 4 && size(B, 1) == 3) ||
            error("\"B\" must have shape (3, nx, ny, nz).")
        nx, ny, nz = size(B, 2), size(B, 3), size(B, 4)

        pt = OffsetArray([Float64[B[1, a, b, c], B[2, a, b, c], B[3, a, b, c]]
                          for a in 1:nx, b in 1:ny, c in 1:nz],
                         lb[1]:lb[1]+nx-1, lb[2]:lb[2]+ny-1, lb[3]:lb[3]+nz-1)

        return Dict{String,Any}("r0_grid" => r0_grid, "dr_grid" => dr_grid,
                                "g_ref" => g_ref, "pt" => pt)
    end
end

"""
    write_field_grid(path; r0_grid, dr_grid, pt, g_ref = 0.0)

Write a field grid to an HDF5 file readable by [`read_field_grid`].  `pt[ix,iy,iz]`
must be a `[Bx,By,Bz]` 3-vector (an array of 3-vectors; may be an OffsetArray).
"""
function write_field_grid(path::AbstractString; r0_grid, dr_grid, pt, g_ref::Real = 0.0)
    ix_lo, iy_lo, iz_lo = first.(axes(pt))
    nx, ny, nz = length.(axes(pt))
    B = Array{Float64}(undef, 3, nx, ny, nz)
    for (c, iz) in enumerate(iz_lo:iz_lo+nz-1), (b, iy) in enumerate(iy_lo:iy_lo+ny-1),
        (a, ix) in enumerate(ix_lo:ix_lo+nx-1)
        B[:, a, b, c] = pt[ix, iy, iz]
    end
    h5open(path, "w") do f
        attributes(f)["g_ref"] = Float64(g_ref)
        f["r0_grid"]     = Float64[r0_grid...]
        f["dr_grid"]     = Float64[dr_grid...]
        f["lower_bound"] = Int[ix_lo, iy_lo, iz_lo]
        f["B"]           = B
    end
    return path
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
