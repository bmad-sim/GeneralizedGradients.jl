# ---------------------------------------------------------------------------
# hdf5_grid_field.jl
#
# Read and write Bmad `grid_field` field maps in openPMD HDF5 format, matching
# Bmad's hdf5_write_grid_field.f90 / hdf5_read_grid_field.f90.  Currently
# supports `geometry = xyz` (rectangular) magnetic grids, which is what
# grid_to_bmad.jl produces.
#
# openPMD/HDF5 ordering note: HDF5.jl reverses dimensions on write (column-major,
# like Fortran), so an (nx, ny, nz) grid is written by reshaping to (nz, ny, nx)
# to land on disk with C-dims (nx, ny, nz) -- byte-identical to Bmad's Fortran
# writer.  Each dataset carries `gridDataOrder = "F"`, which Bmad's reader honors
# first (overriding axisLabels).  The reader here honors it the same way.
# ---------------------------------------------------------------------------

# openPMD complex field samples are a compound type with real ("r") and
# imaginary ("i") double members -- identical to Bmad's pmd_init_compound_complex.
struct ComplexPMD
    r::Float64
    i::Float64
end

# openPMD SI base-unit exponents (L, M, T, I, Theta, N, J) for Tesla.
const _DIM_TESLA = [0.0, 1, -2, -1, 0, 0, 0]

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

"""
    write_grid_field_hdf5(path, pt, lb, gf_r0, dr, g_ref, field_scale)

Write a magnetic `grid_field` as an openPMD HDF5 file matching Bmad's
`hdf5_write_grid_field` (geometry = xyz, field_type = magnetic).

  path         Output file path (conventionally ending in ".h5").
  pt           Field array; `pt[ix,iy,iz]` holds [Bx, By, Bz] (may be an OffsetArray).
  lb           Grid lower-bound index triple (ix_lo, iy_lo, iz_lo).
  gf_r0        Grid origin offset (x0, y0, z0).
  dr           Grid spacing (dx, dy, dz).
  g_ref            Bending strength g = 1/radius [1/m]. A non-zero `g_ref` means the
               reference curve is an arc, so gridCurvatureRadius is set to 1/g_ref
               (non-zero) and Bmad enables `curved_ref_frame`.
  field_scale  Overall field scale factor.
"""
function write_grid_field_hdf5(path, pt, lb, gf_r0, dr, g_ref, field_scale)
    ix_lo, iy_lo, iz_lo = lb
    nx, ny, nz = length.(axes(pt))

    # Build a 1-based (nx, ny, nz) complex array per component, then reshape to
    # reversed dims so HDF5.jl (column-major, dims-reversing) lays the dataset on
    # disk with C-dims (nx, ny, nz) and column-major data.
    comp(c) = reshape([ComplexPMD(pt[ix, iy, iz][c], 0.0)
                       for ix in ix_lo:ix_lo+nx-1, iy in iy_lo:iy_lo+ny-1,
                           iz in iz_lo:iz_lo+nz-1], (nz, ny, nx))

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
        a["gridGeometry"]        = "rectangular"
        a["fieldScale"]          = [Float64(field_scale)]
        a["componentFieldScale"] = [Float64(field_scale)]
        a["axisLabels"]          = ["z", "y", "x"]          # reversed -> data_order F
        a["eleAnchorPt"]         = "beginning"
        a["gridOriginOffset"]    = Float64[gf_r0...]
        a["gridSpacing"]         = Float64[dr...]
        a["harmonic"]            = Int32[0]
        a["interpolationOrder"]  = Int32[1]
        a["gridLowerBound"]      = Int32[ix_lo, iy_lo, iz_lo]
        a["gridSize"]            = Int32[nx, ny, nz]
        g_ref == 0 ? a["gridCurvatureRadius"] = [0.0] :  a["gridCurvatureRadius"] = Float64[1/g_ref]
        
        b = create_group(g1, "magneticField")
        for (i, name) in enumerate(("x", "y", "z"))
            b[name] = comp(i)
            da = attributes(b[name])
            da["gridDataOrder"] = "F"           # explicit; Bmad reader honors this first
            da["localName"]     = name
            da["unitSI"]        = [1.0]
            da["unitDimension"] = _DIM_TESLA
            da["unitSymbol"]    = "Tesla"
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

# Read an attribute if present, else return `default`.
_attr(obj, name, default) = haskey(attributes(obj), name) ? read_attribute(obj, name) : default

# Map a stored component dataset to a 1-based (nx, ny, nz) real array, applying
# the data order the same way Bmad does.  `R` is the array HDF5.jl returns (its
# column-major flat == the on-disk byte order).
function _order_component(R, nx, ny, nz, order)
    flat = real.(vec(R))
    if uppercase(string(order)) == "C"
        return permutedims(reshape(flat, (nz, ny, nx)), (3, 2, 1))
    else                                   # "F" (Bmad default) and anything else
        return reshape(flat, (nx, ny, nz))
    end
end

# Decide the data order for a dataset: explicit `gridDataOrder` wins, else infer
# from `axisLabels` (reversed labels => "F"), else default "F".
function _data_order(dset, axis_labels)
    haskey(attributes(dset), "gridDataOrder") && return read_attribute(dset, "gridDataOrder")
    axis_labels !== nothing && axis_labels == ["z", "y", "x"] && return "F"
    axis_labels !== nothing && axis_labels == ["x", "y", "z"] && return "C"
    return "F"
end

"""
    read_grid_field_hdf5(path; index = 1) -> NamedTuple

Read a Bmad/openPMD `grid_field` HDF5 file (as written by
`write_grid_field_hdf5`, or by Bmad itself) and return its contents:

  geometry          "rectangular" (only xyz is supported).
  field_type        :magnetic or :electric.
  ele_anchor_pt     "beginning" / "center" / "end".
  g_ref             Real. Curvilinear coordinates bending strength = 1/bend_radius
  field_scale       Real.
  r0                gridOriginOffset, 3-vector.
  dr                gridSpacing, 3-vector.
  lb                gridLowerBound, integer 3-vector.
  pt                OffsetArray; `pt[ix,iy,iz]` = [Fx, Fy, Fz], indexed from `lb`.

`index` selects which grid under `/ExternalFieldMesh/` to read (default 1).
"""
function read_grid_field_hdf5(path::AbstractString; index::Integer = 1)
    h5open(path, "r") do f
        haskey(f, "ExternalFieldMesh") ||
            error("Not a Bmad grid_field HDF5 file (missing /ExternalFieldMesh): $path")
        g1 = f["ExternalFieldMesh"][string(index)]

        geometry = _attr(g1, "gridGeometry", "rectangular")
        geometry == "rectangular" ||
            error("read_grid_field_hdf5 supports only 'rectangular' (xyz) grids, got: $geometry")

        lb  = Int.(read_attribute(g1, "gridLowerBound"))
        sz  = Int.(read_attribute(g1, "gridSize"))
        r0  = collect(Float64, read_attribute(g1, "gridOriginOffset"))
        dr  = collect(Float64, read_attribute(g1, "gridSpacing"))
        nx, ny, nz = sz[1], sz[2], sz[3]

        field_scale = first(_attr(g1, "fieldScale", _attr(g1, "componentFieldScale", [1.0])))
        anchor      = _attr(g1, "eleAnchorPt", "beginning")
        rho         = first(_attr(g1, "gridCurvatureRadius", [0.0]))
        axis_labels = haskey(attributes(g1), "axisLabels") ?
                      collect(String, read_attribute(g1, "axisLabels")) : nothing

        # Pick the field group (magnetic preferred, else electric).
        grp_name, ftype = haskey(g1, "magneticField") ? ("magneticField", :magnetic) :
                          haskey(g1, "electricField") ? ("electricField", :electric) :
                          error("Grid has neither magneticField nor electricField: $path")
        fg = g1[grp_name]

        comps = [zeros(Float64, nx, ny, nz) for _ in 1:3]
        for (i, name) in enumerate(("x", "y", "z"))
            haskey(fg, name) || continue   # missing component => zero field
            dset = fg[name]
            comps[i] = _order_component(read(dset), nx, ny, nz, _data_order(dset, axis_labels))
        end

        # Assemble pt[ix,iy,iz] = [Fx, Fy, Fz] as an OffsetArray indexed from lb.
        pt = OffsetArray([ [comps[1][a, b, c], comps[2][a, b, c], comps[3][a, b, c]]
                           for a in 1:nx, b in 1:ny, c in 1:nz ],
                         lb[1]:lb[1]+nx-1, lb[2]:lb[2]+ny-1, lb[3]:lb[3]+nz-1)

        rho == 0 ? g_ref = 0.0 : g_ref = 1/rho
        return (; geometry, field_type = ftype, ele_anchor_pt = anchor,
                  g_ref, field_scale, r0, dr, lb, pt)
    end
end
