# ---------------------------------------------------------------------------
# hdf5_grid_field.jl
#
# Read and write Bmad `grid_field` field maps (as a `FieldGridTable`) in openPMD
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

Write a `FieldGridTable` as an openPMD HDF5 `grid_field` file matching Bmad's
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
# In a Bmad grid_field file each component dataset is written Fortran-order
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
      error("grid_field dataset $name/$axis has size $(size(comp)), expected ($nx, $ny, $nz) " *
            "-- not a Bmad-format (Fortran-order) grid_field file.")
    comps[c] .= real.(comp)
  end
  field = [Float64[comps[1][a, b, k], comps[2][a, b, k], comps[3][a, b, k]]
      for a in 1:nx, b in 1:ny, k in 1:nz]
  return OffsetArray(field, lb[1]:lb[1]+nx-1, lb[2]:lb[2]+ny-1, lb[3]:lb[3]+nz-1)
end

"""
    read_field_grid_hdf5(path; index = 1) -> FieldGridTable

Read a Bmad/openPMD `grid_field` HDF5 file (as written by
`write_field_grid_hdf5`, or by Bmad itself) into a [`FieldGridTable`].  The
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
      error("Not a Bmad grid_field HDF5 file (missing /ExternalFieldMesh): $path")
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
