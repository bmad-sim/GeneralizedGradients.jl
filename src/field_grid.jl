# ===========================================================================
# Bmad grid_field export
#
# Read a 3D field grid and write it out in Bmad `grid_field` format (a lattice
# element with the field grid attached). `write_bmad_field_grid` is the public
# function; programs/run_write_bmad_field_grid.jl is a shell wrapper.
# The underscore-prefixed helper functions used here live in src/helpers.jl.
# ===========================================================================

"""
    write_bmad_field_grid(field; ele_name, output_base, field_scale, hdf5)

Write a field grid table in Bmad `grid_field` format (a lattice element with the field grid attached).
Also write the field grid to a file in HDF5 or ASCII format.

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
#
# The underscore-prefixed read/write helpers used below live in src/helpers.jl.
# ---------------------------------------------------------------------------

# openPMD complex field samples are a compound type with real ("r") and
# imaginary ("i") double members -- identical to Bmad's pmd_init_compound_complex.
struct ComplexPMD
  r::Float64
  i::Float64
end

# The unit-exponent constants `_DIM_TESLA` / `_DIM_VPERM` are defined in
# GeneralizedGradients.jl.

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------
# read_field_grid_hdf5

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
