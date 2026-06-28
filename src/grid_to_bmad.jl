# ---------------------------------------------------------------------------
# grid_to_bmad.jl
#
# Read a 3D field grid and write it out in Bmad `grid_field` format (a lattice
# element with the field grid attached).  `grid_to_bmad` is the public function;
# programs/run_grid_to_bmad.jl is a shell wrapper.
# ---------------------------------------------------------------------------

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

"""
    write_bmad_field_grid(field::FieldGridTable; ele_name, output_base, g_ref, field_scale, hdf5)

Translate a field grid (a `FieldGridTable`, see `read_field_grid`) into Bmad
`grid_field` format. Writes two files and returns the path of the lattice-element file.

Keyword arguments:
  ele_name     Name of the Bmad lattice element. Default "fieldmap_ele".
  output_base  Base path for the two output files. Default `ele_name`.
  g_ref        Reference bending strength = `1/bending_radius` [1/m]. Default `field.g_ref`.
               Non-zero => the element is an `sbend`.
  field_scale  Overall field scale factor written to the field grid. Default 1.
  hdf5         If true, write the field grid as an openPMD HDF5 file
               (`<output_base>_grid.h5`) instead of a plain-text block.
"""
function write_bmad_field_grid(field::FieldGridTable;
                               ele_name::AbstractString = "fieldmap_ele",
                               output_base::AbstractString = ele_name,
                               g_ref::Real = field.g_ref,
                               field_scale::Real = 1.0,
                               hdf5::Bool = false)

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
    println(io, "! Generated from a field grid by grid_to_bmad.")
    println(io, "!")
    if is_bend
      println(io, "! Reference curve is an arc (g = ", _grid_num(g_ref),
            " 1/m) => sbend; field grid is in the bend curvilinear frame.")
      println(io)
      println(io, ele_name, ": sbend,")
      println(io, "  l = ", _grid_num(L), ",")
      println(io, "  g = ", _grid_num(g_ref), ",")
    else
      println(io, "! Reference curve is straight => em_field.")
      println(io)
      println(io, ele_name, ": em_field,")
      println(io, "  l = ", _grid_num(L), ",")
    end
    println(io, "  field_calc = fieldmap,")
    println(io, "  tracking_method = runge_kutta,")
    println(io, "  mat6_calc_method = tracking,")
    println(io, "  grid_field = call::", grid_name)
  end

  return ele_file
end

"""
    grid_to_bmad(input; output_base, g_ref, hdf5) -> lattice_file_path

Read a 3D field grid and write it out in Bmad `grid_field` format, producing a
Bmad lattice element with the field grid attached.

## Usage

As a function:
```
using GeneralizedGradients
grid_to_bmad("field_grid.h5")
```
From the shell (see `programs/run_grid_to_bmad.jl`):
```
julia programs/run_grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]
```

Arguments:
  input        Input field-grid file (read by `read_field_grid` into a `FieldGridTable`).
  output_base  Base name for the output files. Default: input name without extension.
               Two files are written:
                 <output_base>.bmad        -- the lattice element
                 <output_base>_grid.bmad   -- the field grid block (text), or
                 <output_base>_grid.h5     -- the field grid (HDF5, `hdf5 = true`)
  g_ref        Reference-curve bending "strength" = `1/bend_radius` [1/m]. Defaults
               to the input grid's `g_ref`. If non-zero the reference curve is an arc
               and the element is written as an `sbend`; otherwise it is an `em_field`.
  hdf5         Write the field grid as an openPMD HDF5 binary file (.h5) instead of a
               plain-text Bmad block. Faster for Bmad to parse.

## Output

A Bmad `grid_field` of `geometry = xyz`, `field_type = magnetic`, attached to a
lattice element via `grid_field = call::<output_base>_grid.bmad`.  The grid is
anchored at the entrance of the element (`ele_anchor_pt = beginning`) and the
element length L = dz*(nz-1) so the grid fills the element from entrance to exit.
The field-grid `r0` keeps the transverse offset (x0, y0) of the input grid and
shifts z so the first grid plane sits at the element entrance.
"""
function grid_to_bmad(input::AbstractString;
                      output_base::AbstractString =
                          joinpath(dirname(input), first(splitext(basename(input)))),
                      g_ref::Union{Real,Nothing} = nothing,
                      hdf5::Bool = false)
  ele_name = basename(output_base)
  field = read_field_grid(input)
  gref = g_ref === nothing ? field.g_ref : g_ref

  ele_file = write_bmad_field_grid(field; ele_name, output_base, g_ref = gref, hdf5)

  nx, ny, nz = size(field.magnetic, 1), size(field.magnetic, 2), size(field.magnetic, 3)
  println("="^72)
  println("Field grid -> Bmad grid_field")
  println("  input file   : ", input)
  println("  grid size    : ", nx, " x ", ny, " x ", nz, "  (ix, iy, iz)")
  println("  reference    : ", gref == 0 ? "straight (em_field)" :
      @sprintf("arc, g = %.6g 1/m (sbend)", gref))
  println("  format       : ", hdf5 ? "HDF5 (openPMD)" : "text")
  println("  element      : ", ele_name)
  println("  lattice file : ", ele_file)
  println("  grid file    : ", output_base * (hdf5 ? "_grid.h5" : "_grid.bmad"))
  println("="^72)
  return ele_file
end
