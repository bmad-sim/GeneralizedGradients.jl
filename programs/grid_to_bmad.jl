#!/usr/bin/env julia

"""
    grid_to_bmad.jl

Read in a 3D field grid and write it out in Bmad `grid_field` format, producing a
Bmad lattice element with the field grid attached.

## Usage

  julia programs/grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]

  <field_grid.h5>    Input field-grid file (HDF5; see "Input field grid" below).
  [output_base]      Base name for the output files. Default: input name without
                     extension. Two files are written:
                       <output_base>.bmad        -- the lattice element
                       <output_base>_grid.bmad   -- the grid_field block (text), or
                       <output_base>_grid.h5     -- the grid_field (HDF5, --hdf5)
  [g_ref]            Reference-curve bending "strength" = `1/bend_radius` [1/m].
                     Defaults to the input grid's `g_ref` (field.g_ref).
                     If g_ref is non-zero the reference curve is an arc and the
                     lattice element is written as an `sbend`; otherwise the
                     reference curve is straight and an `em_field` element is used.
  --hdf5             Write the grid_field as an openPMD HDF5 binary file (.h5)
                     instead of a plain-text Bmad block. Faster for Bmad to parse.

The program may also be `include`d to use `read_field_grid` and
`write_bmad_grid_field` directly.

## Input field grid

The input is an HDF5 field-grid file read by `read_field_grid` into a
`FieldGridTable` (curvilinear (x, y, z) coordinates):

  field.magnetic[c,ix,iy,iz]  Field components (c = 1,2,3 -> Bx, By, Bz)   [T]
                              (an OffsetArray; grid indices need not start at 0/1)
  field.r0                     Grid origin (x0, y0, z0)                      [m]
  field.dr                     Grid spacing (dx, dy, dz)                     [m]
  field.g_ref                  Curvilinear coordinates bending strength = `1/bending_radius` [1/m]

A grid point (ix, iy, iz) is at curvilinear position
  (x, y, z) = r0 + dr .* (ix, iy, iz).

## Output

A Bmad `grid_field` of `geometry = xyz`, `field_type = magnetic`, attached to a
lattice element via `grid_field = call::<output_base>_grid.bmad`.

The grid is anchored at the entrance of the element (`ele_anchor_pt = beginning`)
and the element length L is set to the longitudinal span of the grid,
L = dz * (iz_max - iz_min), so the grid fills the element from entrance to exit.
The grid_field `r0` keeps the transverse offset (x0, y0) of the input grid and
shifts z so the first grid plane sits at the element entrance.

For an arc reference curve (g_ref != 0) the element is an `sbend` with `g = g_ref` and the
field grid is expressed in the bend's curvilinear frame, so `curved_ref_frame = T`
is set. For a straight reference curve (g_ref == 0) the element is an `em_field`.
""" grid_to_bmad

using OffsetArrays, Printf, GeneralizedGradients

# `read_field_grid` (HDF5 field-grid reader) and `write_grid_field_hdf5` come
# from the GeneralizedGradients package.

# ---------------------------------------------------------------------------
# Write the Bmad grid_field
# ---------------------------------------------------------------------------

# Format a real for a Bmad lattice file: compact but lossless. `iszero` guard
# avoids printing a signed "-0".
_num(x::Real) = iszero(x) ? "0" : @sprintf("%.15g", float(x))

# Write the plain-text grid_field block from a (3, ix, iy, iz) magnetic OffsetArray,
# using the grid's own indices (grid origin `r0`, spacing `dr`, anchor = beginning).
function _write_grid_field_text(path, mag, r0, dr, is_bend, field_scale)
    ax = axes(mag)
    open(path, "w") do io
        println(io, "{")
        println(io, "  geometry = xyz,")
        println(io, "  field_type = magnetic,")
        println(io, "  ele_anchor_pt = beginning,")
        is_bend && println(io, "  curved_ref_frame = T,")
        field_scale != 1 && println(io, "  field_scale = ", _num(field_scale), ",")
        println(io, "  r0 = (", _num(r0[1]), ", ", _num(r0[2]), ", ", _num(r0[3]), "),")
        println(io, "  dr = (", _num(dr[1]), ", ", _num(dr[2]), ", ", _num(dr[3]), "),")
        println(io, "  {")
        for iz in ax[4], iy in ax[3], ix in ax[2]
            @printf(io, "    %d %d %d: %s %s %s,\n",
                    ix, iy, iz,
                    _num(mag[1, ix, iy, iz]), _num(mag[2, ix, iy, iz]), _num(mag[3, ix, iy, iz]))
        end
        println(io, "  }")
        println(io, "}")
    end
end

"""
    write_bmad_grid_field(field::FieldGridTable; ele_name, output_base, g_ref, field_scale)

Translate a field grid (a `FieldGridTable`, see `read_field_grid`) into Bmad
`grid_field` format. Writes two files and returns the path of the lattice-element file.

Keyword arguments:
  ele_name     Name of the Bmad lattice element. Default "fieldmap_ele".
  output_base  Base path for the two output files. Default `ele_name`.
  g_ref        Reference bending strength = `1/bending_radius` [1/m]. Default `field.g_ref`.
               Non-zero => the element is an `sbend`.
  field_scale  Overall field scale factor written to the grid_field. Default 1.
  hdf5         If true, write the grid_field as an openPMD HDF5 file
               (`<output_base>_grid.h5`) instead of a plain-text block.
"""
function write_bmad_grid_field(field::FieldGridTable;
                               ele_name::AbstractString = "fieldmap_ele",
                               output_base::AbstractString = ele_name,
                               g_ref::Real = field.g_ref,
                               field_scale::Real = 1.0,
                               hdf5::Bool = false)

    mag = field.magnetic
    dr  = field.dr
    dz  = dr[3]
    zax = axes(mag, 4)
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

    # ---- Write the grid_field --------------------------------------------
    if hdf5
        fg = FieldGridTable{Float64}(;
            magnetic = mag,                  # keep offset indices -> gridLowerBound preserved
            r0 = r0_out,
            dr = collect(Float64, dr),
            g_ref = float(g_ref),
            scale = float(field_scale),
            anchor_pt = GridAnchorPt.Beginning,
            geometry = GridGeometry.XYZ)
        write_grid_field_hdf5(grid_file, fg)
    else
        _write_grid_field_text(grid_file, mag, r0_out, dr, is_bend, field_scale)
    end

    # ---- Write the lattice element ---------------------------------------
    open(ele_file, "w") do io
        println(io, "! Bmad lattice element with attached field grid.")
        println(io, "! Generated from a field grid by grid_to_bmad.jl.")
        println(io, "!")
        if is_bend
            println(io, "! Reference curve is an arc (g = ", _num(g_ref),
                        " 1/m) => sbend; field grid is in the bend curvilinear frame.")
            println(io)
            println(io, ele_name, ": sbend,")
            println(io, "  l = ", _num(L), ",")
            println(io, "  g = ", _num(g_ref), ",")
        else
            println(io, "! Reference curve is straight => em_field.")
            println(io)
            println(io, ele_name, ": em_field,")
            println(io, "  l = ", _num(L), ",")
        end
        println(io, "  field_calc = fieldmap,")
        println(io, "  tracking_method = runge_kutta,")
        println(io, "  mat6_calc_method = tracking,")
        println(io, "  grid_field = call::", grid_name)
    end

    return ele_file
end

# ---------------------------------------------------------------------------
# Command-line driver
# ---------------------------------------------------------------------------

function main(args)
    hdf5 = "--hdf5" in args
    args = filter(!startswith("--"), args)
    isempty(args) && error("Usage: julia grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]")
    input = args[1]
    output_base = length(args) >= 2 ? args[2] :
                  joinpath(dirname(input), first(splitext(basename(input))))
    ele_name = basename(output_base)

    field = read_field_grid(input)
    g_ref = length(args) >= 3 ? parse(Float64, args[3]) : field.g_ref

    ele_file = write_bmad_grid_field(field; ele_name, output_base, g_ref, hdf5)

    nx, ny, nz = size(field.magnetic, 2), size(field.magnetic, 3), size(field.magnetic, 4)
    println("="^72)
    println("Field grid -> Bmad grid_field")
    println("  input file   : ", input)
    println("  grid size    : ", nx, " x ", ny, " x ", nz, "  (ix, iy, iz)")
    println("  reference    : ", g_ref == 0 ? "straight (em_field)" :
            @sprintf("arc, g = %.6g 1/m (sbend)", g_ref))
    println("  format       : ", hdf5 ? "HDF5 (openPMD)" : "text")
    println("  element      : ", ele_name)
    println("  lattice file : ", ele_file)
    println("  grid file    : ", output_base * (hdf5 ? "_grid.h5" : "_grid.bmad"))
    println("="^72)
    return ele_file
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
