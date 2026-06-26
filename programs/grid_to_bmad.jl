#!/usr/bin/env julia

"""
    grid_to_bmad.jl

Read in a 3D field grid and write it out in Bmad `grid_field` format, producing a
Bmad lattice element with the field grid attached.

## Usage

  julia programs/grid_to_bmad.jl <field_grid.jld2> [output_base] [g_ref] [--hdf5]

  <field_grid.jld2>  Input field-grid file (see "Input field grid" below).
  [output_base]      Base name for the output files. Default: input name without
                     extension. Two files are written:
                       <output_base>.bmad        -- the lattice element
                       <output_base>_grid.bmad   -- the grid_field block (text), or
                       <output_base>_grid.h5     -- the grid_field (HDF5, --hdf5)
  [g_ref]            Reference-curve bending "strength" = `1/bend_radius` [1/m]. Default 0,
                     or the value of field["g_ref"] if present in the input file.
                     If g_ref is non-zero the reference curve is an arc and the
                     lattice element is written as an `sbend`; otherwise the
                     reference curve is straight and an `em_field` element is used.
  --hdf5             Write the grid_field as an openPMD HDF5 binary file (.h5)
                     instead of a plain-text Bmad block. Faster for Bmad to parse.

The program may also be `include`d to use `read_field_grid` and
`write_bmad_grid_field` directly.

## Input field grid

The input is a JLD2 file holding a field-grid Dict with the same layout used by
`gg_fit.jl` (curvilinear (x, y, z) coordinates):

  field["r0_grid"]         Grid origin, 3-vector  (x0, y0, z0)        [m]
  field["dr_grid"]         Grid spacing, 3-vector (dx, dy, dz)        [m]
  field["pt"][ix,iy,iz]    Magnetic field at the point, [Bx, By, Bz] [T]
  field["g_ref"]           (optional) curvilinear coordinates bending strength = `1/bending_radius` [1/m]

A point field["pt"][ix,iy,iz] is at curvilinear position
  (x, y, z) = r0_grid + dr_grid .* (ix, iy, iz).
`pt` may be an OffsetArray (indices need not start at 1).

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

using JLD2, OffsetArrays, Printf

# HDF5 grid_field reading/writing lives in the package source.
include(joinpath(@__DIR__, "..", "src", "hdf5_grid_field.jl"))

# ---------------------------------------------------------------------------
# Read a field grid
# ---------------------------------------------------------------------------

"""
    read_field_grid(path) -> Dict

Load a field-grid file and sanity check it. Returns the loaded Dict with keys
"r0_grid", "dr_grid", "pt", and (optionally) "g_ref".
"""
function read_field_grid(path::AbstractString)
    field = load(path)
    for key in ("r0_grid", "dr_grid", "pt")
        haskey(field, key) || error("Field grid file is missing the \"$key\" entry: $path")
    end
    length(field["r0_grid"]) == 3 || error("\"r0_grid\" must be a 3-vector.")
    length(field["dr_grid"]) == 3 || error("\"dr_grid\" must be a 3-vector.")
    ndims(field["pt"]) == 3 || error("\"pt\" must be a 3-dimensional array (indexed by ix, iy, iz).")
    return field
end

# ---------------------------------------------------------------------------
# Write the Bmad grid_field
# ---------------------------------------------------------------------------

# Format a real for a Bmad lattice file: compact but lossless. `iszero` guard
# avoids printing a signed "-0".
_num(x::Real) = iszero(x) ? "0" : @sprintf("%.15g", float(x))

# Write the plain-text grid_field block. `lb` = (ix_lo, iy_lo, iz_lo).
function _write_grid_field_text(path, pt, lb, gf_r0, dr, is_bend, field_scale)
    ix_lo, iy_lo, iz_lo = lb
    ix_hi, iy_hi, iz_hi = last.(axes(pt))
    open(path, "w") do io
        println(io, "{")
        println(io, "  geometry = xyz,")
        println(io, "  field_type = magnetic,")
        println(io, "  ele_anchor_pt = beginning,")
        is_bend && println(io, "  curved_ref_frame = T,")
        field_scale != 1 && println(io, "  field_scale = ", _num(field_scale), ",")
        println(io, "  r0 = (", _num(gf_r0[1]), ", ", _num(gf_r0[2]), ", ", _num(gf_r0[3]), "),")
        println(io, "  dr = (", _num(dr[1]), ", ", _num(dr[2]), ", ", _num(dr[3]), "),")
        println(io, "  {")
        for iz in iz_lo:iz_hi, iy in iy_lo:iy_hi, ix in ix_lo:ix_hi
            B = pt[ix, iy, iz]
            @printf(io, "    %d %d %d: %s %s %s,\n",
                    ix, iy, iz, _num(B[1]), _num(B[2]), _num(B[3]))
        end
        println(io, "  }")
        println(io, "}")
    end
end

"""
    write_bmad_grid_field(field; ele_name, output_base, g_ref, field_scale)

Translate a field-grid Dict (see `read_field_grid`) into Bmad `grid_field`
format. Writes two files and returns the path of the lattice-element file.

Keyword arguments:
  ele_name     Name of the Bmad lattice element. Default "fieldmap_ele".
  output_base  Base path for the two output files. Default `ele_name`.
  g_ref        Reference bending strength = `1/bending_radius` [1/m]. Default `field["g_ref"]`
               if present, else 0. Non-zero => the element is an `sbend`.
  field_scale  Overall field scale factor written to the grid_field. Default 1.
  hdf5         If true, write the grid_field as an openPMD HDF5 file
               (`<output_base>_grid.h5`) instead of a plain-text block.
"""
function write_bmad_grid_field(field::AbstractDict;
                               ele_name::AbstractString = "fieldmap_ele",
                               output_base::AbstractString = ele_name,
                               g_ref::Real = get(field, "g_ref", 0.0),
                               field_scale::Real = 1.0,
                               hdf5::Bool = false)

    r0 = field["r0_grid"]
    dr = field["dr_grid"]
    pt = field["pt"]

    ix_lo, ix_hi = first(axes(pt, 1)), last(axes(pt, 1))
    iy_lo, iy_hi = first(axes(pt, 2)), last(axes(pt, 2))
    iz_lo, iz_hi = first(axes(pt, 3)), last(axes(pt, 3))

    dx, dy, dz = dr[1], dr[2], dr[3]
    is_bend = g_ref != 0
    L = dz * (iz_hi - iz_lo)                 # longitudinal span of the grid

    # Anchor the grid at the entrance of the element: choose the grid_field z
    # origin so the first plane (iz_lo) lands at element z = 0. The transverse
    # offset (x0, y0) of the input grid is preserved.
    z0 = -dz * iz_lo
    gf_r0 = (r0[1], r0[2], z0)

    grid_file = output_base * (hdf5 ? "_grid.h5" : "_grid.bmad")
    ele_file  = output_base * ".bmad"
    grid_name = basename(grid_file)

    # ---- Write the grid_field --------------------------------------------
    lb = (ix_lo, iy_lo, iz_lo)
    if hdf5
        write_grid_field_hdf5(grid_file, pt, lb, gf_r0, (dx, dy, dz),
                              g_ref, field_scale)
    else
        _write_grid_field_text(grid_file, pt, lb, gf_r0, (dx, dy, dz),
                               is_bend, field_scale)
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
    isempty(args) && error("Usage: julia grid_to_bmad.jl <field_grid.jld2> [output_base] [g_ref] [--hdf5]")
    input = args[1]
    output_base = length(args) >= 2 ? args[2] :
                  joinpath(dirname(input), first(splitext(basename(input))))
    ele_name = basename(output_base)

    field = read_field_grid(input)
    g_ref = length(args) >= 3 ? parse(Float64, args[3]) : get(field, "g_ref", 0.0)

    ele_file = write_bmad_grid_field(field; ele_name, output_base, g_ref, hdf5)

    pt = field["pt"]
    nx, ny, nz = length.(axes(pt))
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
