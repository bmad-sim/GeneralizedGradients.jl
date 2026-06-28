"""
    run_grid_to_bmad

Read a 3D field grid and write it out in Bmad `grid_field` format, producing a
Bmad lattice element with the field grid attached.

## Usage

Run with command:
  julia programs/run_grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]

  <field_grid.h5>    Input field-grid file (read by `read_field_grid`).
  [output_base]      Base name for the output files. Default: input name without
                     extension.
  [g_ref]            Reference-curve bending strength = `1/bend_radius` [1/m].
                     Defaults to the input grid's `g_ref`. Non-zero => `sbend`.
  --hdf5             Write the field grid as an openPMD HDF5 binary file (.h5)
                     instead of a plain-text Bmad block.

See the documentation for the `grid_to_bmad` function for more details.
"""

using OffsetArrays, Printf, GeneralizedGradients

hdf5 = "--hdf5" in ARGS
args = filter(!startswith("--"), ARGS)
isempty(args) && error("Usage: julia programs/run_grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]")

input = args[1]
output_base = length(args) >= 2 ? args[2] :
              joinpath(dirname(input), first(splitext(basename(input))))
g_ref = length(args) >= 3 ? parse(Float64, args[3]) : nothing

grid_to_bmad(input; output_base, g_ref, hdf5)
