"""
    run_field_grid_to_bmad

Read a 3D field grid and write it out in Bmad `grid_field` format, producing a
Bmad lattice element with the field grid attached.

## Usage

Run with command:
  julia programs/run_field_grid_to_bmad.jl <field_grid.h5> [output_base] [--text]

  <field_grid.h5>    Input field-grid HDF5 file (read by `read_field_grid_hdf5`).
  [output_base]      Base name for the output files. Default: input name without
                     extension.
  --text             Write the field grid as a plain-text Bmad block instead of
                     the default openPMD HDF5 binary file (.h5).

The reference-curve bending strength is taken from the field grid's `g_ref`;
non-zero => the element is written as an `sbend`.

See the documentation for the `field_grid_to_bmad` function for more details.
"""

using OffsetArrays, Printf, GeneralizedGradients

text = "--text" in ARGS
args = filter(!startswith("--"), ARGS)
isempty(args) && error("Usage: julia programs/run_field_grid_to_bmad.jl <field_grid.h5> [output_base] [--text]")

input = args[1]
output_base = length(args) >= 2 ? args[2] :
              joinpath(dirname(input), first(splitext(basename(input))))

field_grid_to_bmad(input; output_base, hdf5 = !text)
