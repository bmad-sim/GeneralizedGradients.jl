"""
    run_write_bmad_gg_fit

Convert generalized-gradient (GG) coefficients produced by `gg_fit` into Bmad
`gen_grad_map` format, producing a Bmad lattice element with the GG map attached.

## Usage

Run with command:
  julia programs/run_write_bmad_gg_fit.jl <gg_fit_result.h5> [output_base] [cutoff]

  <gg_fit_result.h5>  Input GG-fit file (output of gg_fit).
  [output_base]       Base name for the output files. Default: input name
                      without extension.
  [cutoff]            Relative magnitude cutoff for pruning negligible multipole
                      curves. Default 0 (keep every non-zero curve).

See the documentation for the `write_bmad_gg_fit` function for more details.
"""

using OffsetArrays, LinearAlgebra, Printf, GeneralizedGradients

isempty(ARGS) && error("Usage: julia programs/run_write_bmad_gg_fit.jl <gg_fit_result.h5> [output_base] [cutoff]")

input = ARGS[1]
output_base = length(ARGS) >= 2 ? ARGS[2] :
              joinpath(dirname(input), first(splitext(basename(input))))
cutoff = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0

write_bmad_gg_fit(input; output_base, cutoff)
