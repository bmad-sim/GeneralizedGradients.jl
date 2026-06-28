"""
    run_gg_fit

Fit a 3D magnetic field table to generalized-gradient (GG) coefficients
a_n(z), b_n(z), b_s(z) and their z-derivatives, plane by plane.

## Usage

Run with command:
  julia programs/run_gg_fit.jl [parameter_input_file]
The parameter input file defines the field grid `field`, the transverse 
`origin`, and the fit-control parameters `n_planes_add`, `core_weight`, `outer_plane_weight`. 

See the documentation for the gg_fit function for more details.
"""

using OffsetArrays, LinearAlgebra, Printf, GeneralizedGradients

const PARAM_FILE = joinpath(pwd(), ARGS[1])

gg_fit(PARAM_FILE)