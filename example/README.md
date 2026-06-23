## Overview

This example starts with a field grid from which gg function values are created. 
From the gg function values, field and vector potential values are calculated.

Note: All commands are executed in the `example` directory.

## Preliminary: Create a field data file for use in this example

File created: `example/wsnk_fieldmap_reduced.jld2`
This is a reduced field map file containing the first 12 planes of the full field map.
This speeds things up.

Note: jld2 data files are actually larger than the Julia ASCII data files they come from. 
It is unclear why, however the speed at which data can be read in is much faster.

Created via:
```
julia> using JLD2, OffsetArrays
julia> include("ags-snakes/wsnk_fieldmap.jl")       # Full table 
julia> pt = OffsetArray(pt[:,:,0:11], 0, 0, -1);    # Truncate for test
julia> save("wsnk_fieldmap_reduced.jld2", Dict("r0_grid" => r0_grid, "dr_grid" =>  dr_grid, "pt" => pt))
```

To read back in use:
```
julia> field = load("wsnk_fieldmap_reduced.jld2")
```

## Create a GG fit file.

Fitting parameters for this example are in:
```
example/fit_params.jl
```
Fit:
```
julia ../src/gg_fit.jl fit_params.jl
```
The data file produced is `fit_params.jl`. This file will have the following parameters:
```
Dict{String, Any} with 14 entries:
  "outer_plane_weight" => 1                                          # Fit input parameter
  "rms_plane"          => [4.17469e-6, 6.421e-6,  …                  # Per plaine fit RMS  
  "b"                  => Dict((1, 2)=>[-0.00317784, -0.00341029, …  # b function fit values
  "m_max"              => 2                                          # max order
  "input_file"         => "/Users/dcs16/.julia/dev/GeneralizedGradients/example/fit_params.jl"
  "a"                  => Dict((1, 2)=>[-0.0046122, -0.00615161, …   # a function fit values
  "h"                  => 0                                          # Curvilinear curvature
  "core_weight"        => 1                                          # Fit input parameter
  "bs"                 => Dict(0=>[-2.23633e-7, -2.31509e-7, …       # bs function fit values
  "dz_grid"            => 0.005                                      # Fit values plane spacing
  "origin"             => [-0.0, 0.0]                                # Fit (x, y) origin
  "n_planes_add"       => 1                                          # Fit input parameter.
  "z_base"             => [0.0, 0.005, ...]                          # Fit plane values.
```

## Read in fit parameters.

The fit parameters are stored in a Dict with string keys. For faster evaluation, these parameters
are transfered to a Struct. This struct has the same components as the Dict (see above).
```
julia> include("../src/gg_eval.jl")
julia> fit = gg_load_fit("gg_fit_result.jld2");
julia> fit.dz_grid              # Returns 0.005
```

field-expansion coefficients at a given s-position:
```
julia> gg_coefficients_at_s(fit, 0.0)
```

generalized-gradient coefficients (a, b, bs) at a grid plane index or at a given s-position:
```
julia> gg_coefficients_at_plane(fit, 1)      # at grid plane 1
julia> gg_coefficients_at_s(fit, 0.0) # at s = 0.0
```
