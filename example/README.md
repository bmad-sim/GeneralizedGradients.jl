## Overview

This example starts with a field grid from which gg function values are created. 
From the gg function values, field and vector potential values are calculated.

Note: All commands are executed in the `example` directory.

## Preliminary: Create a field data file for use in this example

File created: `example/wsnk_fieldmap_reduced.h5`

To speed things up, this is a reduced field map file containing the first 12 planes of the full 
field map for the AGS warm snake
Field grids and GG fit results are stored as HDF5 files.

Created via:
```
julia> using GeneralizedGradients, OffsetArrays
julia> field_file = "../ags-snakes/wsnk_fieldmap.hdf5"
julia> fg = read_field_grid_hdf5(field_file);            # full field map
julia> m = fg.magnetic;                                  # m[ix,iy,iz] = [Bx,By,Bz]
julia> fg.magnetic = OffsetArray(m[:, :, 0:11], axes(m, 1), axes(m, 2), 0:11);   # keep first 12 z-planes
julia> write_field_grid_hdf5("wsnk_fieldmap_reduced.h5", fg)
```

To read back in use:
```
julia> field = read_field_grid("wsnk_fieldmap_reduced.h5")   # a FieldGridTable
```

## Create a GG fit file.

Fitting parameters for this example are in:
```
example/fit_params.jl
```
To run the fit use the command:
```
julia ../programs/run_gg_fit.jl fit_params.jl
```
or, from Julia, `using GeneralizedGradients; gg_fit("fit_params.jl")`.
The data file produced is `gg_fit_result.h5` (HDF5). Loaded with `gg_load_fit`, it
yields a NamedTuple with the following fields:
```
  outer_plane_weight   1                                          # Fit input parameter
  rms_plane            [4.17469e-6, 6.421e-6,  …                  # Per plane fit RMS
  b                    Dict((1, 2)=>[-0.00317784, -0.00341029, …  # b function fit values
  m_max                2                                          # max order
  input_file           ".../example/fit_params.jl"
  a                    Dict((1, 2)=>[-0.0046122, -0.00615161, …   # a function fit values
  g_ref                0                                          # Curvilinear curvature
  core_weight          1                                          # Fit input parameter
  bs                   Dict(0=>[-2.23633e-7, -2.31509e-7, …       # bs function fit values
  dz_grid              0.005                                      # Fit values plane spacing
  origin               [-0.0, 0.0]                                # Fit (x, y) origin
  n_planes_add         1                                          # Fit input parameter.
  z_base               [0.0, 0.005, ...]                          # Fit plane values.
```

## Read in fit parameters.

`gg_load_fit` reads the HDF5 fit file into a NamedTuple (fields as above):
```
julia> using GeneralizedGradients
julia> fit = gg_load_fit("gg_fit_result.h5");
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
