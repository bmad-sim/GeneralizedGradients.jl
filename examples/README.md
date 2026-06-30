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

Note: To read back in use:
```
julia> field = read_field_grid_hdf5("wsnk_fieldmap_reduced.h5")   # a FieldGridTable
```

## Create a GG fit file.

An example fit run:
```
julia> include("run_gg_fit.jl")
```
See the `run_gg_fit.jl` and `src/gg_fit.jl` for documentation on the fit.
The data file produced is `gg_fit_result.h5` (HDF5 format). 

## Read in fit parameters.

To load the fit use
```
julia> fit, meta = read_gg_fit("gg_fit_result.h5")
```

which returns a two-tuple. `fit` is a `GGCoefs` struct with the fields:
```
  rms_plane            [4.17469e-6, 6.421e-6,  …                  # Per plane fit RMS
  a                    Dict((1, 2)=>[-0.0046122, -0.00615161, …   # a function fit values
  b                    Dict((1, 2)=>[-0.00317784, -0.00341029, …  # b function fit values
  bs                   Dict(0=>[-2.23633e-7, -2.31509e-7, …       # bs function fit values
  m_max                2                                          # max order
  g_ref                0                                          # Curvilinear curvature
  z_base               [0.0, 0.005, ...]                          # Fit plane values.
```
and `meta` is a NamedTuple of the fit metadata:
```
  outer_plane_weight   1                                          # Fit input parameter
  core_weight          1                                          # Fit input parameter
  dz_grid              0.005                                      # Fit values plane spacing
  origin               [-0.0, 0.0]                                # Fit (x, y) origin
  n_planes_add         1                                          # Fit input parameter.
```

## Write Field Table to Bmad

```
write_bmad_field_grid("gg_fit_result.h5"
```

## GG Coefficient Manipulation

field-expansion coefficients at a given s-position:
```
julia> gg_coefficients_at_s(fit, meta, 0.0)
```

generalized-gradient coefficients (a, b, bs) at a grid plane index or at a given s-position:
```
julia> gg_coefficients_at_plane(fit, meta, 1)      # at grid plane 1
julia> gg_coefficients_at_s(fit, meta, 0.0) # at s = 0.0
```
