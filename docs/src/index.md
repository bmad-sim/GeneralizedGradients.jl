# GeneralizedGradients.jl

Julia code to calculate and manipulate **generalized gradients** (GGs): functions
that compactly describe magnetic (and electric) fields. The package is currently
geared toward magnetic fields and supports both straight and curved (`sbend`)
reference frames.

The notation follows S. Van der Schueren *et al.*, *"Magnetic Field Modelling and
Symplectic Integration of Magnetic Fields on Curved Reference Frames for Improved
Synchrotron Design: First Steps"* (a copy is in the `papers` directory of the
repository). Here the **gg functions** are `a(s)`, `b(s)`, and `b_s(s)` together
with their `s`-derivatives.

```{toctree}
:hidden:
:caption: User Guide

guide/installation
guide/fitting
guide/evaluation
guide/bmad-export
guide/theory
```

## What it does

Starting from a 3D field grid (e.g. from a magnet solver or a measurement), the
package can:

1. **Fit** the grid to generalized gradients, plane by plane — see
   [Fitting a field grid](guide/fitting.md).
2. **Evaluate** the field, vector potential, and its derivatives anywhere from
   the fitted GGs — see [Evaluating the fitted field](guide/evaluation.md).
3. **Export** to [Bmad](https://www.classe.cornell.edu/bmad/) as either a
   `grid_field` or a `gen_grad_map` element — see
   [Exporting to Bmad](guide/bmad-export.md).

The complete docstring reference is in the **API Reference** (linked in the
sidebar).

## Quick example

```julia
using GeneralizedGradients

field  = read_field_grid_hdf5("wsnk_fieldmap_reduced.h5")  # a FieldGridTable
params = GGFitInputParams()
params.n_planes_add = 1
params.output_file  = "gg_fit_result.h5"

results = gg_fit(field, params)                # fit GGs plane by plane
gg_fit_show_results(results, field, params)    # print a summary
gg_fit_write_results(results, field, params)   # save to HDF5

# Convert the fit to a Bmad gen_grad_map element:
gg_to_bmad("gg_fit_result.h5")
```
