# Fitting a field grid to generalized gradients

The central operation is `gg_fit`, which fits a 3D magnetic field grid to
generalized-gradient functions `a_n(s)`, `b_n(s)`, `b_s(s)` and their
`s`-derivatives, **plane by plane**.

## 1. Read a field grid

A field grid is held in a `FieldGridTable`. Read one from a Bmad openPMD
`field_grid` HDF5 file with `read_field_grid_hdf5`:

```julia
using GeneralizedGradients
field = read_field_grid_hdf5("examples/wsnk_fieldmap_reduced.h5")
```

In a `FieldGridTable`, `field.magnetic[ix, iy, iz]` is the `[Bx, By, Bz]`
3-vector at the grid point whose position is

```
(x, y, z) = field.r0 .+ field.dr .* (ix, iy, iz)
```

A non-zero `field.g_ref` (= `1 / bending_radius`) marks a curved (curvilinear)
reference frame.

## 2. Set the fit parameters

Fit controls live in a `GGFitInputParams` struct:

```julia
params = GGFitInputParams()
params.origin             = [0.0, 0.0]   # (x, y) axis the GGs are expanded about
params.n_planes_add       = 1            # z-planes added either side of the base plane
params.core_weight        = 1            # up-weight near-axis points (1 = uniform)
params.outer_plane_weight = 1            # weight of the outer z-planes (1 = uniform)
params.output_file        = "gg_fit_result.h5"
```

If `field.g_ref` is non-zero, `origin` must be `[0, 0]`. The maximum derivative
order resolved is `m_max = 2 * n_planes_add`. See [Theory](theory.md) for what
the weights do.

## 3. Run the fit

```julia
results = gg_fit(field, params)
```

`results` is a `GGFitResults` holding the fitted coefficient functions
(`a`, `b`, `bs`) sampled at every base plane (`z_base`), along with
the per-plane weighted-RMS residuals (`rms_plane`) and `m_max`.

## 4. Inspect and save

Print a human-readable summary (fit settings, per-plane residuals, leading
multipoles at the central plane):

```julia
gg_fit_show_results(results, field, params)
```

Write the result to an HDF5 file (readable later by `gg_load_fit`, and the input
to the Bmad exporters):

```julia
gg_fit_write_results(results, field, params)   # writes params.output_file
```

## Putting it together

The complete script lives at `examples/run_gg_fit.jl`:

```julia
using GeneralizedGradients

field = read_field_grid_hdf5("wsnk_fieldmap_reduced.h5")

params = GGFitInputParams()
params.n_planes_add = 1
params.output_file  = "gg_fit_result.h5"

results = gg_fit(field, params)
gg_fit_show_results(results, field, params)
gg_fit_write_results(results, field, params)
```

```{tip}
The fit does not strictly require a rectangular, evenly spaced grid — the merit
function is a sum over field points — but the current `gg_fit` assumes the GG
functions are sampled on the grid's own `z`-planes.
```

Next: [evaluate the fitted field](evaluation.md) or
[export it to Bmad](bmad-export.md).
