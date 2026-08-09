# Fitting a field grid to generalized gradients

The central operation is `gg_calc_fit`, which fits a 3D magnetic field grid to
generalized-gradient functions `a_m(s)`, `b_m(s)`, `b_s(s)` and their
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
params.nd_max_for_m       = Dict()       # per-m derivative limit, e.g. Dict(4 => 2, 5 => 1)
params.exclude_functions  = []           # GG functions to leave out, e.g. [(:a, 2), (:bs, 0)]
params.prune_ave_limit    = 0            # drop GG functions with negligible field (0 = off)
params.prune_max_limit    = 0            # ... judged on ave and/or max contribution
params.output_file        = "gg_fit_result.h5"
```

If `field.g_ref` is non-zero, `origin` must be `[0, 0]`. The maximum derivative
order resolved is `nd_max = 2 * n_planes_add`. See [Theory](theory.md) for what
the weights do.

`nd_max` is one cut across every multipole order. Since `a_m`/`b_m` fall off as
`r^m`, the high-`m` functions are sampled meaningfully only near the aperture,
and their high derivative orders are the worst-conditioned unknowns in the fit.
`nd_max_for_m` holds them to a lower order without touching the low-`m`
functions, keyed by multipole order with `0` standing for `b_s`:

```julia
params.nd_max       = 4
params.nd_max_for_m = Dict(4 => 2, 5 => 1, 0 => 0)
```

An `m` not listed is cut by `nd_max` alone, and a listed limit can only tighten —
the effective cut is `min(nd_max, nd_max_for_m[m])`.

`m_max` can only cut multipole orders off the top of the range. Two settings
remove them from anywhere in it, for when a magnet's symmetry makes whole orders
useless:

- `exclude_functions` names them outright, as `(:a, m)`, `(:b, m)` or `(:bs, 0)`
  tuples. They are dropped as the unknowns are assembled, so they are never
  fitted at all.
- `prune_ave_limit` and `prune_max_limit` let the fit decide, dropping any
  function whose field contribution falls below **every** limit in force and
  refitting what is left. Both are fractions of the field table's mean `|B|`, and
  `0` switches a test off.

Either way the surviving coefficients are stored sparsely in `m`, and every
consumer treats a missing key as an identically zero function. See the `gg_calc_fit`
docstring for the full rules.

## 3. Run the fit

```julia
gg_fit = gg_calc_fit(field, params)
```

`gg_fit` is a `GGCoefs` holding the fitted coefficient functions
(`a`, `b`, `bs`) sampled at every base plane (`z_base`), along with
the per-plane weighted-RMS residuals (`rms_weighted_plane`), `m_max`, `nd_max`, and the
reference-frame bending strength `g_ref`.

## 4. Inspect and save

Print a human-readable summary (fit settings, per-plane residuals, leading
multipoles at the central plane):

```julia
gg_show_fit_results(gg_fit, field, params)
```

Write the result to an HDF5 file (readable later by `read_gg_fit`, and the input
to the Bmad exporters):

```julia
write_gg_fit(gg_fit, field, params)   # writes params.output_file
```

## 5. Diagnose a bad fit

A large RMS residual on some plane does not say what to do about it. Adding GG
terms helps only if the residual is something the expansion can represent, and
that is what `gg_show_fit_residuals` measures:

```julia
gg_show_fit_residuals(gg_fit, field; detail = [12])
```

Per base plane it reports

- **`out%`** — the share of the squared residual coming from the corners of the
  grid, outside the largest circle the expansion is well posed on. On a square
  grid the corners sit `√2` further out than the inscribed radius, where every
  multipole is largest and the series least convergent, so a plane's RMS can be
  almost entirely made of them while the fit is good everywhere it is meant to be
  used. Read this column first.
- **`rough%`** — how much of the residual inside that circle is point-to-point
  irregular rather than smooth. Rough is noise in the field table and will not
  improve however many terms are added; smooth is model error and will.
- **`top harmonic`** — the largest azimuthal harmonic `m` of the residual and its
  measured radial exponent `p`. A missing multipole of order `m` gives `p = m-1`
  in `B_r`/`B_θ` and `p = m` in `B_s`; when they match, raising `m_max` (or
  lifting an `nd_max_for_m` cap) removes it.
- **`divB`, `curlB`** — the field table's own violation of Maxwell's equations,
  as a multiple of what the fitted GG field shows under the same finite
  differences. A GG expansion is Maxwellian by construction, so anything the
  table violates by is a floor no fit reaches below.
- **`d²B/dz²`** — the second difference of the table along `z`. A plane orders of
  magnitude above its neighbours is a seam or a bad plane in the map.

The check none of this replaces is the direct one: refit with a larger `m_max`
and see whether the plane's residual actually falls. A residual made of missing
GG terms drops; one that is not saturates.

To see the residual rather than summarize it, `gg_make_fit_residual_table` returns it
over one plane's grid:

```julia
r = gg_make_fit_residual_table(gg_fit, field, 12)
r.dB[:, :, 1]        # Bx residual, indexed [ix, iy]; 2 and 3 are By and Bs
```

`programs/plot_gg_residuals.jl` draws these as 3D surfaces with Makie — a missing
multipole of order `m` appears as `2m` alternating lobes around the axis, and a
corner-dominated residual is unmistakable.

## Putting it together

The complete script lives at `examples/run_gg_fit.jl`:

```julia
using GeneralizedGradients

field = read_field_grid_hdf5("wsnk_fieldmap_reduced.h5")

params = GGFitInputParams()
params.n_planes_add = 1
params.output_file  = "gg_fit_result.h5"

gg_fit = gg_calc_fit(field, params)
gg_show_fit_results(gg_fit, field, params)
write_gg_fit(gg_fit, field, params)
```

```{tip}
The fit does not strictly require a rectangular, evenly spaced grid — the merit
function is a sum over field points — but the current `gg_calc_fit` assumes the GG
functions are sampled on the grid's own `z`-planes.
```

Next: [evaluate the fitted field](evaluation.md) or
[export it to Bmad](bmad-export.md).
