# Evaluating the fitted field

Once a fit has been written with `write_gg_fit`, load it back with
`read_gg_fit` and evaluate the field, the vector potential, and the
field-expansion coefficients anywhere.

```julia
using GeneralizedGradients
fit, meta = read_gg_fit("gg_fit_result.h5")
```

`read_gg_fit` returns a two-tuple. `fit` is a `GGFit` struct with the GG
coefficient dictionaries `a`, `b`, `bs` (plus `z_base`, `m_max`, `nd_max`, `rms_weighted_plane`,
`g_ref`, `origin`, `dz_grid`), and `meta` is a NamedTuple of the associated
fit-control metadata `n_planes_add`, `core_weight`, `outer_plane_weight`. Both
are passed together to the evaluation functions below.

## Field and vector potential

The GG coefficients are stored only at the grid planes `fit.z_base`. Evaluate
at a specific plane index `ip`:

```julia
B, A, dA = field_and_potential_evaluate(fit, ip, x, y)
```

or at an arbitrary longitudinal position `s` (Hermite-interpolated between the
straddling planes):

```julia
B, A, dA = field_and_potential_evaluate_at(fit, x, y, s)
```

Here `B` is the field 3-vector `[Bx, By, Bs]`, `A` is the vector potential, and
`dA[i, j] = ∂A_i/∂u_j` with `u = (x, y, s)`. The transverse coordinates `x`, `y`
are absolute; `fit.origin` is subtracted internally.

```{note}
On a curved reference frame (`g_ref ≠ 0`) the returned `B` is exactly the
Frenet–Serret curl of `A`, which is a useful self-consistency check.
```

## Field-expansion coefficients

To get the coefficients `C_{c,i,j}` of `xⁱ yʲ` in each field component:

```julia
CBx, CBy, CBs = field_coefficients_at_plane(fit, ip)   # at grid plane ip
CBx, CBy, CBs = field_coefficients_at_s(fit, s)        # at arbitrary s
```

## Generalized-gradient coefficients

To get the GG coefficients themselves as scalar dictionaries:

```julia
a, b, bs = gg_coefficients_at_plane(fit, ip)   # at grid plane ip
a, b, bs = gg_coefficients_at_s(fit, s)        # at arbitrary s
```

`a` and `b` are keyed by `(m, nd)` with `a(m,nd) = dⁿᵈaₘ/dsⁿᵈ`; `bs` is keyed by `nd`
with `bs(nd) = dⁿᵈb_s/dsⁿᵈ`.

See the **API Reference** (sidebar) for the full signatures and return types.
