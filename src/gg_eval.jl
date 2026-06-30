# ---------------------------------------------------------------------------
# gg_eval.jl
#
# Evaluate the magnetic field B = (Bx, By, Bs) and the vector potential
# A = (Ax, Ay, As) -- together with the 3x3 Jacobian of A with respect to
# (x, y, s) -- at a chosen base plane and transverse position, given the
# generalized-gradient (GG) coefficients produced by src/gg_fit.jl.
#
# Both the field and the vector potential are evaluated the same way: as the
# monomial expansions whose coefficients are tabulated in tables/gg_coef_table.jl.
#
#   B_c(x,y,s) = Σ_{n,m} CS(Bc_a, n,m; x,y)·a(n,m)
#              + Σ_{n,m} CS(Bc_b, n,m; x,y)·b(n,m)
#              + Σ_{m}   CS(Bc_bs, m; x,y)·bs(m)
#   A_c(x,y,s) = Σ_{n,m} CS(Ac_a, n,m; x,y)·a(n,m)
#              + Σ_{n,m} CS(Ac_b, n,m; x,y)·b(n,m)
#              + Σ_{m}   CS(Ac_bs, m; x,y)·bs(m)
#
# with a(n,m)=dᵐa_n/dsᵐ, b(n,m)=dᵐb_n/dsᵐ, bs(m)=dᵐ⁺¹a_0/dsᵐ⁺¹ = dᵐb_s/dsᵐ.
# Here c ∈ {x,y,s} is the component; Bc_a denotes the table Bx_a/By_a/Bs_a (and
# likewise Bc_b, Bc_bs and the potential tables Ac_a, Ac_b, Ac_bs); and the
# coefficient sum CS(T, n,m; x,y) = Σ coeff·g_refᵏ·xᵖ·yᵠ runs over the entries
# (coeff,p,q,k) stored under key (n,m) of table T.  The A tables (Ax_a, …,
# As_bs) are precomputed in tables/gg_coef_table.jl from the α/β/γ construction
# of papers/vector-potential and satisfy B = ∇×A exactly.
#
# Because A is linear in the GG functions, its (x,y) derivatives are the
# monomial partials and its s-derivative is obtained by bumping the GG
# derivative order ( ∂_s a(n,m) = a(n,m+1), etc. ) — exactly as for the field.
#
# The underscore-prefixed evaluation/interpolation helpers used below live in
# src/helpers.jl.

# The gg_coef tables (Bx_a … As_bs), `_NMAX`, and the other package constants are
# defined in GeneralizedGradients.jl; read_gg_fit lives in field_grid.jl.

#---------------------------------------------------------------------------------------------------

"""
    field_and_potential_evaluate(fit, meta, ip::Integer, x::Real, y::Real) -> (B, A, dA)

Main entry point. Evaluate the field, vector potential and the Jacobian of `A`
at grid plane `ip` and transverse position `(x, y)`.

- `fit`, `meta` — the `GGCoefs` struct and metadata NamedTuple returned by
  `read_gg_fit`.
- `ip` — 1-based plane index into `fit.z_base`.
- `x`, `y` — absolute transverse coordinates. `meta.origin` is subtracted
  internally to obtain the position relative to the GG expansion axis (the
  coordinate the expansion is written in). Pass an origin of `(0,0)` — or use
  the default — for axis-relative input.

Returns `(B, A, dA)` where

```
B  = [Bx, By, Bs]
A  = [Ax, Ay, As]
dA = 3x3 matrix, dA[i,j] = ∂A_i/∂u_j  with  (A_1,A_2,A_3) = (Ax,Ay,As)
     and (u_1,u_2,u_3) = (x,y,s).
```
"""
function field_and_potential_evaluate(fit, meta, ip::Integer, x::Real, y::Real)
  g_ref = fit.g_ref
  # Shift absolute coordinates onto the GG expansion axis.
  x = float(x) - meta.origin[1]
  y = float(y) - meta.origin[2]

  # GG value getters at this plane (0 when an order is unavailable).
  aval(n, m)  = (m >= 0 && haskey(fit.a, (n, m)))  ? fit.a[(n, m)][ip]  : 0.0
  bval(n, m)  = (m >= 0 && haskey(fit.b, (n, m)))  ? fit.b[(n, m)][ip]  : 0.0
  bsval(m)    = (m >= 0 && haskey(fit.bs, m))      ? fit.bs[m][ip]      : 0.0

  # Bumped (s-derivative) getters:  ∂_s a(n,m) = a(n,m+1), etc.
  avalp(n, m) = aval(n, m + 1)
  bvalp(n, m) = bval(n, m + 1)
  bsvalp(m)   = bsval(m + 1)

  # --- field ---
  Bx = _polyval(_comp_array(Bx_a, Bx_b, Bx_bs, aval, bval, bsval, g_ref), x, y)[1]
  By = _polyval(_comp_array(By_a, By_b, By_bs, aval, bval, bsval, g_ref), x, y)[1]
  Bs = _polyval(_comp_array(Bs_a, Bs_b, Bs_bs, aval, bval, bsval, g_ref), x, y)[1]

  # --- vector potential: value and (x,y) partials straight from the tables ---
  Axv, Axx, Axy = _polyval(_comp_array(Ax_a, Ax_b, Ax_bs, aval, bval, bsval, g_ref), x, y)
  Ayv, Ayx, Ayy = _polyval(_comp_array(Ay_a, Ay_b, Ay_bs, aval, bval, bsval, g_ref), x, y)
  Asv, Asx, Asy = _polyval(_comp_array(As_a, As_b, As_bs, aval, bval, bsval, g_ref), x, y)

  # ∂A/∂s: same tables evaluated with bumped GG derivative orders.
  dAxv = _polyval(_comp_array(Ax_a, Ax_b, Ax_bs, avalp, bvalp, bsvalp, g_ref), x, y)[1]
  dAyv = _polyval(_comp_array(Ay_a, Ay_b, Ay_bs, avalp, bvalp, bsvalp, g_ref), x, y)[1]
  dAsv = _polyval(_comp_array(As_a, As_b, As_bs, avalp, bvalp, bsvalp, g_ref), x, y)[1]

  B  = [Bx, By, Bs]
  A  = [Axv, Ayv, Asv]
  dA = [Axx Axy dAxv;
        Ayx Ayy dAyv;
        Asx Asy dAsv]
  return B, A, dA
end

#---------------------------------------------------------------------------------------------------

"""
    field_and_potential_evaluate_at(fit, meta, x::Real, y::Real, s::Real) -> (B, A, dA)

Evaluate at an arbitrary `(x, y, s)` point.

The GG coefficients are stored only at the grid planes `fit.z_base`, but the
fit gives, at each plane, the whole derivative tower of every GG function:
`a(n,0..N)`, `b(n,0..N)`, `bs(0..N)` with `a(n,m) = dᵐaₙ/dsᵐ` and `N` the
maximum order. So for an `s` between two planes `z_L`, `z_R` we have, for each
function `f`, the value and its first `N` `s`-derivatives at both ends —
`2(N+1)` data — which fix a unique two-point Hermite polynomial `H(s)` of degree
`2N+1`. Each interpolated derivative is taken from the SAME polynomial,
`a(n,m)(s) = H_aₙ⁽ᵐ⁾(s)`, so the tower stays self-consistent: the interpolated
`a(n,1)` is exactly `d/ds` of the interpolated `a(n,0)`, etc.

This is more accurate than independent per-order interpolation (error
`O(h^{2N+2})` for the base coefficient, using only the two straddling planes)
and, because the orders are mutually consistent, the `∂A/∂s` that
`field_and_potential_evaluate` forms by bumping `a(n,m) → a(n,m+1)` equals the
true `s`-derivative of the interpolated field. The curl identity `B = ∇×A`
holds at `s` as before.

- `fit`, `meta` — the `GGCoefs` struct and metadata NamedTuple from `read_gg_fit`.
- `x`, `y` — absolute transverse coordinates (`meta.origin` subtracted internally).
- `s` — absolute longitudinal coordinate.

Returns `(B, A, dA)` exactly as `field_and_potential_evaluate`.
"""
function field_and_potential_evaluate_at(fit, meta, x::Real, y::Real, s::Real)
  return field_and_potential_evaluate(_interp_gg_fit(fit, meta, s)..., 1, x, y)
end

#---------------------------------------------------------------------------------------------------
"""
    field_coefficients_at_plane(fit, meta, ip::Integer) -> (CBx, CBy, CBs)

Field-expansion coefficients at a grid plane.

- `fit`, `meta` — the `GGCoefs` struct and metadata NamedTuple from `read_gg_fit`.
- `ip` — 1-based plane index into `fit.z_base`.

Returns `(CBx, CBy, CBs)`; each is a matrix with `CB[i+1, j+1] = CB_{c,i,j}`,
the coefficient of `xⁱ yʲ` in that field component at the plane.
"""
function field_coefficients_at_plane(fit, meta, ip::Integer)
  return _trim3(_field_CB(fit, meta, ip)...)
end

#---------------------------------------------------------------------------------------------------
"""
    field_coefficients_at_s(fit, meta, s::Real) -> (CBx, CBy, CBs)

Field-expansion coefficients at an arbitrary `s`, via the same Hermite
interpolation of the GG quantities used by `field_and_potential_evaluate_at`.
Returns `(CBx, CBy, CBs)` where each `CB` is a matrix with
`CB[i+1, j+1] = CB_{c,i,j}`, the coefficient of `xⁱ yʲ` in that field component
at the plane.
"""
function field_coefficients_at_s(fit, meta, s::Real)
  return _trim3(_field_CB(_interp_gg_fit(fit, meta, s)..., 1)...)
end

#---------------------------------------------------------------------------------------------------
"""
    gg_coefficients_at_plane(fit, meta, ip::Integer) -> (a, b, bs)

Generalized-gradient coefficients at a grid plane.

- `fit`, `meta` — the `GGCoefs` struct and metadata NamedTuple from
  `read_gg_fit` (`meta` is accepted for a uniform call signature; it is unused).
- `ip` — 1-based plane index into `fit.z_base`.

Returns the three GG-function dicts of scalar values at the plane: `a` and `b`
keyed by `(n,m)` with `a(n,m) = dᵐaₙ/dsᵐ`, `b(n,m) = dᵐbₙ/dsᵐ`; and `bs` keyed
by `m` with `bs(m) = dᵐ⁺¹a_0/dsᵐ⁺¹ = dᵐb_s/dsᵐ`.
"""
function gg_coefficients_at_plane(fit, meta, ip::Integer)
  a  = Dict{Tuple{Int,Int},Float64}((nm => v[ip]) for (nm, v) in fit.a)
  b  = Dict{Tuple{Int,Int},Float64}((nm => v[ip]) for (nm, v) in fit.b)
  bs = Dict{Int,Float64}((m => v[ip]) for (m, v) in fit.bs)
  return a, b, bs
end

#---------------------------------------------------------------------------------------------------
"""
    gg_coefficients_at_s(fit, meta, s::Real) -> (a, b, bs)

Generalized-gradient coefficients at an arbitrary `s`, Hermite-interpolated from
the straddling grid planes (the same interpolation used by
`field_and_potential_evaluate_at`). Returns the three GG-function dicts of
scalar values, as in `gg_coefficients_at_plane`.
"""
function gg_coefficients_at_s(fit, meta, s::Real)
  return gg_coefficients_at_plane(_interp_gg_fit(fit, meta, s)..., 1)
end
