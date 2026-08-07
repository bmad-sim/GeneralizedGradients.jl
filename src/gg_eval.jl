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
#   B_c(x,y,s) = Σ_{m,nd} CS(Bc_a, m,nd; x,y)·a(m,nd)
#              + Σ_{m,nd} CS(Bc_b, m,nd; x,y)·b(m,nd)
#              + Σ_{nd}   CS(Bc_bs, nd; x,y)·bs(nd)
#   A_c(x,y,s) = Σ_{m,nd} CS(Ac_a, m,nd; x,y)·a(m,nd)
#              + Σ_{m,nd} CS(Ac_b, m,nd; x,y)·b(m,nd)
#              + Σ_{nd}   CS(Ac_bs, nd; x,y)·bs(nd)
#
# with a(m,nd)=dⁿᵈa_m/dsⁿᵈ, b(m,nd)=dⁿᵈb_m/dsⁿᵈ, bs(nd)=dⁿᵈ⁺¹a_0/dsⁿᵈ⁺¹ = dⁿᵈb_s/dsⁿᵈ.
# Here c ∈ {x,y,s} is the component; Bc_a denotes the table Bx_a/By_a/Bs_a (and
# likewise Bc_b, Bc_bs and the potential tables Ac_a, Ac_b, Ac_bs); and the
# coefficient sum CS(T, m,nd; x,y) = Σ coeff·g_refᵏ·xᵖ·yᵠ runs over the entries
# (coeff,p,q,k) stored under key (m,nd) of table T.  The A tables (Ax_a, …,
# As_bs) are precomputed in tables/gg_coef_table.jl from the α/β/γ construction
# of papers/vector-potential and satisfy B = ∇×A exactly.
#
# Because A is linear in the GG functions, its (x,y) derivatives are the
# monomial partials and its s-derivative is obtained by bumping the GG
# derivative order ( ∂_s a(m,nd) = a(m,nd+1), etc. ) — exactly as for the field.
#
# The underscore-prefixed evaluation/interpolation helpers used below live in
# src/low_level.jl.

# The gg_coef tables (Bx_a … As_bs), `_NMAX`, and the other package constants are
# defined in GeneralizedGradients.jl; read_gg_fit lives in gg_utils.jl.

#---------------------------------------------------------------------------------------------------

"""
    field_and_potential_evaluate(fit, ip::Integer, x::Real, y::Real) -> (B, A, dA)

Main entry point. Evaluate the field, vector potential and the Jacobian of `A`
at grid plane `ip` and transverse position `(x, y)`.

- `fit` — the `GGCoefs` struct returned by `read_gg_fit`.
- `ip` — 1-based plane index into `fit.z_base`.
- `x`, `y` — absolute transverse coordinates. `fit.origin` is subtracted
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
function field_and_potential_evaluate(fit, ip::Integer, x::Real, y::Real)
  g_ref = fit.g_ref
  # Shift absolute coordinates onto the GG expansion axis.
  x = float(x) - fit.origin[1]
  y = float(y) - fit.origin[2]

  # GG value getters at this plane (0 when an order is unavailable).
  aval(m, nd)  = (nd >= 0 && haskey(fit.a, (m, nd))) ? fit.a[(m, nd)][ip] : 0.0
  bval(m, nd)  = (nd >= 0 && haskey(fit.b, (m, nd))) ? fit.b[(m, nd)][ip] : 0.0
  bsval(nd)    = (nd >= 0 && haskey(fit.bs, nd))     ? fit.bs[nd][ip]     : 0.0

  # Bumped (s-derivative) getters:  ∂_s a(m,nd) = a(m,nd+1), etc.
  avalp(m, nd) = aval(m, nd + 1)
  bvalp(m, nd) = bval(m, nd + 1)
  bsvalp(nd)   = bsval(nd + 1)

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
    field_and_potential_evaluate_at(plan::GGEvalPlan, x, y, s) -> (B, A, dA)

Evaluate the field, vector potential and Jacobian of `A` at an arbitrary
`(x, y, s)` point, given the compiled evaluation `plan`.

Obtain the plan once from a fit with [`eval_plan`](@ref) and reuse it:

```julia
plan = eval_plan(fit)
B, A, dA = field_and_potential_evaluate_at(plan, x, y, s)
```

Taking the `plan` (rather than the `fit`) is what makes evaluation fast: this
method is `@inline`, allocation-free (stack-resident `SVector` scratch), and
generic over the coordinate type, so it can be called inside a GPU kernel on an
`Adapt.adapt`-ed plan and with `Float32` or `ForwardDiff.Dual` coordinates.

The GG coefficients are stored only at the grid planes `fit.z_base`, but the fit
gives, at each plane, the whole derivative tower of every GG function:
`a(m,0..N)`, `b(m,0..N)`, `bs(0..N)` with `a(m,nd) = dⁿᵈaₘ/dsⁿᵈ` and `N` the
maximum order. So for an `s` between two planes `z_L`, `z_R` we have, for each
function `f`, the value and its first `N` `s`-derivatives at both ends —
`2(N+1)` data — which fix a unique two-point Hermite polynomial `H(s)` of degree
`2N+1`. Each interpolated derivative is taken from the SAME polynomial,
`a(m,nd)(s) = H_aₘ⁽ⁿᵈ⁾(s)`, so the tower stays self-consistent: the interpolated
`a(m,1)` is exactly `d/ds` of the interpolated `a(m,0)`, etc. The plan compiles
these Hermite polynomials once (see low_level.jl).

This is more accurate than independent per-order interpolation (error
`O(h^{2N+2})` for the base coefficient, using only the two straddling planes)
and, because the orders are mutually consistent, the `∂A/∂s` that
`field_and_potential_evaluate` forms by bumping `a(m,nd) → a(m,nd+1)` equals the
true `s`-derivative of the interpolated field. The curl identity `B = ∇×A`
holds at `s` as before.

- `plan` — the compiled [`GGEvalPlan`](@ref) from `eval_plan(fit)`.
- `x`, `y` — absolute transverse coordinates (the fit's `origin` subtracted internally).
- `s` — absolute longitudinal coordinate.

Returns `(B, A, dA)` with the same values as `field_and_potential_evaluate`, but
as stack-allocated `StaticArrays`: `B`, `A` are `SVector{3,T}` and `dA` is an
`SMatrix{3,3,T}` where `T` is the promoted coordinate type (`Float64` for the
usual `Float64` inputs). They index like ordinary vectors/matrices (`A[1]`,
`dA[1,3]`).
"""
@inline function field_and_potential_evaluate_at(plan::GGEvalPlan, x, y, s)
  gvals, xp, yq = _eval_scratch(plan, x, y, s)

  c = plan.comps
  Bx = _comp_value(c[1], gvals, xp, yq)
  By = _comp_value(c[2], gvals, xp, yq)
  Bs = _comp_value(c[3], gvals, xp, yq)
  Axv, Axx, Axy = _comp_full(c[4], gvals, xp, yq)
  Ayv, Ayx, Ayy = _comp_full(c[5], gvals, xp, yq)
  Asv, Asx, Asy = _comp_full(c[6], gvals, xp, yq)
  dAxv = _comp_value(c[7], gvals, xp, yq)
  dAyv = _comp_value(c[8], gvals, xp, yq)
  dAsv = _comp_value(c[9], gvals, xp, yq)

  B = SVector(Bx, By, Bs)
  A = SVector(Axv, Ayv, Asv)
  dA = _make_dA(Axx, Axy, dAxv, Ayx, Ayy, dAyv, Asx, Asy, dAsv)
  return B, A, dA
end

#---------------------------------------------------------------------------------------------------

"""
    potential_evaluate_at(plan::GGEvalPlan, x, y, s) -> (A, dA)

Like [`field_and_potential_evaluate_at`](@ref) but returns only the vector
potential `A` and its Jacobian `dA`, skipping the magnetic field `B`. This is the
entry point used for tracking; get `plan` from a fit with `eval_plan(fit)` and
reuse it.

For tracking, only `A` and `dA` are needed. The `B` field is the majority of the
per-call work (its monomial expansion has more terms than `A`'s), so skipping it
is roughly `1.8x` faster than the full evaluator while returning identical
`A`, `dA`. `A` is an `SVector{3,T}` and `dA` an `SMatrix{3,3,T}` for the promoted
coordinate type `T`. Allocation-free, GPU-capable and type-generic. See
`field_and_potential_evaluate_at` for the `(x, y, s)` conventions.
"""
@inline function potential_evaluate_at(plan::GGEvalPlan, x, y, s)
  gvals, xp, yq = _eval_scratch(plan, x, y, s)

  c = plan.comps
  Axv, Axx, Axy = _comp_full(c[4], gvals, xp, yq)
  Ayv, Ayx, Ayy = _comp_full(c[5], gvals, xp, yq)
  Asv, Asx, Asy = _comp_full(c[6], gvals, xp, yq)
  dAxv = _comp_value(c[7], gvals, xp, yq)
  dAyv = _comp_value(c[8], gvals, xp, yq)
  dAsv = _comp_value(c[9], gvals, xp, yq)

  A = SVector(Axv, Ayv, Asv)
  dA = _make_dA(Axx, Axy, dAxv, Ayx, Ayy, dAyv, Asx, Asy, dAsv)
  return A, dA
end

#---------------------------------------------------------------------------------------------------

"""
    field_evaluate_at(plan::GGEvalPlan, x, y, s) -> B

Like [`field_and_potential_evaluate_at`](@ref) but returns only the magnetic
field `B = [Bx, By, Bs]` as an `SVector{3,T}` for the promoted coordinate type
`T`, skipping the vector potential `A` and its Jacobian. `B` is identical to that
of the full evaluator. Allocation-free, GPU-capable and type-generic; get `plan`
from a fit with `eval_plan(fit)`. See `field_and_potential_evaluate_at` for the
`(x, y, s)` conventions.
"""
@inline function field_evaluate_at(plan::GGEvalPlan, x, y, s)
  gvals, xp, yq = _eval_scratch(plan, x, y, s)

  c = plan.comps
  Bx = _comp_value(c[1], gvals, xp, yq)
  By = _comp_value(c[2], gvals, xp, yq)
  Bs = _comp_value(c[3], gvals, xp, yq)

  return SVector(Bx, By, Bs)
end

#---------------------------------------------------------------------------------------------------

"""
    field_coefficients_at_plane(fit, ip::Integer) -> (CBx, CBy, CBs)

Field-expansion coefficients at a grid plane.

- `fit` — the `GGCoefs` struct from `read_gg_fit`.
- `ip` — 1-based plane index into `fit.z_base`.

Returns `(CBx, CBy, CBs)`; each is a matrix with `CB[i+1, j+1] = CB_{c,i,j}`,
the coefficient of `xⁱ yʲ` in that field component at the plane.
"""
function field_coefficients_at_plane(fit, ip::Integer)
  return _trim3(_field_CB(fit, ip)...)
end

#---------------------------------------------------------------------------------------------------
"""
    field_coefficients_at_s(fit, s::Real) -> (CBx, CBy, CBs)

Field-expansion coefficients at an arbitrary `s`, via the same Hermite
interpolation of the GG quantities used by `field_and_potential_evaluate_at`.
Returns `(CBx, CBy, CBs)` where each `CB` is a matrix with
`CB[i+1, j+1] = CB_{c,i,j}`, the coefficient of `xⁱ yʲ` in that field component
at the plane.
"""
function field_coefficients_at_s(fit, s::Real)
  return _trim3(_field_CB(_interp_gg_fit(fit, s), 1)...)
end

#---------------------------------------------------------------------------------------------------
"""
    gg_coefficients_at_plane(fit, ip::Integer) -> (a, b, bs)

Generalized-gradient coefficients at a grid plane.

- `fit` — the `GGCoefs` struct from `read_gg_fit`.
- `ip` — 1-based plane index into `fit.z_base`.

Returns the three GG-function dicts of scalar values at the plane: `a` and `b`
keyed by `(m,nd)` with `a(m,nd) = dⁿᵈaₘ/dsⁿᵈ`, `b(m,nd) = dⁿᵈbₘ/dsⁿᵈ`; and `bs`
keyed by `nd` with `bs(nd) = dⁿᵈ⁺¹a_0/dsⁿᵈ⁺¹ = dⁿᵈb_s/dsⁿᵈ`.
"""
function gg_coefficients_at_plane(fit, ip::Integer)
  a  = Dict{Tuple{Int,Int},Float64}((mnd => v[ip]) for (mnd, v) in fit.a)
  b  = Dict{Tuple{Int,Int},Float64}((mnd => v[ip]) for (mnd, v) in fit.b)
  bs = Dict{Int,Float64}((nd => v[ip]) for (nd, v) in fit.bs)
  return a, b, bs
end

#---------------------------------------------------------------------------------------------------
"""
    gg_coefficients_at_s(fit, s::Real) -> (a, b, bs)

Generalized-gradient coefficients at an arbitrary `s`, Hermite-interpolated from
the straddling grid planes (the same interpolation used by
`field_and_potential_evaluate_at`). Returns the three GG-function dicts of
scalar values, as in `gg_coefficients_at_plane`.
"""
function gg_coefficients_at_s(fit, s::Real)
  return gg_coefficients_at_plane(_interp_gg_fit(fit, s), 1)
end
