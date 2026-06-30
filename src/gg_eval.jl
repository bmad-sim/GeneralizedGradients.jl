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

#---------------------------------------------------------------------------------------------------

const _TABLE_FILE = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
include(_TABLE_FILE)   # Bx_a … Bs_bs (field) and Ax_a … As_bs (vector potential)

# Working size for the truncated (x,y) coefficient arrays.  The table is built
# to total monomial degree MAXTOT (12), so 20 leaves ample headroom.
const _NMAX = 20

_newK() = zeros(Float64, _NMAX, _NMAX)

# read_gg_fit lives in field_grid_io.jl (HDF5 storage); files are written by write_gg_fit.

#---------------------------------------------------------------------------------------------------
# Coefficient-array builders.  K[p+1,q+1] = coefficient of xᵖ yᵠ.

"""
    _accum(tdict, valfun, g_ref) -> K

Coefficient-array builder. `K[p+1,q+1]` = coefficient of `xᵖ yᵠ`.
`valfun(key)` returns the GG function value multiplying that table entry.
"""
function _accum(tdict, valfun, g_ref)
  K = _newK()
  for (key, terms) in tdict
    v = valfun(key)
    v == 0.0 && continue
    for (c, p, q, k) in terms
      K[p+1, q+1] += float(c) * (k == 0 ? 1.0 : float(g_ref)^k) * v
    end
  end
  return K
end

#---------------------------------------------------------------------------------------------------

"""
    _comp_array(Ta, Tb, Tbs, aval, bval, bsval, g_ref) -> K

Combined coefficient array of a component: sum of its `a`, `b` and `bs` parts.
`Ta`/`Tb` are keyed by `(n,m)` and `Tbs` by `m`.
"""
function _comp_array(Ta, Tb, Tbs, aval, bval, bsval, g_ref)
  return _accum(Ta, k -> aval(k...), g_ref) .+
         _accum(Tb, k -> bval(k...), g_ref) .+
         _accum(Tbs, m -> bsval(m), g_ref)
end

#---------------------------------------------------------------------------------------------------

"""
    _polyval(K, x, y) -> (val, dvx, dvy)

Value and `(x,y)` partials of the plain polynomial `Σ K[i,j] xⁱ yʲ`.
"""
function _polyval(K, x, y)
  val = 0.0; dvx = 0.0; dvy = 0.0
  for i in 0:_NMAX-1, j in 0:_NMAX-1
    c = K[i+1, j+1]
    c == 0.0 && continue
    xi = x^i; yj = y^j
    val += c * xi * yj
    i > 0 && (dvx += c * i * x^(i-1) * yj)
    j > 0 && (dvy += c * j * xi * y^(j-1))
  end
  return val, dvx, dvy
end

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
  g_ref = meta.g_ref
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

# ---------------------------------------------------------------------------
# Hermite interpolation of the GG towers between grid planes (see
# field_and_potential_evaluate_at for the method and rationale).
# ---------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

"""
    _fct(k::Integer) -> Float64

Float factorial (orders are small but keep it overflow-proof).
"""
_fct(k::Integer) = (f = 1.0; for i in 2:k; f *= i; end; f)

"""
    _taylor_derivs(z0, f0, sq) -> Vector

Single-point Taylor tower: from `f` and its derivatives at `z0`, return
`[P⁽ᵐ⁾(sq) for m=0..N]` with `P` the Taylor series, i.e. `f` extrapolated to `sq`.
"""
function _taylor_derivs(z0, f0, sq)
  N = length(f0) - 1
  u = sq - z0
  out = zeros(Float64, N + 1)
  for m in 0:N
    acc = 0.0
    for j in m:N
      acc += f0[j+1] * u^(j-m) / _fct(j-m)
    end
    out[m+1] = acc
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _hermite_derivs(zL, zR, fL, fR, sq) -> Vector

Two-point Hermite tower: `fL[j+1] = f⁽ʲ⁾(zL)`, `fR[j+1] = f⁽ʲ⁾(zR)`, `j = 0..N`.
Returns `[H⁽ᵐ⁾(sq) for m=0..N]` where `H` is the degree-`(2N+1)` Hermite
interpolant. Built via confluent Newton divided differences in the local
coordinate `u = s - zL` (nodes: `0` with multiplicity `N+1`, `hstep` with
multiplicity `N+1`).
"""
function _hermite_derivs(zL, zR, fL, fR, sq)
  N = length(fL) - 1
  K = 2N + 1                       # polynomial degree = (#nodes) - 1
  hstep = zR - zL
  fval(i, j) = (i <= N ? fL : fR)[j+1]    # j-th derivative at node i's plane
  isR(i) = i > N                          # node i (0-based) in the R block?

  memo = Dict{Tuple{Int,Int},Float64}()
  function dd(i, k)                # divided difference f[t_i, …, t_{i+k}]
    haskey(memo, (i, k)) && return memo[(i, k)]
    v = isR(i) == isR(i + k) ? fval(i, k) / _fct(k) :       # one block: f⁽ᵏ⁾/k!
      (dd(i + 1, k - 1) - dd(i, k - 1)) / hstep           # spans both blocks
    memo[(i, k)] = v
    return v
  end

  c = [dd(0, k) for k in 0:K]                 # Newton coefficients
  tnode(i) = i <= N ? 0.0 : hstep             # node positions in u

  # Accumulate the Newton form into monomial coefficients in u.
  poly  = zeros(Float64, K + 1)               # poly[d+1] = coeff of u^d
  basis = [1.0]                               # current ∏ (u - t_j)
  for k in 0:K
    @inbounds for d in 1:length(basis)
      poly[d] += c[k+1] * basis[d]
    end
    if k < K                                # multiply basis by (u - t_k)
      tk = tnode(k)
      nb = zeros(Float64, length(basis) + 1)
      @inbounds for d in 1:length(basis)
        nb[d+1] += basis[d]
        nb[d]   -= tk * basis[d]
      end
      basis = nb
    end
  end

  # Evaluate H and its derivatives at uq.
  uq = sq - zL
  out = zeros(Float64, N + 1)
  for m in 0:N
    acc = 0.0
    for d in m:K
      ff = 1.0                            # falling factorial d·(d-1)···(d-m+1)
      for r in 0:m-1
        ff *= (d - r)
      end
      acc += poly[d+1] * ff * uq^(d-m)
    end
    out[m+1] = acc
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_tower(fL, fR, zL, zR, sq, single) -> Vector

Interpolate one GG function's derivative tower onto `sq` (Hermite, or Taylor if `single`).
"""
_interp_tower(fL, fR, zL, zR, sq, single) =
  single ? _taylor_derivs(zL, fL, sq) : _hermite_derivs(zL, zR, fL, fR, sq)

#---------------------------------------------------------------------------------------------------

"""
    _contiguous_order(orders) -> N

Largest `N` such that orders `0,1,…,N` are all present in `orders` (sorted).
"""
function _contiguous_order(orders)
  N = -1
  for (idx, m) in enumerate(orders)
    m == idx - 1 ? (N = m) : break
  end
  return N
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_nm_dict(d, iL, iR, zL, zR, sq, single) -> Dict

Interpolate an `(n,m)`-keyed dict (`a`, `b`): build one Hermite per multipole `n`.
"""
function _interp_nm_dict(d, iL, iR, zL, zR, sq, single)
  out = Dict{Tuple{Int,Int},Vector{Float64}}()
  byn = Dict{Int,Vector{Int}}()
  for (n, m) in keys(d)
    push!(get!(byn, n, Int[]), m)
  end
  for (n, ms) in byn
    sort!(ms)
    N  = _contiguous_order(ms)
    fL = [d[(n, j)][iL] for j in 0:N]
    fR = [d[(n, j)][iR] for j in 0:N]
    vals = _interp_tower(fL, fR, zL, zR, sq, single)
    for j in 0:N
      out[(n, j)] = [vals[j+1]]
    end
    for m in ms                              # any non-contiguous order: nearest plane
      m > N && (out[(n, m)] = [d[(n, m)][iL]])
    end
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_m_dict(d, iL, iR, zL, zR, sq, single) -> Dict

Interpolate an `m`-keyed dict (`bs`): a single Hermite tower.
"""
function _interp_m_dict(d, iL, iR, zL, zR, sq, single)
  out = Dict{Int,Vector{Float64}}()
  ms  = sort(collect(keys(d)))
  N   = _contiguous_order(ms)
  fL  = [d[j][iL] for j in 0:N]
  fR  = [d[j][iR] for j in 0:N]
  vals = _interp_tower(fL, fR, zL, zR, sq, single)
  for j in 0:N
    out[j] = [vals[j+1]]
  end
  for m in ms
    m > N && (out[m] = [d[m][iL]])
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_gg_fit(fit, meta, s::Real) -> (fit::GGCoefs, meta)

Take GG fit results `fit` which give the GG functions at a set of planes and
return a similar `(fit, meta)` pair but with one plane: the GG coefficients for
that plane are the interpolated GG coefficients at the given `s`-position. The
`meta` is returned unchanged.

- `fit`, `meta` — GG coefficients for all planes and the associated metadata.

Builds a single virtual plane at `s` by Hermite-interpolating every GG derivative
tower from the two straddling grid planes (one-plane Taylor if only one plane).
"""
function _interp_gg_fit(fit, meta, s::Real)
  z  = fit.z_base
  P  = length(z)
  sq = float(s)

  if P == 1
    iL = iR = 1
  else
    i0 = searchsortedlast(z, sq)             # z[i0] <= s < z[i0+1]
    iL = clamp(i0, 1, P - 1)                 # straddling pair (extrapolates at ends)
    iR = iL + 1
  end
  single = iL == iR
  zL = z[iL]; zR = z[iR]

  a2  = _interp_nm_dict(fit.a,  iL, iR, zL, zR, sq, single)
  b2  = _interp_nm_dict(fit.b,  iL, iR, zL, zR, sq, single)
  bs2 = _interp_m_dict(fit.bs, iL, iR, zL, zR, sq, single)

  fit2 = GGCoefs(; z_base = [sq], a = a2, b = b2, bs = bs2,
                        m_max = fit.m_max, rms_plane = [NaN])
  return fit2, meta
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
    _field_CB(fit, meta, ip::Integer) -> (CBx, CBy, CBs)

Field-expansion coefficients `B_c(x,y,s) = Σ_{i,j} CB_{c,i,j}(s) xⁱ yʲ`.
Returns full `_NMAX×_NMAX` arrays summed over the `a`, `b`, `bs` parts.
"""
function _field_CB(fit, meta, ip::Integer)
  g_ref = meta.g_ref
  aval(n, m) = (m >= 0 && haskey(fit.a, (n, m))) ? fit.a[(n, m)][ip] : 0.0
  bval(n, m) = (m >= 0 && haskey(fit.b, (n, m))) ? fit.b[(n, m)][ip] : 0.0
  bsval(m)   = (m >= 0 && haskey(fit.bs, m))     ? fit.bs[m][ip]     : 0.0
  CBx = _accum(Bx_a, k -> aval(k...), g_ref) .+ _accum(Bx_b, k -> bval(k...), g_ref) .+ _accum(Bx_bs, m -> bsval(m), g_ref)
  CBy = _accum(By_a, k -> aval(k...), g_ref) .+ _accum(By_b, k -> bval(k...), g_ref) .+ _accum(By_bs, m -> bsval(m), g_ref)
  CBs = _accum(Bs_a, k -> aval(k...), g_ref) .+ _accum(Bs_b, k -> bval(k...), g_ref) .+ _accum(Bs_bs, m -> bsval(m), g_ref)
  return CBx, CBy, CBs
end

#---------------------------------------------------------------------------------------------------
"""
    _trim3(CBx, CBy, CBs) -> (CBx, CBy, CBs)

Trim three coefficient arrays to the smallest `(x,y)` extent holding every
nonzero entry, so the returned matrices are indexed `CB[i+1, j+1] = CB_{c,i,j}`.
"""
function _trim3(CBx, CBy, CBs)
  pmax = 1; qmax = 1
  for K in (CBx, CBy, CBs), j in 1:_NMAX, i in 1:_NMAX
    if K[i, j] != 0.0
      pmax = max(pmax, i); qmax = max(qmax, j)
    end
  end
  return CBx[1:pmax, 1:qmax], CBy[1:pmax, 1:qmax], CBs[1:pmax, 1:qmax]
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
