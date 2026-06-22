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
#   B_c(x,y,s) = Σ_{(n,m)} CS^B_c,a(n,m;x,y)·a(n,m)
#              + Σ_{(n,m)} CS^B_c,b(n,m;x,y)·b(n,m)
#              + Σ_{m}     CS^B_c,bs(m;x,y)·bs(m)
#   A_c(x,y,s) = Σ_{(n,m)} CS^A_c,a(n,m;x,y)·a(n,m)
#              + Σ_{(n,m)} CS^A_c,b(n,m;x,y)·b(n,m)
#              + Σ_{m}     CS^A_c,bs(m;x,y)·bs(m)
#
# with a(n,m)=dᵐa_n/dsᵐ, b(n,m)=dᵐb_n/dsᵐ, bs(m)=dᵐ⁺¹a_0/dsᵐ⁺¹ = dᵐb_s/dsᵐ,
# and CS_c,f = Σ (coeff·hᵏ·xᵖ·yᵠ) the sum of that function's table entries.  The
# A tables (Ax_a, …, As_bs) are precomputed in tables/gg_coef_table.jl from the
# α/β/γ construction of papers/vector-potential and satisfy B = ∇×A exactly.
#
# Because A is linear in the GG functions, its (x,y) derivatives are the
# monomial partials and its s-derivative is obtained by bumping the GG
# derivative order ( ∂_s a(n,m) = a(n,m+1), etc. ) — exactly as for the field.
# ---------------------------------------------------------------------------

using JLD2

const _TABLE_FILE = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
include(_TABLE_FILE)   # Bx_a … Bs_bs (field) and Ax_a … As_bs (vector potential)

# Working size for the truncated (x,y) coefficient arrays.  The table is built
# to total monomial degree MAXTOT (12), so 20 leaves ample headroom.
const _NMAX = 20

_newK() = zeros(Float64, _NMAX, _NMAX)

# ---------------------------------------------------------------------------
# Load a gg_fit.jl result file into a NamedTuple.
# ---------------------------------------------------------------------------
function gg_load_result(path::AbstractString)
    d = load(path)
    return (; z_base   = d["z_base"],
              a        = d["a"],   b  = d["b"],  bs = d["bs"],
              h        = d["h"],   origin = d["origin"],
              r0_grid  = d["r0_grid"], dz_grid = d["dz_grid"],
              m_max    = d["m_max"],   rms_plane = d["rms_plane"])
end

# ---------------------------------------------------------------------------
# Coefficient-array builders.  K[p+1,q+1] = coefficient of xᵖ yᵠ.
# `valfun(key)` returns the GG function value multiplying that table entry.
# ---------------------------------------------------------------------------
function _accum(tdict, valfun, h)
    K = _newK()
    for (key, terms) in tdict
        v = valfun(key)
        v == 0.0 && continue
        for (c, p, q, k) in terms
            K[p+1, q+1] += float(c) * (k == 0 ? 1.0 : float(h)^k) * v
        end
    end
    return K
end

# Combined coefficient array of a component: sum of its a, b and bs parts.
# `Ta`/`Tb` are keyed by (n,m) and `Tbs` by m.
function _comp_array(Ta, Tb, Tbs, aval, bval, bsval, h)
    return _accum(Ta, k -> aval(k...), h) .+
           _accum(Tb, k -> bval(k...), h) .+
           _accum(Tbs, m -> bsval(m), h)
end

# Value and (x,y) partials of the plain polynomial Σ K_{i,j} xⁱ yʲ.
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

# ---------------------------------------------------------------------------
# Main entry point.
#
#   res   : NamedTuple from gg_load_result (or gg_fit.jl's `result`)
#   ip    : 1-based plane index into res.z_base
#   x, y  : absolute transverse coordinates.  res.origin is subtracted
#           internally to obtain the position relative to the GG expansion
#           axis (the coordinate the expansion is written in).  Pass an
#           origin of (0,0) — or use the default — for axis-relative input.
#
# Returns (B, A, dA) where
#   B  = [Bx, By, Bs]
#   A  = [Ax, Ay, As]
#   dA = 3x3 matrix, dA[i,j] = ∂A_i/∂u_j  with  (A_1,A_2,A_3)=(Ax,Ay,As)
#        and (u_1,u_2,u_3)=(x,y,s).
# ---------------------------------------------------------------------------
function gg_evaluate(res, ip::Integer, x::Real, y::Real)
    h = res.h
    # Shift absolute coordinates onto the GG expansion axis.
    x = float(x) - res.origin[1]
    y = float(y) - res.origin[2]

    # GG value getters at this plane (0 when an order is unavailable).
    aval(n, m)  = (m >= 0 && haskey(res.a, (n, m)))  ? res.a[(n, m)][ip]  : 0.0
    bval(n, m)  = (m >= 0 && haskey(res.b, (n, m)))  ? res.b[(n, m)][ip]  : 0.0
    bsval(m)    = (m >= 0 && haskey(res.bs, m))      ? res.bs[m][ip]      : 0.0

    # Bumped (s-derivative) getters:  ∂_s a(n,m) = a(n,m+1), etc.
    avalp(n, m) = aval(n, m + 1)
    bvalp(n, m) = bval(n, m + 1)
    bsvalp(m)   = bsval(m + 1)

    # --- field ---
    Bx = _polyval(_comp_array(Bx_a, Bx_b, Bx_bs, aval, bval, bsval, h), x, y)[1]
    By = _polyval(_comp_array(By_a, By_b, By_bs, aval, bval, bsval, h), x, y)[1]
    Bs = _polyval(_comp_array(Bs_a, Bs_b, Bs_bs, aval, bval, bsval, h), x, y)[1]

    # --- vector potential: value and (x,y) partials straight from the tables ---
    Axv, Axx, Axy = _polyval(_comp_array(Ax_a, Ax_b, Ax_bs, aval, bval, bsval, h), x, y)
    Ayv, Ayx, Ayy = _polyval(_comp_array(Ay_a, Ay_b, Ay_bs, aval, bval, bsval, h), x, y)
    Asv, Asx, Asy = _polyval(_comp_array(As_a, As_b, As_bs, aval, bval, bsval, h), x, y)

    # ∂A/∂s: same tables evaluated with bumped GG derivative orders.
    dAxv = _polyval(_comp_array(Ax_a, Ax_b, Ax_bs, avalp, bvalp, bsvalp, h), x, y)[1]
    dAyv = _polyval(_comp_array(Ay_a, Ay_b, Ay_bs, avalp, bvalp, bsvalp, h), x, y)[1]
    dAsv = _polyval(_comp_array(As_a, As_b, As_bs, avalp, bvalp, bsvalp, h), x, y)[1]

    B  = [Bx, By, Bs]
    A  = [Axv, Ayv, Asv]
    dA = [Axx Axy dAxv;
          Ayx Ayy dAyv;
          Asx Asy dAsv]
    return B, A, dA
end

# ---------------------------------------------------------------------------
# Evaluate at an arbitrary (x, y, s) point.
#
# The GG coefficients are stored only at the grid planes res.z_base.  For an
# s that falls between planes, each stored quantity a(n,m), b(n,m), bs(m) is
# interpolated as a function of s with a Lagrange polynomial through the
# `order+1` grid planes straddling s (a symmetric stencil "to either side",
# shifted inward near the ends of the table).  The interpolated values define
# a single virtual plane at s, which is then handed to gg_evaluate.
#
# Because ∂A/∂s is still formed from the field-expansion derivative structure
# (a(n,m) → a(n,m+1)) of the interpolated coefficients, the curl identity
# B = ∇×A continues to hold exactly at the interpolated point.
#
#   res    : NamedTuple from gg_load_result (or gg_fit.jl's `result`)
#   x, y   : absolute transverse coordinates (res.origin subtracted internally)
#   s      : absolute longitudinal coordinate
#   order  : interpolation polynomial degree (default 3 = cubic, 2 planes each
#            side).  Reduced automatically when fewer planes are available.
#
# Returns (B, A, dA) exactly as gg_evaluate.
# ---------------------------------------------------------------------------
# Interpolate every stored GG quantity onto a single virtual plane at s,
# using a Lagrange polynomial of the given order through the grid planes
# straddling s (symmetric stencil, shifted inward near the table ends).
function _interp_res(res, s::Real, order::Integer)
    z = res.z_base
    P = length(z)
    npts = clamp(order + 1, 1, P)

    i0    = searchsortedlast(z, s)                       # z[i0] <= s < z[i0+1]
    start = clamp(i0 - (npts ÷ 2) + 1, 1, P - npts + 1)
    idx   = start:(start + npts - 1)
    nodes = @view z[idx]

    L = ones(Float64, npts)                              # Lagrange weights at s
    for k in 1:npts, j in 1:npts
        j == k && continue
        L[k] *= (s - nodes[j]) / (nodes[k] - nodes[j])
    end
    interp(vec) = sum(L[k] * vec[idx[k]] for k in 1:npts)

    a2  = Dict(key => [interp(v)] for (key, v) in res.a)
    b2  = Dict(key => [interp(v)] for (key, v) in res.b)
    bs2 = Dict(key => [interp(v)] for (key, v) in res.bs)

    return (; z_base = [float(s)], a = a2, b = b2, bs = bs2,
              h = res.h, origin = res.origin,
              r0_grid = res.r0_grid, dz_grid = res.dz_grid,
              m_max = res.m_max, rms_plane = [NaN])
end

function gg_evaluate_at(res, x::Real, y::Real, s::Real; order::Integer=3)
    return gg_evaluate(_interp_res(res, s, order), 1, x, y)
end

# ---------------------------------------------------------------------------
# Field-expansion coefficients  B_c(x,y,s) = Σ_{i,j} C_{c,i,j}(s) xⁱ yʲ.
# Returns full _NMAX×_NMAX arrays summed over the a, b, bs parts.
# ---------------------------------------------------------------------------
function _field_C(res, ip::Integer)
    h = res.h
    aval(n, m) = (m >= 0 && haskey(res.a, (n, m))) ? res.a[(n, m)][ip] : 0.0
    bval(n, m) = (m >= 0 && haskey(res.b, (n, m))) ? res.b[(n, m)][ip] : 0.0
    bsval(m)   = (m >= 0 && haskey(res.bs, m))     ? res.bs[m][ip]     : 0.0
    Cx = _accum(Bx_a, k -> aval(k...), h) .+ _accum(Bx_b, k -> bval(k...), h) .+ _accum(Bx_bs, m -> bsval(m), h)
    Cy = _accum(By_a, k -> aval(k...), h) .+ _accum(By_b, k -> bval(k...), h) .+ _accum(By_bs, m -> bsval(m), h)
    Cs = _accum(Bs_a, k -> aval(k...), h) .+ _accum(Bs_b, k -> bval(k...), h) .+ _accum(Bs_bs, m -> bsval(m), h)
    return Cx, Cy, Cs
end

# Trim three coefficient arrays to the smallest (x,y) extent holding every
# nonzero entry, so the returned matrices are indexed C[i+1, j+1] = C_{c,i,j}.
function _trim3(Cx, Cy, Cs)
    pmax = 1; qmax = 1
    for K in (Cx, Cy, Cs), j in 1:_NMAX, i in 1:_NMAX
        if K[i, j] != 0.0
            pmax = max(pmax, i); qmax = max(qmax, j)
        end
    end
    return Cx[1:pmax, 1:qmax], Cy[1:pmax, 1:qmax], Cs[1:pmax, 1:qmax]
end

# ---------------------------------------------------------------------------
# C coefficients at a grid plane.
#
#   res : NamedTuple from gg_load_result (or gg_fit.jl's `result`)
#   ip  : 1-based plane index into res.z_base
#
# Returns (Cx, Cy, Cs); each is a matrix with C[i+1, j+1] = C_{c,i,j}, the
# coefficient of xⁱ yʲ in that field component at the plane.
# ---------------------------------------------------------------------------
function gg_coefficients(res, ip::Integer)
    return _trim3(_field_C(res, ip)...)
end

# ---------------------------------------------------------------------------
# C coefficients at an arbitrary s, via the same Lagrange interpolation of the
# GG quantities used by gg_evaluate_at.  Returns (Cx, Cy, Cs) as above.
# ---------------------------------------------------------------------------
function gg_coefficients_at(res, s::Real; order::Integer=3)
    return _trim3(_field_C(_interp_res(res, s, order), 1)...)
end
