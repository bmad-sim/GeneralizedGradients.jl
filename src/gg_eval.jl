# ---------------------------------------------------------------------------
# gg_eval.jl
#
# Evaluate the magnetic field B = (Bx, By, Bs) and the vector potential
# A = (Ax, Ay, As) -- together with the 3x3 Jacobian of A with respect to
# (x, y, s) -- at a chosen base plane and transverse position, given the
# generalized-gradient (GG) coefficients produced by src/gg_fit.jl.
#
# Field expansion (tables/field_function_table.jl):
#   B_c(x,y,s) = Σ_{(n,m)} CS_c,a(n,m;x,y)·a(n,m)
#              + Σ_{(n,m)} CS_c,b(n,m;x,y)·b(n,m)
#              + Σ_{m}     CS_c,bs(m;x,y)·bs(m)
# with a(n,m)=dᵐa_n/dsᵐ, b(n,m)=dᵐb_n/dsᵐ, bs(m)=dᵐ⁺¹a_0/dsᵐ⁺¹ = dᵐb_s/dsᵐ,
# and CS_c,f = Σ (coeff·hᵏ·xᵖ·yᵠ) the sum of that function's table entries.
#
# Vector potential (papers/vector_potential.tex).  The field is split linearly
# into an a_n part (α), a b_n part (β), and a b_s part (γ); each gets its own
# gauge and the results are summed:
#
#   α (gauge A_y=0):
#     Ax = -Σ 1/(j+1) α_{s,i,j} xⁱ y^{j+1}
#     As =  Σ 1/(j+1) α_{x,i,j} xⁱ y^{j+1}
#   β (gauge A_y=0, + midplane-B_y term):
#     Ax = -Σ 1/(j+1) β_{s,i,j} xⁱ y^{j+1}
#     As =  Σ 1/(j+1) β_{x,i,j} xⁱ y^{j+1}
#          - 1/(1+hx) Σ_i β_{y,i,0} ( x^{i+1}/(i+1) + h x^{i+2}/(i+2) )
#   γ (gauge A_s=0):
#     Ax =  (1+hx) Σ [∫ds γ_{y,i,j}] xⁱ yʲ
#     Ay = -(1+hx) Σ [∫ds γ_{x,i,j}] xⁱ yʲ
#
# where α_{c,i,j}, β_{c,i,j}, γ_{c,i,j} are the a/b/bs parts of the field
# coefficient of xⁱ yʲ in B_c at the plane, and ∫ds γ lowers the b_s derivative
# order by one ( ∫ds bs(m) = bs(m-1) ).  s-derivatives of A are obtained by
# bumping the GG derivative order (a(n,m)→a(n,m+1), etc.) and, for the γ piece,
# by ∂_s∫ds γ = γ.
# ---------------------------------------------------------------------------

using JLD2

const _TABLE_FILE = joinpath(@__DIR__, "..", "tables", "field_function_table.jl")
include(_TABLE_FILE)   # Bx_a By_a Bs_a  Bx_b By_b Bs_b  Bx_bs By_bs Bs_bs

# Working size for the truncated (x,y) coefficient arrays.  The table is built
# to total monomial degree MAXTOT (12); the A construction shifts powers by at
# most +2, so a little headroom is enough.
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

# y-integration:  Σ K_{i,q} xⁱ yᵠ  ->  Σ K_{i,q}/(q+1) xⁱ y^{q+1}
function _yint(K)
    Y = _newK()
    for i in 1:_NMAX, j in 1:_NMAX-1
        K[i, j] == 0.0 && continue
        Y[i, j+1] += K[i, j] / j        # j == q+1
    end
    return Y
end

# midplane x-antiderivative used by the β A_s term, acting on the j=0 column:
#   Σ_i K_{i,0} ( x^{i+1}/(i+1) + h x^{i+2}/(i+2) )
function _midpoly(K, h)
    P = _newK()
    for i in 1:_NMAX
        c = K[i, 1]                      # β_{y,i-1,0}
        c == 0.0 && continue
        ip = i - 1
        ip + 1 <= _NMAX - 1 && (P[ip+2, 1] += c / (ip + 1))
        ip + 2 <= _NMAX - 1 && (P[ip+3, 1] += h * c / (ip + 2))
    end
    return P
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

# Evaluate a sum of terms, each tagged with its (1+hx) prefactor:
#   :plain -> P,   :mul -> (1+hx)·P,   :div -> P/(1+hx).
# Returns (value, ∂/∂x, ∂/∂y).
function _eval_terms(terms, x, y, h)
    val = 0.0; vx = 0.0; vy = 0.0
    g = 1 + h * x
    for (gtype, K) in terms
        p, px, py = _polyval(K, x, y)
        if gtype === :plain
            val += p;        vx += px;                vy += py
        elseif gtype === :mul
            val += g * p;    vx += h * p + g * px;    vy += g * py
        else # :div
            val += p / g;    vx += (px * g - p * h) / g^2; vy += py / g
        end
    end
    return val, vx, vy
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

    # --- field coefficient arrays, split into α (a), β (b), γ (bs) parts ---
    ax = _accum(Bx_a, k -> aval(k...), h); ay = _accum(By_a, k -> aval(k...), h); as = _accum(Bs_a, k -> aval(k...), h)
    bx = _accum(Bx_b, k -> bval(k...), h); by = _accum(By_b, k -> bval(k...), h); bs = _accum(Bs_b, k -> bval(k...), h)
    gx = _accum(Bx_bs, m -> bsval(m), h);  gy = _accum(By_bs, m -> bsval(m), h);  gs = _accum(Bs_bs, m -> bsval(m), h)

    # s-derivatives of the α and β parts (bump derivative order).
    dax = _accum(Bx_a, k -> aval(k[1], k[2]+1), h); das = _accum(Bs_a, k -> aval(k[1], k[2]+1), h)
    dbx = _accum(Bx_b, k -> bval(k[1], k[2]+1), h); dbs = _accum(Bs_b, k -> bval(k[1], k[2]+1), h)
    dby = _accum(By_b, k -> bval(k[1], k[2]+1), h)

    # γ s-integrals:  ∫ds bs(m) = bs(m-1).
    igx = _accum(Bx_bs, m -> bsval(m-1), h)
    igy = _accum(By_bs, m -> bsval(m-1), h)

    # --- field ---
    Bx = _polyval(ax .+ bx .+ gx, x, y)[1]
    By = _polyval(ay .+ by .+ gy, x, y)[1]
    Bs = _polyval(as .+ bs .+ gs, x, y)[1]

    # --- vector potential terms ---
    # A_x = -[int_y (α_s+β_s)]        + (1+hx)·int_s γ_y
    Ax_terms  = [(:plain, .-_yint(as .+ bs)), (:mul, igy)]
    # A_y =                              -(1+hx)·int_s γ_x
    Ay_terms  = [(:mul, .-igx)]
    # A_s =  [int_y (α_x+β_x)]        - (1/(1+hx))·midpoly(β_y)
    As_terms  = [(:plain,  _yint(ax .+ bx)), (:div, .-_midpoly(by, h))]

    # ∂_s of each (bumped orders; ∂_s ∫ds γ = γ)
    dAx_terms = [(:plain, .-_yint(das .+ dbs)), (:mul, gy)]
    dAy_terms = [(:mul, .-gx)]
    dAs_terms = [(:plain,  _yint(dax .+ dbx)), (:div, .-_midpoly(dby, h))]

    Axv, Axx, Axy = _eval_terms(Ax_terms, x, y, h)
    Ayv, Ayx, Ayy = _eval_terms(Ay_terms, x, y, h)
    Asv, Asx, Asy = _eval_terms(As_terms, x, y, h)
    dAxv = _eval_terms(dAx_terms, x, y, h)[1]
    dAyv = _eval_terms(dAy_terms, x, y, h)[1]
    dAsv = _eval_terms(dAs_terms, x, y, h)[1]

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
