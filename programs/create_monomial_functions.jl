using Symbolics

# ---------------------------------------------------------------------------
# Reproduces and extends Table 1 of Van der Schueren et al., IPAC'24):
# Taylor expansion coefficients of B_x, B_y, B_s in x and y, up to total
# monomial degree MAXTOT, expressed in terms of a_n(s), b_n(s), b_s(s), g_ref
# and their s-derivatives.
# ---------------------------------------------------------------------------

const MAXTOT = parse(Int, get(ENV, "MAXTOT", "12"))     # max total degree p+q of x^p y^q monomials
const N = MAXTOT + 2  # truncation order in x (degrees 0..N-1)

@variables s
Ds = Differential(s)

# Symbolic functions a_n(s), b_n(s), a_0(s)
for n in 0:13
    @eval @variables $(Symbol("a$n"))(s)
    if n >= 1
        @eval @variables $(Symbol("b$n"))(s)
    end
end

@variables g_ref  # g_ref is taken to be constant (independent of s)

avars = [eval(Symbol("a$n")) for n in 0:13]
bvars = [eval(Symbol("b$n")) for n in 1:13]

# phi_0(x,s) = -a_0(s) - sum_{n=1}^{13} a_n(s) x^n / n!

phi0 = Vector{Num}(undef, N)
fill!(phi0, Num(0))
phi0[1] = -avars[1]            # x^0 coefficient: -a_0(s)
for n in 1:13
    if n <= N - 1
        phi0[n+1] = -avars[n+1] / factorial(n)
    end
end

# phi_1(x,s) = - sum_{n=1}^{13} b_n(s) x^{n-1} / (n-1)!

phi1 = Vector{Num}(undef, N)
fill!(phi1, Num(0))
for n in 1:13
    if n - 1 <= N - 1
        phi1[n] = -bvars[n] / factorial(n - 1)
    end
end

# ---------------------------------------------------------------------------
# Truncated power-series operations on length-N coefficient vectors
# (index i in 1..N corresponds to x^{i-1})
# ---------------------------------------------------------------------------

# d/dx
function dx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    for i in 1:N-1
        q[i] = i * p[i+1]
    end
    q[N] = Num(0)
    return q
end

# multiply by (1 + g_ref*x)
function mul1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    q[1] = p[1]
    for i in 2:N
        q[i] = p[i] + g_ref * p[i-1]
    end
    return q
end

# multiply by 1/(1+g_ref*x) = sum_j (-g_ref)^j x^j  (truncated)
function mulinv1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    hpow = Vector{Num}(undef, N)
    hpow[1] = Num(1)
    for j in 2:N
        hpow[j] = -g_ref * hpow[j-1]
    end
    for i in 1:N
        acc = Num(0)
        for j in 1:i
            acc += hpow[j] * p[i-j+1]
        end
        q[i] = acc
    end
    return q
end

# d/ds, with full expansion of derivatives
function dsarr(p::Vector{Num})
    return [expand_derivatives(Ds(x)) for x in p]
end

# ---------------------------------------------------------------------------
# Recurrence: phi_{i+2} = -1/(1+hx) [ d_x((1+hx) d_x phi_i) + d_s( 1/(1+hx) d_s phi_i ) ]
# ---------------------------------------------------------------------------

phi = Dict{Int,Vector{Num}}()
phi[0] = phi0
phi[1] = phi1

for i in 0:(MAXTOT-1)
    println("computing phi[$(i+2)] ...")
    p = phi[i]
    term1 = mul1phx(dx(dx(p))) .+ g_ref .* dx(p)
    term2 = dsarr(mulinv1phx(dsarr(p)))
    pnew = -mulinv1phx(term1 .+ term2)
    phi[i+2] = [expand(x) for x in pnew]
end

println("phi computed up to order ", MAXTOT + 1)

# s-derivatives of phi_i, needed for B_s
g = Dict{Int,Vector{Num}}()
for i in 0:MAXTOT
    g[i] = dsarr(phi[i])
end

# ---------------------------------------------------------------------------
# Field expansion coefficients T_{p,q} of x^p y^q in B_x, B_y, B_s
# ---------------------------------------------------------------------------

TBx = Dict{Tuple{Int,Int},Num}()
TBy = Dict{Tuple{Int,Int},Num}()
TBs = Dict{Tuple{Int,Int},Num}()

hpow_static = Vector{Num}(undef, N)
hpow_static[1] = Num(1)
for j in 2:N
    hpow_static[j] = -g_ref * hpow_static[j-1]
end

for q in 0:MAXTOT
    dphiq = dx(phi[q])
    for p in 0:(MAXTOT - q)
        # B_x
        TBx[(p,q)] = expand(-dphiq[p+1] / factorial(q))
        # B_y
        TBy[(p,q)] = expand(-phi[q+1][p+1] / factorial(q))
        # B_s
        acc = Num(0)
        for j in 0:p
            acc += hpow_static[j+1] * g[q][p-j+1]
        end
        TBs[(p,q)] = expand(-acc / factorial(q))
    end
end

println("coefficients computed")

# ---------------------------------------------------------------------------
# Vector potential monomial coefficients T_{p,q} of x^p y^q in A_x, A_y, A_s
#
# Following papers/vector-potential, A is split into three pieces according to
# which GG family the field coefficient C depends on:
#   alpha : part depending on a_n (n >= 1)
#   beta  : part depending on b_n
#   gamma : part depending on b_s  (= the derivatives of a_0)
# Using the gauge A_y = 0 for the alpha/beta pieces and A_s = 0 for gamma:
#
#   A_x = - sum 1/(j+1) (alpha+beta)_{s,i,j} x^i y^{j+1}
#         + (1+hx) sum [int ds gamma_{y,i,j}] x^i y^j
#   A_y = - (1+hx) sum [int ds gamma_{x,i,j}] x^i y^j
#   A_s =   sum 1/(j+1) (alpha+beta)_{x,i,j} x^i y^{j+1}
#         - 1/(1+hx) sum_i beta_{y,i,0} ( x^{i+1}/(i+1) + g_ref x^{i+2}/(i+2) )
#
# (alpha+beta)_{c} is the part of C_c not involving b_s; gamma_c is the b_s
# part.  int ds lowers the b_s derivative order by one, which is always well
# defined because the b_s parts of B_x and B_y start at order m = 1.
# ---------------------------------------------------------------------------

println("computing vector potential coefficients ...")

const MDER = MAXTOT + 4

function nth_ds(v, m)
    r = v
    for _ in 1:m
        r = expand_derivatives(Ds(r))
    end
    return r
end

# Substitution dictionaries that project an expression onto one GG family by
# zeroing the others (the field coefficients are linear in the GG functions).
zero_a0    = Dict{Num,Num}()   # zero a_0 and its s-derivatives
zero_apos  = Dict{Num,Num}()   # zero a_1 .. a_13 and derivatives
zero_b     = Dict{Num,Num}()   # zero b_1 .. b_13 and derivatives
zero_a_all = Dict{Num,Num}()   # zero a_0 .. a_13 and derivatives
for m in 0:MDER
    zero_a0[nth_ds(avars[1], m)] = Num(0)
end
for n in 1:13, m in 0:MDER
    zero_apos[nth_ds(avars[n+1], m)] = Num(0)
    zero_b[nth_ds(bvars[n], m)]      = Num(0)
end
for n in 0:13, m in 0:MDER
    zero_a_all[nth_ds(avars[n+1], m)] = Num(0)
end
zero_not_a0 = merge(zero_apos, zero_b)        # keep only a_0  (the b_s family)

# int ds on a b_s (a_0-derivative) expression:  D^k(a_0) -> D^{k-1}(a_0).
intds_a0 = Dict{Num,Num}()
for k in 1:MDER
    intds_a0[nth_ds(avars[1], k)] = nth_ds(avars[1], k - 1)
end

ab_part(e) = substitute(e, zero_a0)       # a_n (n>=1) + b_n  part of C
b_part(e)  = substitute(e, zero_a_all)    # b_n               part of C
bs_part(e) = substitute(e, zero_not_a0)   # b_s               part of C
intds(e)   = substitute(e, intds_a0)

# Projected / integrated field pieces used in the A construction.
TBx_ab = Dict{Tuple{Int,Int},Num}()   # (alpha+beta)_x
TBs_ab = Dict{Tuple{Int,Int},Num}()   # (alpha+beta)_s
Igy    = Dict{Tuple{Int,Int},Num}()   # int ds gamma_y  (b_s part of B_y)
Igx    = Dict{Tuple{Int,Int},Num}()   # int ds gamma_x  (b_s part of B_x)
for q in 0:MAXTOT, p in 0:(MAXTOT-q)
    TBx_ab[(p,q)] = ab_part(TBx[(p,q)])
    TBs_ab[(p,q)] = ab_part(TBs[(p,q)])
    Igy[(p,q)]    = intds(bs_part(TBy[(p,q)]))
    Igx[(p,q)]    = intds(bs_part(TBx[(p,q)]))
end
getD(D, p, q) = (p >= 0 && q >= 0 && haskey(D, (p,q))) ? D[(p,q)] : Num(0)

# Midplane-correction polynomial P(x) = sum_i beta_{y,i,0}(x^{i+1}/(i+1)+g_ref x^{i+2}/(i+2))
# stored as a length-N coefficient vector (index k <-> x^{k-1}); As_corr = -P/(1+hx).
Pvec = fill(Num(0), N)
for i in 0:MAXTOT
    byi0 = b_part(TBy[(i,0)])
    i + 2 <= N && (Pvec[i+2] += byi0 * (1 // (i + 1)))   # x^{i+1}
    i + 3 <= N && (Pvec[i+3] += byi0 * g_ref * (1 // (i + 2)))   # x^{i+2}
end
As_corr = (-1) .* mulinv1phx(Pvec)

TAx = Dict{Tuple{Int,Int},Num}()
TAy = Dict{Tuple{Int,Int},Num}()
TAs = Dict{Tuple{Int,Int},Num}()
for q in 0:MAXTOT, p in 0:(MAXTOT-q)
    ax = getD(Igy, p, q) + g_ref * getD(Igy, p - 1, q)
    q >= 1 && (ax += -(1 // q) * getD(TBs_ab, p, q - 1))
    TAx[(p,q)] = expand(ax)

    ay = -(getD(Igx, p, q) + g_ref * getD(Igx, p - 1, q))
    TAy[(p,q)] = expand(ay)

    as = q == 0 ? As_corr[p+1] : (1 // q) * getD(TBx_ab, p, q - 1)
    TAs[(p,q)] = expand(as)
end

println("vector potential coefficients computed")

# ---------------------------------------------------------------------------
# Convert to a(n,m), b(n,m), bs(m) notation
# ---------------------------------------------------------------------------

function rewrite_notation(expr)
    str = string(expr)

    # a_0(s) derivatives: D^m(a0(s)) -> bs(m-1)
    str = replace(str, r"Differential\(s, (\d+)\)\(a0\(s\)\)" => function(m)
        mm = match(r"Differential\(s, (\d+)\)\(a0\(s\)\)", m)
        k = parse(Int, mm.captures[1]) - 1
        "bs($k)"
    end)

    # general D^m(a_n(s)) -> a(n,m), D^m(b_n(s)) -> b(n,m)
    str = replace(str, r"Differential\(s, (\d+)\)\(a(\d+)\(s\)\)" => SubstitutionString("a(\\2,\\1)"))
    str = replace(str, r"Differential\(s, (\d+)\)\(b(\d+)\(s\)\)" => SubstitutionString("b(\\2,\\1)"))

    # plain (order 0) occurrences
    str = replace(str, r"a(\d+)\(s\)" => SubstitutionString("a(\\1,0)"))
    str = replace(str, r"b(\d+)\(s\)" => SubstitutionString("b(\\1,0)"))

    # sanity check: a0 should never appear undifferentiated
    if occursin("a(0,0)", str)
        @warn "a(0,0) found in expression!" str
    end

    return str
end

# ---------------------------------------------------------------------------
# Write out a markdown table
# ---------------------------------------------------------------------------

open(joinpath(@__DIR__, "..", "tables", "monomial_functions.jl"), "w") do io
    println(io, "# Extended Table 1: Taylor expansion of the magnetic field and the")
    println(io, "# vector potential (constant g_ref)")
    println(io, "")
    println(io, "# Coefficients of the monomials x^p y^q in B_x, B_y, B_s and in the")
    println(io, "# vector potential A_x, A_y, A_s (B = curl A), for total degree")
    println(io, "# p+q <= $MAXTOT, assuming the curvature g_ref is constant (g_ref' = 0).")
    println(io, "# Notation: a(n,m) = d^m a_n/ds^m, b(n,m) = d^m b_n/ds^m,")
    println(io, "# bs(m) = d^m b_s/ds^m.")
    println(io, "")
    for q in 0:MAXTOT
        for p in 0:(MAXTOT-q)
            println(io, "## x^$p y^$q")
            println(io, "")
            println(io, "Bx_coef[($p,$q)] = ", rewrite_notation(TBx[(p,q)]))
            println(io, "")
            println(io, "By_coef[($p,$q)] = ", rewrite_notation(TBy[(p,q)]))
            println(io, "")
            println(io, "Bs_coef[($p,$q)] = ", rewrite_notation(TBs[(p,q)]))
            println(io, "")
            println(io, "Ax_coef[($p,$q)] = ", rewrite_notation(TAx[(p,q)]))
            println(io, "")
            println(io, "Ay_coef[($p,$q)] = ", rewrite_notation(TAy[(p,q)]))
            println(io, "")
            println(io, "As_coef[($p,$q)] = ", rewrite_notation(TAs[(p,q)]))
            println(io, "")
        end
    end
end

println("done")
