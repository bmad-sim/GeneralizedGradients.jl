using Symbolics

# ---------------------------------------------------------------------------
# Reproduces and extends Table 1 of TUPS09 (Van der Schueren et al., IPAC'24):
# Taylor expansion coefficients of B_x, B_y, B_s in x and y, up to total
# monomial degree MAXTOT, expressed in terms of a_n(s), b_n(s), b_s(s), h
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

@variable h  # h is taken to be constant (independent of s)

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

# multiply by (1 + h*x)
function mul1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    q[1] = p[1]
    for i in 2:N
        q[i] = p[i] + h * p[i-1]
    end
    return q
end

# multiply by 1/(1+h*x) = sum_j (-h)^j x^j  (truncated)
function mulinv1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    hpow = Vector{Num}(undef, N)
    hpow[1] = Num(1)
    for j in 2:N
        hpow[j] = -h * hpow[j-1]
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
    term1 = mul1phx(dx(dx(p))) .+ h .* dx(p)
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
    hpow_static[j] = -h * hpow_static[j-1]
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

open(joinpath(@__DIR__, "..", "papers", "field_monomial_table_with_finite_h.jl"), "w") do io
    println(io, "# Extended Table 1: Taylor expansion of the magnetic field (constant h)")
    println(io, "")
    println(io, "# Coefficients of the monomials x^p y^q in B_x, B_y, B_s, for total")
    println(io, "# degree p+q <= $MAXTOT, assuming the curvature h is constant (h' = 0).")
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
        end
    end
end

println("done")
