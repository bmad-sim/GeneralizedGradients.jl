using Symbolics

# ---------------------------------------------------------------------------
# Create a file gg_coef_table.jl 
# Similar to monomial_functions.jl:
# For each field component (Bx, By, Bs), and for each function a, b, bs and derivatives,
# output a vector of coefficients that contribute.
#
# Notation:
#  b(n,m) = (d/ds)^m (b_n), a(n,m) = (d/ds)^m a_n, and bs(m) = (d/ds)^m bs
#
# Output: There are 9 Dicts labeled:
#   Bx_a[(n,m)], Bx_b[(n,m)], Bx_bs[(n,m)], 
#   By_a[(n,m)], By_b[(n,m)], By_bs[(n,m)],
#   Bs_a[(n,m)], Bs_b[(n,m)], Bs_bs[(n,m)]
#
# Example: By_b[(n,m)] = [(coef, p, q, r), ...]
# means contribution by b(n,m) is:
#   By += coef * g_ref^r * x^p * y^q * b(n,m)
# ---------------------------------------------------------------------------

const MAXTOT = parse(Int, get(ENV, "MAXTOT", "12"))
const N = MAXTOT + 2

@variables s
Ds = Differential(s)

for n in 0:13
    @eval @variables $(Symbol("a$n"))(s)
    if n >= 1
        @eval @variables $(Symbol("b$n"))(s)
    end
end

@variables g_ref

avars = [eval(Symbol("a$n")) for n in 0:13]   # avars[k+1] = a_k(s)
bvars = [eval(Symbol("b$n")) for n in 1:13]   # bvars[n]   = b_n(s)

# ---------------------------------------------------------------------------
# phi_0, phi_1 seed functions
# ---------------------------------------------------------------------------

phi0 = Vector{Num}(undef, N)
fill!(phi0, Num(0))
phi0[1] = -avars[1]
for n in 1:13
    n <= N - 1 && (phi0[n+1] = -avars[n+1] / factorial(n))
end

phi1 = Vector{Num}(undef, N)
fill!(phi1, Num(0))
for n in 1:13
    n - 1 <= N - 1 && (phi1[n] = -bvars[n] / factorial(n - 1))
end

# ---------------------------------------------------------------------------
# Truncated power-series operations (coefficients of x^0..x^{N-1})
# ---------------------------------------------------------------------------

function dx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    for i in 1:N-1; q[i] = i * p[i+1]; end
    q[N] = Num(0)
    return q
end

function mul1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    q[1] = p[1]
    for i in 2:N; q[i] = p[i] + g_ref * p[i-1]; end
    return q
end

function mulinv1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    hpow = Vector{Num}(undef, N)
    hpow[1] = Num(1)
    for j in 2:N; hpow[j] = -g_ref * hpow[j-1]; end
    for i in 1:N
        acc = Num(0)
        for j in 1:i; acc += hpow[j] * p[i-j+1]; end
        q[i] = acc
    end
    return q
end

function dsarr(p::Vector{Num})
    return [expand_derivatives(Ds(x)) for x in p]
end

# ---------------------------------------------------------------------------
# Recurrence: phi_{i+2} = -1/(1+hx)[d_x((1+hx)d_x phi_i) + d_s(1/(1+hx) d_s phi_i)]
# ---------------------------------------------------------------------------

phi = Dict{Int,Vector{Num}}()
phi[0] = phi0
phi[1] = phi1

for i in 0:(MAXTOT-1)
    println("computing phi[$(i+2)] ...")
    p = phi[i]
    term1 = mul1phx(dx(dx(p))) .+ g_ref .* dx(p)
    term2 = dsarr(mulinv1phx(dsarr(p)))
    pnew  = -mulinv1phx(term1 .+ term2)
    phi[i+2] = [expand(x) for x in pnew]
end

println("phi computed up to order ", MAXTOT + 1)

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
for j in 2:N; hpow_static[j] = -g_ref * hpow_static[j-1]; end

for q in 0:MAXTOT
    dphiq = dx(phi[q])
    for p in 0:(MAXTOT - q)
        TBx[(p,q)] = expand(-dphiq[p+1] / factorial(q))
        TBy[(p,q)] = expand(-phi[q+1][p+1] / factorial(q))
        acc = Num(0)
        for j in 0:p; acc += hpow_static[j+1] * g[q][p-j+1]; end
        TBs[(p,q)] = expand(-acc / factorial(q))
    end
end

println("field coefficients computed")

# ---------------------------------------------------------------------------
# Build inverse coefficient table (full g_ref dependence)
# ---------------------------------------------------------------------------

# Compute the m-th s-derivative of v by iterative application of Ds,
# matching exactly the form produced by dsarr in the phi recurrence.
function nth_ds_deriv(v, m)
    result = v
    for _ in 1:m
        result = expand_derivatives(Ds(result))
    end
    return result
end

# ---------------------------------------------------------------------------
# Vector potential coefficients T_{p,q} of x^p y^q in A_x, A_y, A_s
# (B = curl A in Frenet coordinates).  See papers/vector-potential.
#
# C is split by GG family: alpha (a_n, n>=1), beta (b_n), gamma (b_s = a_0
# derivatives).  Gauge A_y = 0 for alpha/beta, A_s = 0 for gamma:
#
#   A_x = - sum 1/(j+1) (alpha+beta)_{s,i,j} x^i y^{j+1}
#         + (1+hx) sum [int ds gamma_{y,i,j}] x^i y^j
#   A_y = - (1+hx) sum [int ds gamma_{x,i,j}] x^i y^j
#   A_s =   sum 1/(j+1) (alpha+beta)_{x,i,j} x^i y^{j+1}
#         - 1/(1+hx) sum_i beta_{y,i,0} ( x^{i+1}/(i+1) + g_ref x^{i+2}/(i+2) )
# ---------------------------------------------------------------------------

println("computing vector potential coefficients ...")

const MDER = MAXTOT + 4

# Project an expression onto one GG family by zeroing the others (the field
# coefficients are linear in the GG functions); int ds lowers a b_s order.
zero_a0    = Dict{Num,Num}()   # zero a_0 and its s-derivatives
zero_apos  = Dict{Num,Num}()   # zero a_1 .. a_13 and derivatives
zero_b     = Dict{Num,Num}()   # zero b_1 .. b_13 and derivatives
zero_a_all = Dict{Num,Num}()   # zero a_0 .. a_13 and derivatives
for m in 0:MDER
    zero_a0[nth_ds_deriv(avars[1], m)] = Num(0)
end
for n in 1:13, m in 0:MDER
    zero_apos[nth_ds_deriv(avars[n+1], m)] = Num(0)
    zero_b[nth_ds_deriv(bvars[n], m)]      = Num(0)
end
for n in 0:13, m in 0:MDER
    zero_a_all[nth_ds_deriv(avars[n+1], m)] = Num(0)
end
zero_not_a0 = merge(zero_apos, zero_b)        # keep only a_0  (the b_s family)

intds_a0 = Dict{Num,Num}()                    # int ds: D^k(a_0) -> D^{k-1}(a_0)
for k in 1:MDER
    intds_a0[nth_ds_deriv(avars[1], k)] = nth_ds_deriv(avars[1], k - 1)
end

ab_part(e) = substitute(e, zero_a0)       # a_n (n>=1) + b_n  part of C
b_part(e)  = substitute(e, zero_a_all)    # b_n               part of C
bs_part(e) = substitute(e, zero_not_a0)   # b_s               part of C
intds(e)   = substitute(e, intds_a0)

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

# Midplane-correction polynomial P(x); As_corr = -P/(1+hx) (length-N x-vector).
Pvec = fill(Num(0), N)
for i in 0:MAXTOT
    byi0 = b_part(TBy[(i,0)])
    i + 2 <= N && (Pvec[i+2] += byi0 * (1 // (i + 1)))       # x^{i+1}
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

Dh = Differential(g_ref)

# Build the zero-substitution dictionary once: every symbolic function and
# every s-derivative (up to MAXTOT+4) mapped to 0.  g_ref is NOT zeroed here.
println("building zero substitution dict ...")
all_zero = Dict{Num,Num}()

for n in 0:13
    v = avars[n+1]
    all_zero[v] = Num(0)
    for m in 1:(MAXTOT+4)
        all_zero[nth_ds_deriv(v, m)] = Num(0)
    end
end
for n in 1:13
    v = bvars[n]
    all_zero[v] = Num(0)
    for m in 1:(MAXTOT+4)
        all_zero[nth_ds_deriv(v, m)] = Num(0)
    end
end

println("zero dict built with $(length(all_zero)) entries")

# Helper: convert a Symbolics scalar to Rational{Int}.
function to_rat(v)::Rational{Int}
    v isa Rational && return Rational{Int}(v)
    v isa Integer  && return Rational{Int}(v, 1)
    return rationalize(Int, Float64(v))
end

# Extract the full g_ref-polynomial coefficient of sym in expr with all other
# symbolic functions zeroed out.  Returns a Dict{Int,Rational{Int}} mapping
# g_ref-power => coefficient.  Mutates all_zero temporarily.
function coeff_poly_h(expr, sym, max_h_power)
    all_zero[sym] = Num(1)
    poly = substitute(expr, all_zero)   # polynomial in g_ref
    all_zero[sym] = Num(0)

    result = Dict{Int,Rational{Int}}()
    curr   = poly
    fk     = 1   # factorial(k)
    h0     = Dict(g_ref => Num(0))
    for k in 0:max_h_power
        k > 0 && (fk *= k)
        v = Symbolics.value(substitute(curr, h0))
        r = to_rat(v)
        r != 0 && (result[k] = r // fk)
        k < max_h_power && (curr = expand(expand_derivatives(Dh(curr))))
    end
    return result
end

# Format a rational for Julia output: suppress denominator when it is 1.
function fmt_rat(r::Rational{Int})
    denominator(r) == 1 ? "$(numerator(r))" : "($(numerator(r))//$(denominator(r)))"
end

# Format a vector of (coeff, p, q, h_power) tuples as a Julia-compatible literal.
function fmt_terms(terms)
    parts = String[]
    for (c, p, q, k) in terms
        push!(parts, "($(fmt_rat(c)), $p, $q, $k)")
    end
    return "[" * join(parts, ", ") * "]"
end

# Maximum g_ref power that can appear in any T[(p,q)] coefficient.
const MAX_H = MAXTOT + 2

# ---------------------------------------------------------------------------
# Collect contributions and write output
# ---------------------------------------------------------------------------

outfile = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
open(outfile, "w") do io
    println(io, "# Inverse field and vector-potential coefficient table (full g_ref dependence)")
    println(io, "#")
    println(io, "# By_b[(n,m)] = [(c, p, q, k), ...]  means  By += c * g_ref^k * x^p * y^q * b(n,m)")
    println(io, "# Similarly for Bx_b, Bs_b, By_a, Bx_a, Bs_a, By_bs, Bx_bs, Bs_bs and for")
    println(io, "# the vector potential A (B = curl A):  Ax_a, Ax_b, Ax_bs, Ay_a, Ay_b,")
    println(io, "# Ay_bs, As_a, As_b, As_bs (same meaning, e.g. Ax_b[(n,m)] -> Ax += ...).")
    println(io, "# Notation: b(n,m) = d^m b_n/ds^m,  a(n,m) = d^m a_n/ds^m,")
    println(io, "#           bs(m)  = d^{m+1} a_0/ds^{m+1}")
    println(io)

    for comp in ("Bx", "By", "Bs", "Ax", "Ay", "As")
        println(io, "$(comp)_a  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
        println(io, "$(comp)_b  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
        println(io, "$(comp)_bs = Dict{Int64, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    end
    println(io)

    # --- b_n(s) functions, n = 1..13, m = 0..MAXTOT ---
    println(io, "# --- b(n,m) contributions ---")
    println(io)
    for (T, prefix) in [(TBy, "By_b"), (TBx, "Bx_b"), (TBs, "Bs_b"),
                        (TAy, "Ay_b"), (TAx, "Ax_b"), (TAs, "As_b")]
        print("Processing $(prefix) ...")
        for n in 1:13, m in 0:MAXTOT
            sym   = nth_ds_deriv(bvars[n], m)
            terms = Tuple{Rational{Int},Int,Int,Int}[]
            for q in 0:MAXTOT, p in 0:(MAXTOT-q)
                hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
                for k in sort(collect(keys(hc)))
                    push!(terms, (hc[k], p, q, k))
                end
            end
            isempty(terms) || println(io, "$(prefix)[($n,$m)] = $(fmt_terms(terms))")
        end
        println(io)
        println("done")
    end

    # --- a_n(s) functions, n = 1..13, m = 0..MAXTOT ---
    println(io, "# --- a(n,m) contributions ---")
    println(io)
    for (T, prefix) in [(TBy, "By_a"), (TBx, "Bx_a"), (TBs, "Bs_a"),
                        (TAy, "Ay_a"), (TAx, "Ax_a"), (TAs, "As_a")]
        print("Processing $(prefix) ...")
        for n in 1:13, m in 0:MAXTOT
            sym   = nth_ds_deriv(avars[n+1], m)   # avars[n+1] = a_n(s)
            terms = Tuple{Rational{Int},Int,Int,Int}[]
            for q in 0:MAXTOT, p in 0:(MAXTOT-q)
                hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
                for k in sort(collect(keys(hc)))
                    push!(terms, (hc[k], p, q, k))
                end
            end
            isempty(terms) || println(io, "$(prefix)[($n,$m)] = $(fmt_terms(terms))")
        end
        println(io)
        println("done")
    end

    # --- bs(m) = d^{m+1} a_0/ds^{m+1}, m = 0..MAXTOT ---
    println(io, "# --- bs(m) contributions ---")
    println(io)
    for (T, prefix) in [(TBy, "By_bs"), (TBx, "Bx_bs"), (TBs, "Bs_bs"),
                        (TAy, "Ay_bs"), (TAx, "Ax_bs"), (TAs, "As_bs")]
        print("Processing $(prefix) ...")
        for m in 0:MAXTOT
            sym   = nth_ds_deriv(avars[1], m + 1)  # (m+1)-th deriv of a_0 = bs(m)
            terms = Tuple{Rational{Int},Int,Int,Int}[]
            for q in 0:MAXTOT, p in 0:(MAXTOT-q)
                hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
                for k in sort(collect(keys(hc)))
                    push!(terms, (hc[k], p, q, k))
                end
            end
            isempty(terms) || println(io, "$(prefix)[$m] = $(fmt_terms(terms))")
        end
        println(io)
        println("done")
    end
end

println("done, written to $outfile")
