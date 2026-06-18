using Symbolics

# ---------------------------------------------------------------------------
# Inverted form of field_monomial_table.jl:
# For each function b(n,m) = d^m b_n/ds^m, a(n,m) = d^m a_n/ds^m,
# bs(m) = d^{m+1} a_0/ds^{m+1}, output which monomials x^p*y^q it
# contributes to in Bx, By, Bs at h=0.
#
# Output format: By_b[(n,m)] = [(coeff, p, q), ...]
#   means  By += coeff * x^p * y^q * b(n,m)
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

@variables h

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
    for i in 2:N; q[i] = p[i] + h * p[i-1]; end
    return q
end

function mulinv1phx(p::Vector{Num})
    q = Vector{Num}(undef, N)
    hpow = Vector{Num}(undef, N)
    hpow[1] = Num(1)
    for j in 2:N; hpow[j] = -h * hpow[j-1]; end
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
    term1 = mul1phx(dx(dx(p))) .+ h .* dx(p)
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
for j in 2:N; hpow_static[j] = -h * hpow_static[j-1]; end

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
# Build inverse coefficient table (full h dependence)
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

Dh = Differential(h)

# Build the zero-substitution dictionary once: every symbolic function and
# every s-derivative (up to MAXTOT+4) mapped to 0.  h is NOT zeroed here.
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

# Extract the full h-polynomial coefficient of sym in expr with all other
# symbolic functions zeroed out.  Returns a Dict{Int,Rational{Int}} mapping
# h-power => coefficient.  Mutates all_zero temporarily.
function coeff_poly_h(expr, sym, max_h_power)
    all_zero[sym] = Num(1)
    poly = substitute(expr, all_zero)   # polynomial in h
    all_zero[sym] = Num(0)

    result = Dict{Int,Rational{Int}}()
    curr   = poly
    fk     = 1   # factorial(k)
    h0     = Dict(h => Num(0))
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

# Maximum h power that can appear in any T[(p,q)] coefficient.
const MAX_H = MAXTOT + 2

# ---------------------------------------------------------------------------
# Collect contributions and write output
# ---------------------------------------------------------------------------

outfile = joinpath(@__DIR__, "..", "papers", "field_function_table.jl")
open(outfile, "w") do io
    println(io, "# Inverse field coefficient table (full h dependence)")
    println(io, "#")
    println(io, "# By_b[(n,m)] = [(c, p, q, k), ...]  means  By += c * h^k * x^p * y^q * b(n,m)")
    println(io, "# Similarly for Bx_b, Bs_b, By_a, Bx_a, Bs_a, By_bs, Bx_bs, Bs_bs.")
    println(io, "# Notation: b(n,m) = d^m b_n/ds^m,  a(n,m) = d^m a_n/ds^m,")
    println(io, "#           bs(m)  = d^{m+1} a_0/ds^{m+1}")
    println(io)

    println(io, "Bx_a  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "Bx_b  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "Bx_bs = Dict{Int64, Vector{Tuple{Real, Int64, Int64, Int64}}}()")

    println(io, "By_a  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "By_b  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "By_bs = Dict{Int64, Vector{Tuple{Real, Int64, Int64, Int64}}}()")

    println(io, "Bs_a  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "Bs_b  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "Bs_bs = Dict{Int64, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io)

    # --- b_n(s) functions, n = 1..13, m = 0..MAXTOT ---
    println(io, "# --- b(n,m) contributions ---")
    println(io)
    for (T, prefix) in [(TBy, "By_b"), (TBx, "Bx_b"), (TBs, "Bs_b")]
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
    for (T, prefix) in [(TBy, "By_a"), (TBx, "Bx_a"), (TBs, "Bs_a")]
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
    for (T, prefix) in [(TBy, "By_bs"), (TBx, "Bx_bs"), (TBs, "Bs_bs")]
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
