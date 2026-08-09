using Symbolics

# ---------------------------------------------------------------------------
# Create the two data files in the GeneralizedGradients/tables dir.
#
# Both are the same Taylor expansion of the magnetic field and the vector
# potential -- reproducing and extending Table 1 of Van der Schueren et al.,
# IPAC'24 -- written in two forms, so the expansion is computed once here and
# written out twice.  Neither file has to be recomputed unless higher a_m and
# b_m order or higher derivative order is wanted.
#
# Notation:
#  b(m,nd) = (d/ds)^nd (b_m), a(m,nd) = (d/ds)^nd a_m, and bs(nd) = (d/ds)^nd bs
#
# Output 1: tables/monomial_functions.jl -- one section per monomial x^p y^q,
#   giving the six coefficients as readable symbolic expressions in a(m,nd),
#   b(m,nd), bs(nd) and g_ref.  For reading, and for checking against the
#   paper; the package does not use it.
#
# Output 2: tables/gg_coef_table.jl -- the same content inverted, keyed by GG
#   function instead of by monomial, as numeric tuples.  This one is included
#   by the package and used by gg_calc_fit.  For each field component (Bx, By,
#   Bs) and vector potential component (Ax, Ay, As), and for each function a,
#   b, bs and derivative, it holds a vector of the coefficients that
#   contribute, in 18 Dicts:
#
#     Bx_a[(m,nd)], Bx_b[(m,nd)], Bx_bs[nd],   ... and likewise By, Bs,
#     Ax_a[(m,nd)], Ax_b[(m,nd)], Ax_bs[nd],   ... and likewise Ay, As
#
#   Example: By_b[(m,nd)] = [(coef, p, q, r), ...]
#   means contribution by b(m,nd) is:
#     By += coef * g_ref^r * x^p * y^q * b(m,nd)
# ---------------------------------------------------------------------------

# --- Input parameters (override from the environment) ------------------------
# See the documentation below.

const MAXTOT = parse(Int, get(ENV, "MAXTOT", "13"))
const MMAX   = parse(Int, get(ENV, "MMAX", "14"))

# --- Derived sizes -----------------------------------------------------------
# Internal to this program (they are recorded in the output header, but the
# table defines no constants for them), so they are documented here.

# Truncation order in x: the power series are carried as length-N coefficient
# vectors holding the degrees x^0 .. x^(N-1).  Also the ceiling on MMAX, which
# may not exceed N - 1.
const N = MAXTOT + 2

# Highest s-derivative order that the family-projection and integration
# substitution dictionaries are built for.  It exceeds MAXTOT by a margin
# because the phi recurrence differentiates twice per step, so orders somewhat
# above the tabulated nd range appear in intermediate expressions and must still
# be matched by the substitutions.
const MDER = MAXTOT + 4

# Highest power k of g_ref any tabulated coefficient can carry, and so the
# number of g_ref-derivatives coeff_poly_h takes when it splits a coefficient
# into its (c, p, q, k) terms.  The bound comes from 1/(1 + g_ref x), expanded
# to x^(N-1), being the only source of g_ref powers.
const MAX_H = MAXTOT + 2

# The x-truncation has to reach x^MMAX or the top multipoles are silently
# dropped from phi_0 / phi_1 (the `m <= N - 1` guards below).
MMAX <= N - 1 || error("MMAX = $MMAX exceeds the x truncation N - 1 = $(N - 1). " *
                       "Raise MAXTOT to at least $(MMAX - 1).")

@variables s
Ds = Differential(s)

for m in 0:MMAX
  @eval @variables $(Symbol("a$m"))(s)
  if m >= 1
    @eval @variables $(Symbol("b$m"))(s)
  end
end

@variables g_ref

avars = [eval(Symbol("a$m")) for m in 0:MMAX]   # avars[k+1] = a_k(s)
bvars = [eval(Symbol("b$m")) for m in 1:MMAX]   # bvars[m]   = b_m(s)

# ---------------------------------------------------------------------------
# phi_0, phi_1 seed functions
# ---------------------------------------------------------------------------

phi0 = Vector{Num}(undef, N)
fill!(phi0, Num(0))
phi0[1] = -avars[1]
for m in 1:MMAX
  m <= N - 1 && (phi0[m+1] = -avars[m+1] / factorial(m))
end

phi1 = Vector{Num}(undef, N)
fill!(phi1, Num(0))
for m in 1:MMAX
  m - 1 <= N - 1 && (phi1[m] = -bvars[m] / factorial(m - 1))
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

# Compute the nd-th s-derivative of v by iterative application of Ds,
# matching exactly the form produced by dsarr in the phi recurrence.
function nth_ds_deriv(v, nd)
  result = v
  for _ in 1:nd
    result = expand_derivatives(Ds(result))
  end
  return result
end

# ---------------------------------------------------------------------------
# Vector potential coefficients T_{p,q} of x^p y^q in A_x, A_y, A_s
# (B = curl A in Frenet coordinates).  See papers/vector-potential.
#
# C is split by GG family: alpha (a_m, m>=1), beta (b_m), gamma (b_s = a_0
# derivatives).  Gauge A_y = 0 for alpha/beta, A_s = 0 for gamma:
#
#   A_x = - sum 1/(j+1) (alpha+beta)_{s,i,j} x^i y^{j+1}
#         + (1+hx) sum [int ds gamma_{y,i,j}] x^i y^j
#   A_y = - (1+hx) sum [int ds gamma_{x,i,j}] x^i y^j
#   A_s =   sum 1/(j+1) (alpha+beta)_{x,i,j} x^i y^{j+1}
#         - 1/(1+hx) sum_i beta_{y,i,0} ( x^{i+1}/(i+1) + g_ref x^{i+2}/(i+2) )
# ---------------------------------------------------------------------------

println("computing vector potential coefficients ...")

# Project an expression onto one GG family by zeroing the others (the field
# coefficients are linear in the GG functions); int ds lowers a b_s order.
zero_a0    = Dict{Num,Num}()   # zero a_0 and its s-derivatives
zero_apos  = Dict{Num,Num}()   # zero a_1 .. a_MMAX and derivatives
zero_b     = Dict{Num,Num}()   # zero b_1 .. b_MMAX and derivatives
zero_a_all = Dict{Num,Num}()   # zero a_0 .. a_MMAX and derivatives
for nd in 0:MDER
  zero_a0[nth_ds_deriv(avars[1], nd)] = Num(0)
end
for m in 1:MMAX, nd in 0:MDER
  zero_apos[nth_ds_deriv(avars[m+1], nd)] = Num(0)
  zero_b[nth_ds_deriv(bvars[m], nd)]      = Num(0)
end
for m in 0:MMAX, nd in 0:MDER
  zero_a_all[nth_ds_deriv(avars[m+1], nd)] = Num(0)
end
zero_not_a0 = merge(zero_apos, zero_b)        # keep only a_0  (the b_s family)

intds_a0 = Dict{Num,Num}()                    # int ds: D^k(a_0) -> D^{k-1}(a_0)
for k in 1:MDER
  intds_a0[nth_ds_deriv(avars[1], k)] = nth_ds_deriv(avars[1], k - 1)
end

ab_part(e) = substitute(e, zero_a0)       # a_m (m>=1) + b_m  part of C
b_part(e)  = substitute(e, zero_a_all)    # b_m               part of C
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

# ---------------------------------------------------------------------------
# Output 1: tables/monomial_functions.jl -- the expansion in symbolic form,
# keyed by monomial.  Written first because it needs nothing beyond the
# coefficients computed above.
# ---------------------------------------------------------------------------

# Rewrite a Symbolics expression into the a(m,nd) / b(m,nd) / bs(nd) notation.
function rewrite_notation(expr)
  str = string(expr)

  # a_0(s) derivatives: D^nd(a0(s)) -> bs(nd-1)
  str = replace(str, r"Differential\(s, (\d+)\)\(a0\(s\)\)" => function(mt)
    mm = match(r"Differential\(s, (\d+)\)\(a0\(s\)\)", mt)
    k = parse(Int, mm.captures[1]) - 1
    "bs($k)"
  end)

  # general D^nd(a_m(s)) -> a(m,nd), D^nd(b_m(s)) -> b(m,nd)
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

monofile = joinpath(@__DIR__, "..", "tables", "monomial_functions.jl")
open(monofile, "w") do io
  println(io, "# Extended Table 1: Taylor expansion of the magnetic field and the")
  println(io, "# vector potential (constant g_ref)")
  println(io, "")
  println(io, "# Coefficients of the monomials x^p y^q in B_x, B_y, B_s and in the")
  println(io, "# vector potential A_x, A_y, A_s (B = curl A), for total degree")
  println(io, "# p+q <= $MAXTOT, assuming the curvature g_ref is constant (g_ref' = 0).")
  println(io, "# Notation: a(m,nd) = d^nd a_m/ds^nd, b(m,nd) = d^nd b_m/ds^nd,")
  println(io, "# bs(nd) = d^nd b_s/ds^nd.")
  println(io, "")
  println(io, "# ---------------------------------------------------------------------------")
  println(io, "# GENERATED FILE -- do not edit by hand.")
  println(io, "#")
  println(io, "# Written by programs/create_field_to_gg_coef_tables.jl, which writes gg_coef_table.jl from the")
  println(io, "# same computation, with the input parameters below.  These are the only")
  println(io, "# inputs: the same two values always reproduce this file exactly.  To")
  println(io, "# regenerate:")
  println(io, "#")
  println(io, "#   MAXTOT=$MAXTOT MMAX=$MMAX julia programs/create_field_to_gg_coef_tables.jl")
  println(io, "#")
  println(io, "# Input parameters (documented in tables/gg_coef_table.jl, whose docstrings")
  println(io, "# the package includes: `?MAXTOT` and `?MMAX` in the REPL)")
  println(io, "#   MAXTOT = $MAXTOT   max total degree p+q of the x^p y^q monomials kept")
  println(io, "#   MMAX   = $MMAX   max multipole order m of the a_m and b_m functions")
  println(io, "#")
  println(io, "# Derived sizes")
  println(io, "#   N      = $N   truncation order in x (degrees 0 .. N-1) = MAXTOT + 2")
  println(io, "#   MDER   = $MDER   max s-derivative order carried in the symbolic")
  println(io, "#                projections = MAXTOT + 4")
  println(io, "#")
  println(io, "# Sections below: one per monomial x^p y^q with p + q <= $MAXTOT, each giving")
  println(io, "# the six coefficients as expressions in a(m,nd), b(m,nd), bs(nd) and g_ref.")
  println(io, "# ---------------------------------------------------------------------------")
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

println("done, written to $monofile")

# ---------------------------------------------------------------------------
# Output 2: tables/gg_coef_table.jl -- the same expansion inverted, keyed by GG
# function instead of by monomial, as numeric (coef, p, q, k) tuples.
# ---------------------------------------------------------------------------

Dh = Differential(g_ref)

# Build the zero-substitution dictionary once: every symbolic function and
# every s-derivative (up to MDER) mapped to 0.  g_ref is NOT zeroed here.
println("building zero substitution dict ...")
all_zero = Dict{Num,Num}()

for m in 0:MMAX
  v = avars[m+1]
  all_zero[v] = Num(0)
  for nd in 1:MDER
    all_zero[nth_ds_deriv(v, nd)] = Num(0)
  end
end
for m in 1:MMAX
  v = bvars[m]
  all_zero[v] = Num(0)
  for nd in 1:MDER
    all_zero[nth_ds_deriv(v, nd)] = Num(0)
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

# Docstrings for the two input parameters, emitted into the table on the
# constants that record them.  They describe the table as built, so the values
# are interpolated in; keep them free of the triple-quote delimiter.

const _MAXTOT_DOC = """
    const MAXTOT

Maximum total degree `p + q` of the monomials `x^p y^q` tabulated in
`tables/gg_coef_table.jl`, and the highest derivative order `nd` the table
carries for `a(m,nd)`, `b(m,nd)` and `bs(nd)`. This table was built with
`MAXTOT = $MAXTOT`.

It is a property of the table rather than a fit setting: it is the ceiling on
`GGFitInputParams.nd_max`, which selects from what is tabulated here.

Covering a higher order means rebuilding the table, which is what
`programs/create_field_to_gg_coef_tables.jl` is for -- it writes both of the files in `tables/`
from one computation of the expansion:

    MAXTOT=16 MMAX=17 julia programs/create_field_to_gg_coef_tables.jl

`MAXTOT` and `MMAX` are that program's only inputs. It reads each from the
environment variable of the same name, falling back to the default built into
the program, and writes both back into the table it generates -- as this
docstring and as the header comment.

Raising `MAXTOT` enlarges the table in every direction at once: how far the
`phi` recurrence is carried, the `(MAXTOT+1)(MAXTOT+2)/2` monomials per GG
function, the `nd` range, and the internal sizes derived from it. Generation
time grows steeply as a result, each step costing considerably more than the
last. There is little reason to raise it for its own sake -- a fit is limited by
what the field grid supports long before it is limited by the table.
"""

const _MMAX_DOC = """
    const MMAX

Maximum multipole order `m` of the `a_m` and `b_m` functions in
`tables/gg_coef_table.jl`: the largest `m` appearing in a tabulated `a(m,nd)` or
`b(m,nd)` key. This table was built with `MMAX = $MMAX`.

It is the ceiling on `GGFitInputParams.m_max`: a larger `m_max` is clamped to
`MMAX`, since that is the highest order the table can supply.

`bs(nd)` carries no multipole order -- it is the derivative tower of `a_0` -- so
it is unaffected by this limit.

Changing it means rebuilding the table; see `MAXTOT` for the command and for
what the rebuild costs. `MMAX` may not exceed `MAXTOT + 1`: the seed series
`phi_0` and `phi_1` are truncated at `x^(MAXTOT+1)`, so a higher order would be
dropped as the table was built, leaving it quietly incomplete rather than
visibly wrong. `create_field_to_gg_coef_tables.jl` rejects that combination instead of writing
such a table, so raising `MMAX` usually means raising `MAXTOT` with it.
"""

"""
    write_docstring(io, doc::AbstractString, definition::AbstractString)

Write `definition` to `io` with `doc` attached to it as a Julia docstring.
"""
function write_docstring(io, doc::AbstractString, definition::AbstractString)
  println(io, "\"\"\"")
  print(io, doc)
  println(io, "\"\"\"")
  println(io, definition)
  println(io)
end

coeffile = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
open(coeffile, "w") do io
  println(io, "# Inverse field and vector-potential coefficient table (full g_ref dependence)")
  println(io, "#")
  println(io, "# By_b[(m,nd)] = [(c, p, q, k), ...]  means  By += c * g_ref^k * x^p * y^q * b(m,nd)")
  println(io, "# Similarly for Bx_b, Bs_b, By_a, Bx_a, Bs_a, By_bs, Bx_bs, Bs_bs and for")
  println(io, "# the vector potential A (B = curl A):  Ax_a, Ax_b, Ax_bs, Ay_a, Ay_b,")
  println(io, "# Ay_bs, As_a, As_b, As_bs (same meaning, e.g. Ax_b[(m,nd)] -> Ax += ...).")
  println(io, "# Notation: b(m,nd) = d^nd b_m/ds^nd,  a(m,nd) = d^nd a_m/ds^nd,")
  println(io, "#           bs(nd)  = d^{nd+1} a_0/ds^{nd+1}")
  println(io, "#")
  println(io, "# ---------------------------------------------------------------------------")
  println(io, "# GENERATED FILE -- do not edit by hand.")
  println(io, "#")
  println(io, "# Written by programs/create_field_to_gg_coef_tables.jl, which writes monomial_functions.jl")
  println(io, "# from the same computation, with the input parameters below.  These are the")
  println(io, "# only inputs: the same two values always reproduce this file exactly.  To")
  println(io, "# regenerate:")
  println(io, "#")
  println(io, "#   MAXTOT=$MAXTOT MMAX=$MMAX julia programs/create_field_to_gg_coef_tables.jl")
  println(io, "#")
  println(io, "# Input parameters (documented in the docstrings just below)")
  println(io, "#   MAXTOT = $MAXTOT   max total degree p+q of the x^p y^q monomials kept")
  println(io, "#   MMAX   = $MMAX   max multipole order m of the a_m and b_m functions")
  println(io, "#")
  println(io, "# Derived sizes")
  println(io, "#   N      = $N   truncation order in x (degrees 0 .. N-1) = MAXTOT + 2")
  println(io, "#   MDER   = $MDER   max s-derivative order carried in the symbolic")
  println(io, "#                projections = MAXTOT + 4")
  println(io, "#   MAX_H  = $MAX_H   max power k of g_ref a coefficient can carry = MAXTOT + 2")
  println(io, "#")
  println(io, "# Ranges actually tabulated")
  println(io, "#   a(m,nd), b(m,nd):  m = 1 .. $MMAX,  nd = 0 .. $MAXTOT")
  println(io, "#   bs(nd):            nd = 0 .. $MAXTOT")
  println(io, "#   monomials x^p y^q: p + q <= $MAXTOT")
  println(io, "#")
  println(io, "# An (m,nd) or nd key is absent when every one of its coefficients is zero.")
  println(io, "# ---------------------------------------------------------------------------")
  println(io)

  # The two input parameters, as documented constants: this file is included by
  # the package, so these are what `?MAXTOT` and `?MMAX` find.
  write_docstring(io, _MAXTOT_DOC, "const MAXTOT = $MAXTOT")
  write_docstring(io, _MMAX_DOC,   "const MMAX   = $MMAX")

  for comp in ("Bx", "By", "Bs", "Ax", "Ay", "As")
    println(io, "$(comp)_a  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "$(comp)_b  = Dict{Tuple{Int64, Int64}, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
    println(io, "$(comp)_bs = Dict{Int64, Vector{Tuple{Real, Int64, Int64, Int64}}}()")
  end
  println(io)

  # --- b_m(s) functions, m = 1..MMAX, nd = 0..MAXTOT ---
  println(io, "# --- b(m,nd) contributions ---")
  println(io)
  for (T, prefix) in [(TBy, "By_b"), (TBx, "Bx_b"), (TBs, "Bs_b"),
            (TAy, "Ay_b"), (TAx, "Ax_b"), (TAs, "As_b")]
    print("Processing $(prefix) ...")
    for m in 1:MMAX, nd in 0:MAXTOT
      sym   = nth_ds_deriv(bvars[m], nd)
      terms = Tuple{Rational{Int},Int,Int,Int}[]
      for q in 0:MAXTOT, p in 0:(MAXTOT-q)
        hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
        for k in sort(collect(keys(hc)))
          push!(terms, (hc[k], p, q, k))
        end
      end
      isempty(terms) || println(io, "$(prefix)[($m,$nd)] = $(fmt_terms(terms))")
    end
    println(io)
    println("done")
  end

  # --- a_m(s) functions, m = 1..MMAX, nd = 0..MAXTOT ---
  println(io, "# --- a(m,nd) contributions ---")
  println(io)
  for (T, prefix) in [(TBy, "By_a"), (TBx, "Bx_a"), (TBs, "Bs_a"),
            (TAy, "Ay_a"), (TAx, "Ax_a"), (TAs, "As_a")]
    print("Processing $(prefix) ...")
    for m in 1:MMAX, nd in 0:MAXTOT
      sym   = nth_ds_deriv(avars[m+1], nd)   # avars[m+1] = a_m(s)
      terms = Tuple{Rational{Int},Int,Int,Int}[]
      for q in 0:MAXTOT, p in 0:(MAXTOT-q)
        hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
        for k in sort(collect(keys(hc)))
          push!(terms, (hc[k], p, q, k))
        end
      end
      isempty(terms) || println(io, "$(prefix)[($m,$nd)] = $(fmt_terms(terms))")
    end
    println(io)
    println("done")
  end

  # --- bs(nd) = d^{nd+1} a_0/ds^{nd+1}, nd = 0..MAXTOT ---
  println(io, "# --- bs(nd) contributions ---")
  println(io)
  for (T, prefix) in [(TBy, "By_bs"), (TBx, "Bx_bs"), (TBs, "Bs_bs"),
            (TAy, "Ay_bs"), (TAx, "Ax_bs"), (TAs, "As_bs")]
    print("Processing $(prefix) ...")
    for nd in 0:MAXTOT
      sym   = nth_ds_deriv(avars[1], nd + 1)  # (nd+1)-th deriv of a_0 = bs(nd)
      terms = Tuple{Rational{Int},Int,Int,Int}[]
      for q in 0:MAXTOT, p in 0:(MAXTOT-q)
        hc = coeff_poly_h(T[(p,q)], sym, MAX_H)
        for k in sort(collect(keys(hc)))
          push!(terms, (hc[k], p, q, k))
        end
      end
      isempty(terms) || println(io, "$(prefix)[$nd] = $(fmt_terms(terms))")
    end
    println(io)
    println("done")
  end
end

println("done, written to $coeffile")
