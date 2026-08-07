# ---------------------------------------------------------------------------
# gg_fit.jl
#
# Fit a 3D magnetic field grid to generalized-gradient (GG) coefficients.
# ---------------------------------------------------------------------------

# The field expansion (`tables/gg_coef_table.jl`) is linear in the GG functions
# and their `s`-derivatives:
# 
# ```
# B_c(x,y,z) = Σ_{(m,nd)} CS_c,b(m,nd; x,y) · b(m,nd)(z)
#            + Σ_{(m,nd)} CS_c,a(m,nd; x,y) · a(m,nd)(z)
#            + Σ_{nd}     CS_c,bs(nd; x,y)  · bs(nd)(z)
# ```
#
# for each field component `c ∈ {x, y, z}`, where
#
# ```
# CS_c,f(m,nd; x,y) = Σ (coeff · g_ref^k · x^p · y^q)
# ```
#
# is the sum of the table entries `c_f[(m,nd)] = [(coeff,p,q,k), ...]`, and
# `b(m,nd) = dⁿᵈb_m/dzⁿᵈ`, `a(m,nd) = dⁿᵈa_m/dzⁿᵈ`, `bs(nd) = dⁿᵈ⁺¹a_0/dzⁿᵈ⁺¹`.
#
# The unknowns at a base plane `z0` are the function values and their derivatives
# `f(m,nd)(z0)`, `nd = 0 … nd_max`. The field on a neighbouring plane at offset
# `dz = z - z0` is obtained by Taylor-extrapolating each derivative:
#
# ```
#  f(m,nd)(z0+dz) = Σ_{j≥nd} dz^(j-nd)/(j-nd)! · f(m,j)(z0)
# ```
# where `f` is either `a`, `b`, or `bs`.
# Substituting makes the model linear in the base-plane unknowns `f(m,j)(z0)`:
# ```
# design entry for unknown f(m,j) = Σ_{nd=0}^{j} CS_c,f(m,nd; x,y) · dz^(j-nd)/(j-nd)!
# ```



"""
    gg_fit(field::FieldGridTable, params::GGFitInputParams) -> GGCoefs

Fit a 3D magnetic field grid to generalized-gradient (GG) coefficients
`a_m(z)`, `b_m(z)`, `b_s(z)` and their `z`-derivatives, plane by plane.
A "plane" here  is always a plane at constant z. 

The returned `GGCoefs` holds the fitted coefficients and per-plane diagnostics.
Use `gg_fit_show_results` to print a summary and
`write_gg_fit` to save the result to an HDF5 file (readable by `read_gg_fit`).

See `examples/run_gg_fit.jl` for a complete, runnable example.

## Arguments

- `field` — a `FieldGridTable`.
  `field.magnetic[ix,iy,iz]` is the `[Bx,By,Bz]` 3-vector at the grid point,
  whose `(x, y, z)` position is `field.r0 + field.dr .* (ix, iy, iz)`.
- `params` — a `GGFitInputParams` holding the fit parameters (`origin`,
  `n_planes_add`, `core_weight`, `outer_plane_weight`, `output_file`).

## How the fit works

The GG coefficients are computed at the equally spaced `z`-positions coincident with the
input field-table planes. These planes are sometimes called "principal planes".
The fit is done plane by plane: the coefficients at a given principal plane (the "base plane") 
are computed independently of the calculation of the coefficients at every other plane, and
all coefficients of a base plane are solved for simultaneously by minimizing a merit function

```
Merit = Σₖ  weightₖ · (field_from_tableₖ - field_from_GG_coefsₖ)^2
```

The sum runs over all field points lying in a plane within `n_planes_add` of the
base plane. For example, `n_planes_add = 2` adds two planes on either side, so
five planes are used in total. Near the ends of the table the count is reduced —
a base plane at the very end of the table uses only three planes when
`n_planes_add = 2`.

The merit function is a linear equation in the GG coefficients so
each base plane is solved by weighted linear least squares fit over all field
points lying within `n_planes_add` planes of the base plane.

Adding extra planes smooths the computed values between planes at the cost of making the fit
at the principal planes worse. So adding more planes can give a worse fit.

## Scanning over the model size

Which unknowns exist at all is set by two cutoffs: `m_max` on the multipole order
and `nd_max` on the derivative order. Giving either as a vector or range makes
`gg_fit` fit the full grid of combinations and keep the best one. For example

```julia
p.m_max  = 1:10       # try m_max = 1, 2, … 10
p.nd_max = 0:6        # crossed with nd_max = 0, 1, … 6
```

runs 77 fits. A single `m_max`/`nd_max` applies to every base plane — the model
never varies plane to plane — so the scan is over whole-fit models, not per-plane
ones. Each combination is recorded as a `GGFitScanPoint` in the `scan` field of
the result (and printed by `gg_fit_show_results`) with its coefficient count and
pooled RMS residual.

Cost scales with the number of combinations: the design matrix is built once per
base plane at the largest requested model and each combination is then a least
squares solve on a subset of its columns.

## Choosing between models (`fit_criterion`)

Enlarging the model can only lower the residual: the unknowns of a smaller model
are a subset of those of a larger one, so the larger fit can always reproduce the
smaller one and least squares will never do worse. A scan therefore cannot pick
its winner by residual alone — it needs a rule that charges for coefficients.
`fit_criterion` selects that rule. Each scanned model gets a `score` (recorded in
its `GGFitScanPoint`, printed in the scan table) and the **lowest score wins**.

All three rules are built from two numbers:

```
RSS = Σ_planes Σ_points  weight · (field_from_table - field_from_GG_coefs)^2
N   = total number of fitted field-component values, summed over base planes
      = Σ_planes 3 · (points in that plane's fit region)
k   = total number of fitted coefficients
      = (coefficients per plane) · (number of base planes)
```

`RSS` is the weighted residual — the merit function the fit actually minimized,
pooled over every base plane. `N` counts `Bx`, `By` and `Bs` at each field point
of each plane's fit region, so a point used by several base planes is counted
once per base plane. `k` likewise counts the whole fit, since each base plane is
solved for its own copy of the coefficients.

    fit_criterion = :rms

```
score = sqrt(RSS / N)
```

The pooled weighted RMS residual, which is also the `rms` column of the scan
table. This is a pure goodness-of-fit measure with no charge for coefficients, so
by the argument above it will select the largest model offered. Use it to see the
raw residual ranking, not to choose a model.

    fit_criterion = :aic          # Akaike information criterion

```
score = N · ln(RSS / N) + 2·k
```

    fit_criterion = :bic          # Bayesian information criterion, the default

```
score = N · ln(RSS / N) + k·ln(N)
```

Both are the standard information criteria for a least-squares fit with unknown
error variance. The first term, `N·ln(RSS/N)`, is (up to an additive constant
that is the same for every candidate, and so cannot affect the ranking) minus
twice the maximized Gaussian log-likelihood; it rewards a smaller residual. The
second term is the penalty for model size. They differ only in what one
coefficient costs: `2` for AIC, `ln(N)` for BIC. Since `ln(N) > 2` for any
`N > 7`, **BIC always penalizes more heavily than AIC and so selects equal or
smaller models**. The scores are large negative numbers whose absolute value is
meaningless — only differences between candidates matter.

To read the trade-off quantitatively: adding `Δk` coefficients improves the BIC
score only if it lowers `RSS` by at least the fraction

```
1 - exp(-Δk·ln(N) / N)  ≈  Δk·ln(N) / N     (for Δk·ln(N) « N)
```

with `2` in place of `ln(N)` for AIC. This makes the practical weakness of these
criteria explicit here: a field grid easily gives `N` of order 10^5, so each
extra coefficient need only cut the residual sum of squares by of order
`ln(N)/N ~ 10^-4` to pay for itself under BIC, and `2/N` under AIC. AIC in
particular is often too weak to reject anything at that `N`, which is why `:bic`
is the default.

There is a further caveat specific to this problem. AIC and BIC assume the
residuals are independent draws from a common Gaussian. Here the residual is
dominated by systematic truncation error — the field the model cannot represent
— which is smooth and strongly correlated from point to point, not noise. The
effective number of independent measurements is therefore far below `N`, and the
penalty far weaker than the theory intends. Treat the criteria as a defensible
ranking heuristic rather than as a rigorous probability statement, and use the
scan table's `# coefs` and RMS columns to make the final call: the honest signal
is usually the point of diminishing returns, where the residual stops dropping
appreciably as coefficients are added.

## Weighting

The weight of a field point at `(x, y)` and plane offset `dz` (relative to the
base plane) is the product of a transverse and a longitudinal factor:

```
weight(x,y,dz) = w_core(x,y) · w_plane(dz)
```

The transverse factor is

```
w_core(x,y) = core_weight · rmax^2 / (rmax^2 + r^2 · (core_weight - 1))
```

where `r^2 = x^2 + y^2` and `rmax` is the maximum `r` over all points.
`core_weight = 1` (the default) makes `w_core` constant; `core_weight > 1`
favors the core (low-`r`) points at the expense of points farther out. A better
core fit is usually desired since beam particles spend most of their time near
the core.

The longitudinal factor is

```
w_plane(dz) = 1 + (outer_plane_weight - 1) · |dz| / dz_max
```

where `dz_max` is the largest `|dz|` at the ends of the fit region. If
`n_planes_add = 0` (so `dz_max = 0` and the expression is singular) `w_plane` is
set to 1. `outer_plane_weight = 1` (the default) makes `w_plane` constant; a
value between 0 and 1 weights planes nearer the base plane more than the outer
planes.

## Fit input parameters (`GGFitInputParams`)

- `origin = [x0, y0]` — `(x, y)` line about which the GG coefficients are
  computed. If `field.g_ref` is non-zero, `origin` must be `[0, 0]`.
  Default `[0.0, 0.0]`.
- `n_planes_add` — number of `z`-planes added to either side of the base plane
  used to resolve derivatives.
- `m_max` — highest multipole order `m` kept. `-1` (the default) keeps every `m`
  in the coefficient table. A vector or range scans (see "Scanning" below).
- `nd_max` — highest derivative order `nd` kept. `-1` (the default) means
  `2*n_planes_add`. A vector or range scans.
- `fit_criterion` — how a scan picks its winner: `:bic` (default), `:aic`, or
  `:rms`.
- `core_weight` — merit-function weight for "core" (near-axis) points.
  Default `1` (uniform).
- `outer_plane_weight` — merit-function weight for the outer `z`-planes.
  Default `1`.
- `output_file` — name of the output HDF5 file written by
  `write_gg_fit`. Default `"gg_fit_results.h5"`.

## Side note

In theory, the fitting does not require that the field table be a rectangular
grid of equally spaced points. In fact, a set of randomly spaced field points would work.
Also there is no fundamental requirement that the fit planes be evenly spaced. It is only
for convenience that the `gg_fit` function require a regularly spaced field table and that
the output is at evenly spaced planes.
"""
function gg_fit(field::FieldGridTable, params::GGFitInputParams)
  a_dicts  = (Bx_a, By_a, Bs_a)
  b_dicts  = (Bx_b, By_b, Bs_b)
  bs_dicts = (Bx_bs, By_bs, Bs_bs)

  origin             = params.origin
  n_planes_add       = params.n_planes_add
  core_weight        = params.core_weight
  outer_plane_weight = params.outer_plane_weight

  mag   = field.magnetic            # OffsetArray: mag[ix, iy, iz] == [Bx, By, Bz]
  r0    = field.r0                  # grid origin (gridOriginOffset)
  dr    = field.dr
  g_ref = field.g_ref

  npa     = n_planes_add
  dz_grid = dr[3]

  # ---- Resolve the (m_max, nd_max) models to try -------------------------
  # A scalar pins the model; a vector asks for a scan. Candidates are clamped to
  # what the coefficient table actually holds so that, say, `nd_max = 0:100` does
  # not fit the same top model over and over.
  m_table_max  = maximum(k[1] for d in (a_dicts..., b_dicts...) for k in keys(d))
  nd_table_max = maximum(k[2] for d in (a_dicts..., b_dicts...) for k in keys(d))
  m_cands  = _fit_candidates(params.m_max, m_table_max, m_table_max)
  nd_cands = _fit_candidates(params.nd_max, 2 * npa, nd_table_max)
  scanning = !(params.m_max isa Int && params.nd_max isa Int)

  params.fit_criterion in (:bic, :aic, :rms) ||
      error("Unknown fit_criterion: $(params.fit_criterion). Expecting :bic, :aic, or :rms.")

  # The master model is the union of every candidate: the design matrix is built
  # once at this size and each candidate then uses a subset of its columns.
  m_max_all  = maximum(m_cands)
  nd_max_all = maximum(nd_cands)

  # Grid index ranges, taken from the field arrays (not assumed to start at 0/1).
  # Use plain UnitRanges (not the OffsetArray axes) so the `xs`/`ys`/`z_base`
  # comprehensions below stay 1-based while `mag` is still indexed by real index.
  ixs = first(axes(mag, 1)):last(axes(mag, 1))
  iys = first(axes(mag, 2)):last(axes(mag, 2))
  izs_grid = first(axes(mag, 3)):last(axes(mag, 3))

  # Transverse coordinates relative to the GG origin (the expansion axis).
  xs = [r0[1] + dr[1] * ix - origin[1] for ix in ixs]
  ys = [r0[2] + dr[2] * iy - origin[2] for iy in iys]
  rmax2 = maximum(xs[i]^2 + ys[j]^2 for i in eachindex(xs), j in eachindex(ys))

  # ---- Assemble the master parameter (unknown) list from the table keys --
  # a/b indexed by (m,nd); bs indexed by nd (stored as (0,nd)).
  pset = Set{Tuple{Symbol,Int,Int}}()
  for d in a_dicts, k in keys(d)
    k[1] <= m_max_all && k[2] <= nd_max_all && push!(pset, (:a, k[1], k[2]))
  end
  for d in b_dicts, k in keys(d)
    k[1] <= m_max_all && k[2] <= nd_max_all && push!(pset, (:b, k[1], k[2]))
  end
  for d in bs_dicts, nd in keys(d)
    nd <= nd_max_all && push!(pset, (:bs, 0, nd))
  end
  master_list = sort!(collect(pset))
  ncols_all   = length(master_list)

  # Column subset of the master list belonging to each candidate model. `bs`
  # unknowns describe a_0 and carry no multipole order, so `m_max` never drops them.
  cand_models = [(mx, ndx) for mx in m_cands for ndx in nd_cands]
  cand_cols   = [[c for (c, (typ, m, nd)) in enumerate(master_list)
                    if nd <= ndx && (typ == :bs || m <= mx)] for (mx, ndx) in cand_models]

  # ---- Precompute CB (field-coefficient) grids: (comp,type,m,nd) => matrix over (ix,iy) --
  # comp: 1=Bx, 2=By, 3=Bs.   type: :a,:b,:bs.
  comp_dicts = Dict(
    (1, :a) => a_dicts[1], (2, :a) => a_dicts[2], (3, :a) => a_dicts[3],
    (1, :b) => b_dicts[1], (2, :b) => b_dicts[2], (3, :b) => b_dicts[3],
    (1, :bs) => bs_dicts[1], (2, :bs) => bs_dicts[2], (3, :bs) => bs_dicts[3],
  )
  CB = Dict{Tuple{Int,Symbol,Int,Int},Matrix{Float64}}()
  for ((comp, typ), d) in comp_dicts
    for (key, terms) in d
      m, nd = typ == :bs ? (0, key) : (key[1], key[2])
      (nd <= nd_max_all && m <= m_max_all) || continue
      grid = [_coefsum(terms, xs[i], ys[j], g_ref) for i in eachindex(xs), j in eachindex(ys)]
      CB[(comp, typ, m, nd)] = grid
    end
  end

  # Flatten CB into a per-(component, column) list of (derivative order, grid) so
  # the design-matrix inner loop below does no dictionary lookups. Only the
  # dz-dependent factor is left to vary from plane to plane.
  col_terms = [[Tuple{Int,Matrix{Float64}}[(ndd, CB[(comp, typ, m, ndd)])
                                           for ndd in 0:j if haskey(CB, (comp, typ, m, ndd))]
                for (typ, m, j) in master_list] for comp in 1:3]

  # ---- Result containers ------------------------------------------------
  nplanes = length(izs_grid)
  z_base  = [r0[3] + dr[3] * iz for iz in izs_grid]
  ncand   = length(cand_models)
  # Per candidate: fitted coefficients (rows follow that candidate's column
  # subset) and the two per-plane residuals. Only the winner is expanded into
  # the a/b/bs dictionaries at the end.
  thetas   = [zeros(length(cols), nplanes) for cols in cand_cols]
  rms_c    = [fill(NaN, nplanes) for _ in 1:ncand]
  rmsu_c   = [fill(NaN, nplanes) for _ in 1:ncand]
  nrow_pl  = zeros(Int, nplanes)
  field_ave_plane = fill(NaN, nplanes)

  # ---- Loop over base planes -------------------------------------------
  for (pidx, iz0) in enumerate(izs_grid)
    izs   = max(first(izs_grid), iz0 - npa):min(last(izs_grid), iz0 + npa)
    dzs   = [(iz - iz0) * dz_grid for iz in izs]
    dzmax = maximum(abs, dzs)

    npts  = length(ixs) * length(iys) * length(izs)
    nrows = 3 * npts
    nrow_pl[pidx] = nrows
    A     = zeros(nrows, ncols_all)
    bvec  = zeros(nrows)
    sw    = zeros(nrows)          # sqrt of point weight

    row = 0
    for (zi, iz) in enumerate(izs)
      dz  = dzs[zi]
      wpl = (npa == 0 || dzmax == 0) ? 1.0 :
            1 + (outer_plane_weight - 1) * abs(dz) / dzmax
      # Taylor-extrapolation factors dz^k/k! for this plane offset, k = 0 … nd_max_all.
      dzfac = [dz^k / factorial(k) for k in 0:nd_max_all]
      for (iix, ix) in enumerate(ixs), (iiy, iy) in enumerate(iys)
        r2  = xs[iix]^2 + ys[iiy]^2
        wco = core_weight == 1 ? 1.0 :
              core_weight * rmax2 / (rmax2 + r2 * (core_weight - 1))
        w   = wco * wpl
        B3  = mag[ix, iy, iz]                     # [Bx, By, Bs]
        for comp in 1:3
          row += 1
          bvec[row] = B3[comp]
          sw[row]   = sqrt(w)
          terms_c   = col_terms[comp]
          for col in 1:ncols_all
            j   = master_list[col][3]
            val = 0.0
            for (ndd, grid) in terms_c[col]
              val += grid[iix, iiy] * dzfac[j-ndd+1]
            end
            A[row, col] = val
          end
        end
      end
    end

    # Weighted least squares (pinv = stable min-norm solution if rank-deficient).
    Aw = A .* sw
    bw = bvec .* sw
    for c in 1:ncand
      cols  = cand_cols[c]
      Awc   = length(cols) == ncols_all ? Aw : Aw[:, cols]
      theta = pinv(Awc) * bw
      thetas[c][:, pidx] = theta
      # Residual of the same fit, scored with and without the point weights. The
      # unweighted value says how well the field itself is reproduced; the weighted
      # one is the quantity the fit actually minimized.
      rms_c[c][pidx]  = norm(Awc * theta - bw) / sqrt(nrows)
      rmsu_c[c][pidx] = norm(view(A, :, cols) * theta - bvec) / sqrt(nrows)
    end

    # Average |B| over the base plane alone (unweighted): the field scale against
    # which the residuals above are to be judged.
    field_ave_plane[pidx] = sum(norm(mag[ix, iy, iz0]) for ix in ixs, iy in iys) /
                            (length(ixs) * length(iys))
  end

  # ---- Score the candidates and pick the winner -------------------------
  ndata = sum(nrow_pl)
  scan  = GGFitScanPoint[]
  for c in 1:ncand
    mx, ndx = cand_models[c]
    ncoef   = length(cand_cols[c])
    # Pool the per-plane residuals back into one sum of squares over all planes.
    rss  = sum(rms_c[c][p]^2 * nrow_pl[p] for p in 1:nplanes)
    rssu = sum(rmsu_c[c][p]^2 * nrow_pl[p] for p in 1:nplanes)
    push!(scan, GGFitScanPoint(; m_max = mx, nd_max = ndx, n_coef = ncoef,
                                 rms = sqrt(rss / ndata), rms_unweighted = sqrt(rssu / ndata),
                                 score = _fit_score(params.fit_criterion, rss, ndata,
                                                    ncoef * nplanes),
                                 criterion = params.fit_criterion))
  end
  best = argmin([s.score for s in scan])
  m_max, nd_max = cand_models[best]

  # ---- Expand the winning fit into the coefficient dictionaries ---------
  params_list = master_list[cand_cols[best]]
  a  = Dict{Tuple{Int,Int},Vector{Float64}}()
  b  = Dict{Tuple{Int,Int},Vector{Float64}}()
  bs = Dict{Int,Vector{Float64}}()
  for (col, (typ, m, nd)) in enumerate(params_list)
    typ == :a  && (a[(m, nd)] = thetas[best][col, :])
    typ == :b  && (b[(m, nd)] = thetas[best][col, :])
    typ == :bs && (bs[nd]     = thetas[best][col, :])
  end

  return GGCoefs(; z_base, params = params_list, a, b, bs,
                    rms_plane = rms_c[best], rms_unweighted_plane = rmsu_c[best],
                    field_ave_plane, m_max, nd_max, scan = scanning ? scan : GGFitScanPoint[],
                    g_ref = field.g_ref, origin, dz_grid)
end

#---------------------------------------------------------------------------------------------------

"""
    _fit_candidates(spec, default, table_max) -> Vector{Int}

Resolve an `m_max`/`nd_max` input-parameter setting into the list of values
`gg_fit` should try. `-1` means "use `default`"; any other `Int` pins that value;
a vector or range is taken as-is. Values are clamped to `table_max` (the largest
the coefficient table supports) and deduplicated, so an over-wide request such as
`0:100` does not refit the same top model repeatedly.
"""
function _fit_candidates(spec::Union{Int,AbstractVector{Int}}, default::Int, table_max::Int)
  vals = spec isa Int ? [spec == -1 ? default : spec] : collect(spec)
  isempty(vals) && error("Empty m_max/nd_max candidate list.")
  any(v -> v < 0, vals) && error("Negative m_max/nd_max candidate in $spec.")
  return sort!(unique!([min(v, table_max) for v in vals]))
end

"""
    _fit_score(criterion, rss, ndata, nparam) -> Float64

Selection score for one scanned model; the lowest score wins. `rss` is the pooled
weighted sum of squared residuals over `ndata` field-component values, and
`nparam` the total number of fitted coefficients (per-plane count times the
number of base planes). With `RSS = rss`, `N = ndata` and `k = nparam`:

```
:rms    sqrt(RSS / N)                  goodness of fit only, no charge for k
:aic    N*log(RSS/N) + 2k              Akaike information criterion
:bic    N*log(RSS/N) + k*log(N)        Bayesian information criterion
```

`:aic` and `:bic` share the leading term — minus twice the maximized Gaussian
log-likelihood, dropping an additive constant that is common to every candidate
and so cannot change the ranking — and differ only in the cost of one
coefficient, `2` versus `log(N)`. See the `gg_fit` docstring for how to read the
trade-off and for the caveats that apply when the residual is dominated by
systematic truncation error rather than by noise.
"""
function _fit_score(criterion::Symbol, rss::Float64, ndata::Int, nparam::Int)
  criterion == :rms && return sqrt(rss / ndata)
  # Guard the log against an (effectively) exact fit, which a model with as many
  # coefficients as data points can produce.
  ll = ndata * log(max(rss, floatmin(Float64)) / ndata)
  criterion == :bic && return ll + nparam * log(ndata)
  return ll + 2 * nparam                                  # :aic
end

#---------------------------------------------------------------------------------------------------

"""
    gg_fit_show_results(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)

Print a human-readable summary of a `gg_fit` `results`: the fit settings, the
`(m_max, nd_max)` scan table if a scan was run, a per-plane table of the weighted
and unweighted RMS residuals alongside the average field magnitude of the plane,
and the leading multipoles at the central plane as a quick sanity check.
"""
function gg_fit_show_results(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)
  println("="^72)
  println("GG fit:")
  println("  field grid        : ", join(size(field.magnetic), " x "), "  (ix, iy, iz)")
  println("  g_ref             : ", field.g_ref)
  println("  origin (x,y)      : ", params.origin)
  println("  n_planes_add      : ", params.n_planes_add)
  println("  m_max, nd_max     : ", results.m_max, ", ", results.nd_max,
          isempty(results.scan) ? "" : "   (selected by the scan below)")
  println("  core_weight       : ", params.core_weight)
  println("  outer_plane_weight: ", params.outer_plane_weight)
  println("  # GG coefficients : ", length(results.params), " per plane")
  println("  # base planes     : ", length(results.z_base))

  # ---- (m_max, nd_max) scan table, when a scan was run -------------------
  if !isempty(results.scan)
    crit = results.scan[1].criterion
    println("-"^72)
    println("Model scan (", length(results.scan), " combinations, best by ", crit, " marked *):")
    @printf("%-2s %-6s  %-6s  %-8s  %-12s  %-12s  %-12s\n",
            "", "m_max", "nd_max", "# coefs", "wRMS resid", "RMS resid", string(crit))
    bestscore = minimum(s.score for s in results.scan)
    for s in results.scan
      @printf("%-2s %-6d  %-6d  %-8d  %-12.4e  %-12.4e  %-12.6g\n",
              s.score == bestscore ? "*" : "", s.m_max, s.nd_max, s.n_coef,
              s.rms, s.rms_unweighted, s.score)
    end
  end

  println("-"^72)
  @printf("%-6s  %-12s  %-12s  %-12s  %-12s\n",
          "plane", "z [m]", "wRMS resid", "RMS resid", "<|B|> [T]")
  # A `GGCoefs` built by hand may leave these diagnostics unset; blank the column.
  cell(v, i) = length(v) == length(results.z_base) ? @sprintf("%-12.4e", v[i]) :
                                                     @sprintf("%-12s", "-")
  for i in eachindex(results.z_base)
    @printf("%-6d  %-12.6g  %-12.4e  %s  %s\n", i, results.z_base[i], results.rms_plane[i],
            cell(results.rms_unweighted_plane, i), cell(results.field_ave_plane, i))
  end
  println("-"^72)

  # Show the leading multipoles at the central plane for a quick sanity check.
  ic = cld(length(results.z_base), 2)
  println("Leading coefficients at central plane (z = ",
      @sprintf("%.6g", results.z_base[ic]), "):")
  for m in 1:6
    b00 = get(results.b, (m, 0), nothing)
    a00 = get(results.a, (m, 0), nothing)
    bstr = b00 === nothing ? "     -      " : @sprintf("% .6e", b00[ic])
    astr = a00 === nothing ? "     -      " : @sprintf("% .6e", a00[ic])
    @printf("  m=%-2d   b(m,0)=%s   a(m,0)=%s\n", m, bstr, astr)
  end
  if haskey(results.bs, 0)
    @printf("  bs(0) = % .6e\n", results.bs[0][ic])
  end
  println("="^72)
end

#---------------------------------------------------------------------------------------------------

"""
    write_gg_fit(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams) -> output_file_path

Write a `gg_fit` `results` to an HDF5 file (readable by `read_gg_fit`).

Stores the fitted GG coefficients plus enough metadata to reproduce and
interpret the fit later. The (large) input field table is NOT stored. The file
is written to `params.output_file` and its path is returned.

## HDF5 schema

    root datasets   : z_base, rms_plane, rms_unweighted_plane,
                      field_ave_plane, origin                     (Float64[])
    root attributes : g_ref, dz_grid (Float64); m_max, nd_max, n_planes_add (Int);
                      core_weight, outer_plane_weight (Float64)
    groups a, b     : m (Int[]), nd (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{(m,nd) => values[i,:]}
    group  bs       : nd (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{nd => values[i,:]}
"""
function write_gg_fit(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)
  outfile = params.output_file
  h5open(outfile, "w") do f
    f["z_base"]    = collect(Float64, results.z_base)
    f["rms_plane"] = collect(Float64, results.rms_plane)
    f["rms_unweighted_plane"] = collect(Float64, results.rms_unweighted_plane)
    f["field_ave_plane"]      = collect(Float64, results.field_ave_plane)
    f["origin"]    = collect(Float64, params.origin)
    attributes(f)["m_max"]              = Int(results.m_max)
    attributes(f)["nd_max"]             = Int(results.nd_max)
    attributes(f)["g_ref"]              = Float64(results.g_ref)
    attributes(f)["dz_grid"]            = Float64(field.dr[3])
    # Fit-control parameters, retained for later reference / reproducibility.
    attributes(f)["n_planes_add"]       = Int(params.n_planes_add)
    attributes(f)["core_weight"]        = Float64(params.core_weight)
    attributes(f)["outer_plane_weight"] = Float64(params.outer_plane_weight)
    _write_coef_group(f, "a", results.a)
    _write_coef_group(f, "b", results.b)
    _write_coef_group(f, "bs", results.bs; single = true)
  end
  println("Results written to ", outfile)
  return outfile
end
