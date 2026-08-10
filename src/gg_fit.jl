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
# Substituting makes the fit linear in the base-plane unknowns `f(m,j)(z0)`:
# ```
# design entry for unknown f(m,j) = Σ_{nd=0}^{j} CS_c,f(m,nd; x,y) · dz^(j-nd)/(j-nd)!
# ```


"""
    gg_calc_fit(field::FieldGridTable, params::GGFitInputParams,
                fit_at::Union{Nothing,Tuple{Int,Int}} = nothing) -> GGFit

Fit a 3D DC magnetic field grid to generalized-gradient (GG) coefficients
`a_m(z)`, `b_m(z)`, `b_s(z)` and their `z`-derivatives, plane by plane.
A "plane" here is always a plane at constant z. 

The returned `GGFit` holds the fitted coefficients and per-plane diagnostics.
Use `gg_show_fit_results` to print a summary and
`write_gg_fit` to save the result to an HDF5 file (readable by `read_gg_fit`).

See `examples/run_gg_fit.jl` for a complete, runnable example.

## Arguments

- `field::FieldGridTable` — The field grid and associated parameters.
- `params::GGFitInputParams` - Input fit parameters.
- `fit_at::Union{Nothing,Tuple{Int,Int}}` — Optional `(m_max, nd_max)` override.
  When given, `params.m_max` and `params.nd_max` are ignored and the scan is done
  at this one point only. Everything else in `params` still applies.
  This is the way to pick a particular fit out of a scan that has already been run:
  run the scan once, read the scan table printed by `gg_show_fit_results`, then
  refit at whichever `(m_max, nd_max)` row is wanted without editing `p`.

## How the fit works

The GG coefficients are computed at the equally spaced `z`-positions coincident with the
input field-table planes. These planes are sometimes called "principal planes".
The principal plane where a fit is being done is called the "base plane".
The fit is done plane by plane: the coefficients at a given base plane
are computed independently of the calculation of the coefficients at every other plane, and
all coefficients of a given plane are solved for simultaneously by minimizing a merit function

```
Merit = Σₖ  weightₖ · (field_from_tableₖ - field_from_GG_coefsₖ)^2
```

The sum runs over all field points lying in a plane within `n_planes_add` of the
base plane. For example, `n_planes_add = 2` adds two principal planes on either side, so
five planes are used in total. Near the ends of the table the count is reduced —
a base plane at the very end of the table uses only three planes when
`n_planes_add = 2`.

The merit function is a linear equation in the GG coefficients so
each base plane is solved by weighted linear least squares fit over all field
points lying within `n_planes_add` planes of the base plane.

Adding extra planes smooths the computed values between planes at the cost of making the fit
at the principal planes worse. So adding more planes can give a worse fit.

## Scanning over the fit size

Convention: `m` denotes the multipole order for GG functions `a_m` and `b_m` while
for `a(m, nd)`, `b(m, nd)`, and `bs(nd)`, the `nd` here denotes the derivative order.

A single fit to the field is done using GG coefficients up to some maximum multipole order
and some maximum derivative order. A "scan" is a series of fits using differing maximum 
multipole order and differing maximum derivative order. 
As discussed below, The "best" fit is chosen based upon the goodness of fit and how many
GG coefficients are needed for the fit.

The multipole orders used in a scan is set by `p.m_max` which can be an integer if only one order
is to be used, or can be an integer array where all values of the array will be used in the scan.
Similary, `p.nd_max` determines the range of maximum derivative orders.
For example:

```julia
p.m_max  = 1:10       # try m_max = 1, 2, … 10
p.nd_max = 2:6        # crossed with nd_max = 2, 3, … 6
```
This runs 66 fits. Note: For a given fit, the same maximum multipole order
and same maximum derivative order is used in the GG calculation for all planes.

To save time, the number of fits used in a scan can be fine tuned using `nd_max_for_m` which
is a per-multipole override of `nd_max`, as a `Dict` mapping `m` to a derivative limit.
For example: `nd_max_for_m = Dict(4 => 2, 5 => 1)` maps `a_4`/`b_4` to max `nd` = 2 and 
`a_5`/`b_5` to max `nd` = 1. Generally a good rule of thumb is that `m` + max `nd`
should be roughly constant.

## Excluding and pruning

A magnet with a symmetry may need only odd numbered multipole orders and not even.
The `exclude_functions` parameter can be used to exclude GG functions that are not needed.
Example:
    exclude_functions = [(:a, 2), (:a, 4), (:b, 2), (:b, 4), (:bs, 0)]
Note `b_s` has no multipole order, so the `m` of a `:bs` entry is ignored.

```
prune_ave_limit = 1e-4      # against the function's average contribution
prune_max_limit = 1e-3      # against its largest contribution
```

Pruning is applied after a scan has picked its `(m_max, nd_max)` winner, and
the surviving functions are then **refit**.

## Weighting

The weight of a field point at `(x, y)` and plane offset `dz` (relative to the
base plane) is the product of a transverse and a longitudinal factor:

```
weight(x,y,dz) = w_core(x,y) · w_plane(dz)
```

The transverse factor is determined by `core_weight`:
```
w_core(x,y) = core_weight · rmax^2 / (rmax^2 + r^2 · (core_weight - 1))
```

where `r^2 = x^2 + y^2` and `rmax` is the maximum `r` over all points.
`core_weight = 1` (the default) makes `w_core` constant; `core_weight > 1`
favors the core (low-`r`) points at the expense of points farther out. A better
core fit is usually desired since beam particles spend most of their time near
the core.

The longitudinal factor is set by `outer_plane_weight`:

```
w_plane(dz) = 1 + (outer_plane_weight - 1) · |dz| / dz_max
```

where `dz_max` is the largest `|dz|` at the ends of the fit region. If
`n_planes_add = 0` (so `dz_max = 0` and the expression is singular) `w_plane` is
set to 1. `outer_plane_weight = 1` (the default) makes `w_plane` constant; a
value between 0 and 1 weights planes nearer the base plane more than the outer
planes.

The `fit_radius_max` parameter is used to veto field points outside of the given radius in the
transverse plane.
The origin of the circle is the GG expansion axis origin (not necessarily `(0, 0)`). 
`0` meters (the default) uses every point of the field table.
Vetoing points outside of some radius is useful if the field table is inaccurate 
at large radius or a fit at large radius is not needed since this is outside of where
particles will travel.

Setting ``fit_radius_max` also re-scales `core_weight`, whose profile runs from `core_weight`
on the axis to `1` at the outermost *fitted* point — with a radius set, that is
the radius rather than the grid corner. `field_ave_plane` and the field
contributions that drive `prune_ave_limit`/`prune_max_limit` likewise cover the
fit region only, so pruning judges a GG function by the field it produces where
the fit applies. `gg_show_fit_residuals` reports the residual split at this
radius.

Setting `fit_radius_max` also moves three things onto the fit region, so that they keep
describing the same set of points the residuals do: `core_weight`, whose profile
runs from `core_weight` on the axis to `1` at the outermost *fitted* point;
`field_ave_plane`; and the field contributions that `prune_ave_limit` /
`prune_max_limit` act on, so a GG function is pruned on the field it produces
where the fit applies rather than at a corner it was never fitted to.

Note that `fit_radius_max` bounds the merit function, not the model. The fitted
GG functions still evaluate anywhere — including out at the corners, where they
are now an extrapolation rather than a fit.

## Choosing between fits (`fit_criterion`)

A fit using more GG coefficients will always have a lower RMS residual.
A scan therefore cannot pick its winner by residual alone — it needs a rule that charges for coefficients.
The input `fit_criterion` parameter selects that rule.
Each scanned fit gets a score and the lowest score wins.

Possible settings of `fit_criterion` are:
-  :aic    score = N * ln(RSS/N) + 2 * k         # Akaike information criterion
-  :bic    score = N * ln(RSS/N) + k * ln(N)     # Bayesian, and the default
where
```
RSS = Σ_planes Σ_points  weight · (field_from_table - field_from_GG_coefs)^2
N   = total number of fitted field-component values, summed over base planes
      = Σ_planes · 3 · (points in that plane's fit region)
k   = total number of fitted coefficients
      = (coefficients per plane) · (number of principal planes)
```
`RSS` is the weighted residual — the merit function the fit actually minimized,
pooled over every base plane. `N` counts `Bx`, `By` and `Bs` at each field point
of each plane's fit region, so a point used by several base planes is counted
once per base plane. `k` likewise counts the whole fit,
since each base plane is solved for its own copy of the coefficients.

These are the standard information criteria for a least-squares fit with unknown
error variance. The first term, `N·ln(RSS/N)`, is (up to an additive constant
that is the same for every candidate, and so cannot affect the ranking) minus
twice the maximized Gaussian log-likelihood; it rewards a smaller residual. The
second term is the penalty for fit size. They differ only in what one
coefficient costs: `2` for AIC, `ln(N)` for BIC. Since `ln(N) > 2` for any
`N > 7`, **BIC always penalizes more heavily than AIC and so selects equal or
smaller fits**. The scores are large negative numbers whose absolute value is
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
dominated by systematic truncation error — the field the fit cannot represent
— which is smooth and strongly correlated from point to point, not noise. The
effective number of independent measurements is therefore far below `N`, and the
penalty far weaker than the theory intends. Treat the criteria as a defensible
ranking heuristic rather than as a rigorous probability statement, and use the
scan table's `# coefs` and RMS columns to make the final call: the honest signal
is usually the point of diminishing returns, where the residual stops dropping
appreciably as coefficients are added.


## Other parameters

Other parameters in `GGFitInputParams`:

- `origin = [x0, y0]` — `(x, y)` line about which the GG coefficients are
  computed. If `field.g_ref` is non-zero, `origin` must be `[0, 0]`.
  Default `[0.0, 0.0]`.
- `output_file` — name of the output HDF5 file. Default `"gg_fit_results.h5"`.

## Side note

In theory, the fitting does not require that the field table be a rectangular
grid of equally spaced points. In fact, a set of randomly spaced field points would work.
Also there is no fundamental requirement that the fit planes be evenly spaced. It is only
for convenience that the `gg_calc_fit` function require a regularly spaced field table and that
the output is at evenly spaced planes.
"""
function gg_calc_fit(field::FieldGridTable, params::GGFitInputParams,
                     fit_at::Union{Nothing,Tuple{Int,Int}} = nothing)
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

  # ---- Resolve the (m_max, nd_max) fits to try -------------------------
  # The `fit_at` argument, when given, overrides (m_max, nd_max)
  # a chosen row of a scan already run.
  m_max_in, nd_max_in = fit_at === nothing ? (params.m_max, params.nd_max) : fit_at
  m_table_max  = maximum(k[1] for d in (a_dicts..., b_dicts...) for k in keys(d))
  nd_table_max = maximum(k[2] for d in (a_dicts..., b_dicts...) for k in keys(d))
  m_cands  = _fit_candidates(m_max_in, m_table_max)
  nd_cands = _fit_candidates(nd_max_in, nd_table_max)
  scanning = !(m_max_in isa Int && nd_max_in isa Int)

  params.fit_criterion in (:bic, :aic) ||
      error("Unknown fit_criterion: $(params.fit_criterion). Expecting :bic or :aic.")

  # The master fit is the union of every candidate: the design matrix is built
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

  # The transverse points the fit runs over: every grid point, or only those
  # within fit_radius_max of the expansion axis. Held as a list of
  # (grid index, coordinate index) pairs, since a radial cut does not leave a
  # rectangle behind. It is the same for every plane, so it is built once here.
  params.fit_radius_max >= 0 ||
      error("Negative fit_radius_max = $(params.fit_radius_max). Use 0 to fit every grid point.")
  rfit2 = params.fit_radius_max^2
  xy_pts = [(iix, iiy, ix, iy) for (iix, ix) in enumerate(ixs), (iiy, iy) in enumerate(iys)
            if rfit2 == 0 || xs[iix]^2 + ys[iiy]^2 <= rfit2 * (1 + 1e-12)]
  isempty(xy_pts) && error("fit_radius_max = $(params.fit_radius_max) leaves no field points " *
      "to fit. The nearest grid point to the GG axis is at r = " *
      "$(sqrt(minimum(xs[i]^2 + ys[j]^2 for i in eachindex(xs), j in eachindex(ys)))).")
  # The core weight runs from core_weight on the axis to 1 at the outermost
  # fitted point, so this follows the fit region rather than the grid.
  rmax2 = maximum(xs[p[1]]^2 + ys[p[2]]^2 for p in xy_pts)

  # ---- Assemble the master parameter (unknown) list from the table keys --
  # a/b indexed by (m,nd); bs indexed by nd (stored as (0,nd)). Functions the
  # user excluded are left out here, so they are never fitted at all: they cost
  # no design-matrix column and cannot influence the coefficients that are kept.
  # `nd_cap(m)` is the derivative cut for multipole order `m` (m = 0 is b_s),
  # tighter than `nd_max` wherever the user set a per-order override.
  excluded = _gg_exclude_set(params.exclude_functions)
  nd_caps  = _gg_nd_caps(params.nd_max_for_m)
  nd_cap(m, ndx = nd_max_all) = min(ndx, get(nd_caps, m, ndx))
  pset = Set{Tuple{Symbol,Int,Int}}()
  for d in a_dicts, k in keys(d)
    k[1] <= m_max_all && k[2] <= nd_cap(k[1]) && !((:a, k[1]) in excluded) &&
        push!(pset, (:a, k[1], k[2]))
  end
  for d in b_dicts, k in keys(d)
    k[1] <= m_max_all && k[2] <= nd_cap(k[1]) && !((:b, k[1]) in excluded) &&
        push!(pset, (:b, k[1], k[2]))
  end
  for d in bs_dicts, nd in keys(d)
    nd <= nd_cap(0) && !((:bs, 0) in excluded) && push!(pset, (:bs, 0, nd))
  end
  master_list = sort!(collect(pset))
  ncols_all   = length(master_list)
  ncols_all == 0 && error("No GG unknowns left to fit: exclude_functions = " *
      "$(params.exclude_functions) and nd_max_for_m = $(params.nd_max_for_m) remove " *
      "everything m_max = $m_max_in, nd_max = $nd_max_in would otherwise keep.")

  # Column subset of the master list belonging to each candidate fit. `bs`
  # unknowns describe a_0 and carry no multipole order, so `m_max` never drops them.
  # Candidates that the per-order caps (or an exclusion) collapse onto an earlier
  # candidate's column set are dropped: they would refit an identical model.
  cand_fits = Tuple{Int,Int}[]
  cand_cols = Vector{Int}[]
  for mx in m_cands, ndx in nd_cands
    cols = [c for (c, (typ, m, nd)) in enumerate(master_list)
              if nd <= nd_cap(typ == :bs ? 0 : m, ndx) && (typ == :bs || m <= mx)]
    any(isequal(cols), cand_cols) && continue
    push!(cand_fits, (mx, ndx))
    push!(cand_cols, cols)
  end

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
      (nd <= nd_cap(m) && m <= m_max_all) || continue
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

  # ---- Solve every candidate at every base plane ------------------------
  nplanes = length(izs_grid)
  z_base  = [r0[3] + dr[3] * iz for iz in izs_grid]
  ncand   = length(cand_fits)
  # Everything the per-plane solve needs that does not depend on which columns
  # are being fitted, so that the pruning pass below can re-solve a reduced
  # column set without rebuilding any of it.
  geom = (; mag, xy_pts, izs_grid, xs, ys, rmax2, npa, dz_grid, nd_max_all,
            core_weight, outer_plane_weight, master_list, col_terms, ncols_all)
  sol = _fit_over_planes(cand_cols, geom)
  (; nrow_pl, wsum_pl, wsum_comp_pl, field_ave_plane) = sol

  # ---- Score the candidates and pick the winner -------------------------
  ndata = sum(nrow_pl)
  wdata = sum(wsum_pl)
  wcomp = [sum(view(wsum_comp_pl, k, :)) for k in 1:3]
  scan  = GGFitScanPoint[]
  for c in 1:ncand
    mx, ndx = cand_fits[c]
    ncoef   = length(cand_cols[c])
    # Pool the per-plane residuals back into one sum of squares over all planes.
    rss  = sum(sol.rmsw_c[c][p]^2 * wsum_pl[p] for p in 1:nplanes)
    rssu = sum(sol.rmsu_c[c][p]^2 * nrow_pl[p] for p in 1:nplanes)
    rmswc = ntuple(k -> sqrt(sum(sol.rmsw_comp_c[c][k, p]^2 * wsum_comp_pl[k, p]
                                 for p in 1:nplanes) / wcomp[k]), 3)
    push!(scan, GGFitScanPoint(; m_max = mx, nd_max = ndx, n_coef = ncoef,
                                 rms_weighted = sqrt(rss / wdata),
                                 rms_weighted_comp = rmswc,
                                 rms_unweighted = sqrt(rssu / ndata),
                                 score = _fit_score(params.fit_criterion, rss, ndata,
                                                    ncoef * nplanes),
                                 criterion = params.fit_criterion))
  end
  best = argmin([s.score for s in scan])
  m_max, nd_max = cand_fits[best]

  # ---- Drop GG functions that produce negligible field ------------------
  # Measured on the winning fit, then the survivors are refit: dropping columns
  # from a finished solve would leave coefficients that no longer minimize
  # anything, since each was fitted in the presence of the ones now gone.
  keep_cols = cand_cols[best]
  theta     = sol.thetas[best]
  rmsw_pl   = sol.rmsw_c[best]
  rmsu_pl   = sol.rmsu_c[best]
  pruned    = Tuple{Symbol,Int}[]
  if params.prune_ave_limit > 0 || params.prune_max_limit > 0
    trial = GGFit(; z_base, _expand_coefs(master_list[keep_cols], theta)...,
                    fit_radius_max = params.fit_radius_max,
                    g_ref = field.g_ref, origin, dz_grid)
    rows, b_ave = _gg_field_contributions(trial, field)
    # A function has to fall below every limit that is in force; a limit of 0 is
    # switched off and so cannot by itself keep a function alive.
    drop = Set{Tuple{Symbol,Int}}()
    for (typ, m, ave, mx) in rows
      (params.prune_ave_limit <= 0 || ave < params.prune_ave_limit * b_ave) &&
        (params.prune_max_limit <= 0 || mx < params.prune_max_limit * b_ave) &&
        push!(drop, (typ, m))
    end
    if !isempty(drop)
      keep_cols = [c for c in keep_cols if !(_gg_group(master_list[c]) in drop)]
      isempty(keep_cols) && error("prune_ave_limit = $(params.prune_ave_limit), " *
          "prune_max_limit = $(params.prune_max_limit) drops every GG function. " *
          "Lower the limits: they are fractions of the mean |B| of the field table.")
      refit   = _fit_over_planes([keep_cols], geom)
      theta   = refit.thetas[1]
      rmsw_pl = refit.rmsw_c[1]
      rmsu_pl = refit.rmsu_c[1]
      pruned  = sort!(collect(drop))
    end
  end

  # ---- Expand the fitted coefficients into the dictionaries -------------
  params_list = master_list[keep_cols]
  a, b, bs = _expand_coefs(params_list, theta)

  # Report the orders actually retained. Excluding or pruning the functions at the
  # top of the range leaves these below the cutoff the scan nominally selected.
  m_max  = maximum((m for (typ, m, _) in params_list if typ !== :bs), init = 0)
  nd_max = maximum((nd for (_, _, nd) in params_list), init = 0)

  return GGFit(; z_base, params = params_list, a, b, bs,
                  rms_weighted_plane = rmsw_pl, rms_unweighted_plane = rmsu_pl,
                  field_ave_plane, fit_radius_max = params.fit_radius_max,
                  m_max, nd_max, pruned,
                  scan = scanning ? scan : GGFitScanPoint[],
                  g_ref = field.g_ref, origin, dz_grid)
end

#---------------------------------------------------------------------------------------------------

"""
    _gg_group(param) -> (typ, m)

The GG function a `(typ, m, nd)` unknown belongs to: all derivative orders of one
`a_m` or `b_m` share a group, and every `bs` unknown lands in `(:bs, 0)`. This is
the granularity at which contributions are measured and functions are pruned.
"""
_gg_group(param::Tuple{Symbol,Int,Int}) =
    param[1] === :bs ? (:bs, 0) : (param[1], param[2])

"""
    _gg_label(typ, m) -> String

Printable name of a GG function group: `"a_3"`, `"b_5"`, `"b_s"`.
"""
_gg_label(typ::Symbol, m::Integer) = typ === :bs ? "b_s" : string(typ, "_", m)

"""
    _gg_exclude_set(exclude) -> Set{Tuple{Symbol,Int}}

Validate and normalize a user `exclude_functions` list into the group form used
throughout. `b_s` carries no multipole order, so any `(:bs, m)` normalizes to
`(:bs, 0)`. Naming a function the model does not contain is a harmless no-op; a
type other than `:a`, `:b` or `:bs` is a typo and raises.
"""
function _gg_exclude_set(exclude)
  s = Set{Tuple{Symbol,Int}}()
  for e in exclude
    typ, m = e
    if typ === :bs
      push!(s, (:bs, 0))
    elseif typ === :a || typ === :b
      m >= 0 || error("Negative multipole order $m for $(repr(typ)) in exclude_functions.")
      push!(s, (typ, m))
    else
      error("Unknown GG function type $(repr(typ)) in exclude_functions. " *
            "Expecting :a, :b or :bs, as in exclude_functions = [(:a, 2), (:b, 4), (:bs, 0)].")
    end
  end
  return s
end

"""
    _gg_nd_caps(nd_max_for_m) -> Dict{Int,Int}

Validate a user `nd_max_for_m` setting, the per-multipole override of `nd_max`.
Keys are multipole orders (`0` being `b_s`, which carries no order) and values
the highest derivative order kept for that multipole. Naming an `m` the
coefficient table does not contain is a harmless no-op; a negative order or a
negative limit is a mistake and raises.
"""
function _gg_nd_caps(nd_max_for_m)
  caps = Dict{Int,Int}()
  for (m, nd) in nd_max_for_m
    m >= 0 || error("Negative multipole order $m in nd_max_for_m. " *
                    "Use 0 for b_s, which carries no multipole order.")
    nd >= 0 || error("Negative derivative-order limit $nd for m = $m in nd_max_for_m. " *
                     "Use exclude_functions to remove a GG function entirely.")
    caps[m] = nd
  end
  return caps
end

"""
    _expand_coefs(params_list, theta) -> (a, b, bs)

Scatter a solved coefficient matrix (`theta[col, plane]`, rows following
`params_list`) into the `a`, `b` and `bs` dictionaries of a `GGFit`. Only the
unknowns in `params_list` get keys, so a pruned function is simply absent.
"""
function _expand_coefs(params_list, theta)
  a  = Dict{Tuple{Int,Int},Vector{Float64}}()
  b  = Dict{Tuple{Int,Int},Vector{Float64}}()
  bs = Dict{Int,Vector{Float64}}()
  for (col, (typ, m, nd)) in enumerate(params_list)
    typ === :a  && (a[(m, nd)]  = theta[col, :])
    typ === :b  && (b[(m, nd)]  = theta[col, :])
    typ === :bs && (bs[nd]      = theta[col, :])
  end
  return (; a, b, bs)
end

#---------------------------------------------------------------------------------------------------

"""
    _fit_over_planes(cand_cols, geom) -> NamedTuple

Solve the weighted least-squares fit at every base plane, for every candidate
column subset in `cand_cols`, from one shared design matrix per plane. `geom`
carries the grid, weighting and basis data that does not depend on which columns
are fitted (see the call site in `gg_calc_fit`), so this can be re-run on a reduced
column set — as the pruning pass does — without rebuilding any of it.

Returns `(; thetas, rmsw_c, rmsu_c, rmsw_comp_c, nrow_pl, wsum_pl, wsum_comp_pl,
field_ave_plane)`: per candidate the fitted coefficients (`thetas[c][col, plane]`,
rows following that candidate's column subset), the weighted and unweighted
per-plane residuals, and the weighted residual split by field component; then the
per-plane row counts and weight sums, and the average `|B|` of each base plane.
"""
function _fit_over_planes(cand_cols, geom)
  (; mag, xy_pts, izs_grid, xs, ys, rmax2, npa, dz_grid, nd_max_all,
     core_weight, outer_plane_weight, master_list, col_terms, ncols_all) = geom

  nplanes = length(izs_grid)
  ncand   = length(cand_cols)
  thetas  = [zeros(length(cols), nplanes) for cols in cand_cols]
  rmsw_c  = [fill(NaN, nplanes) for _ in 1:ncand]
  rmsu_c  = [fill(NaN, nplanes) for _ in 1:ncand]
  # Same weighted residual as rmsw_c but split by field component: [comp, plane]
  # with comp = 1, 2, 3 for Bx, By, Bs. Reported per candidate in the scan table
  # so a fit that fails on one component only can be recognized as such.
  rmsw_comp_c  = [fill(NaN, 3, nplanes) for _ in 1:ncand]
  nrow_pl      = zeros(Int, nplanes)
  wsum_pl      = zeros(nplanes)        # Σ weight over the rows of each plane's fit
  wsum_comp_pl = zeros(3, nplanes)     # the same sum, per field component
  field_ave_plane = fill(NaN, nplanes)

  for (pidx, iz0) in enumerate(izs_grid)
    izs   = max(first(izs_grid), iz0 - npa):min(last(izs_grid), iz0 + npa)
    dzs   = [(iz - iz0) * dz_grid for iz in izs]
    dzmax = maximum(abs, dzs)

    npts  = length(xy_pts) * length(izs)
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
      for (iix, iiy, ix, iy) in xy_pts
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
    wsum_pl[pidx] = sum(abs2, sw)
    # Rows cycle Bx, By, Bs at each field point, so component `k` owns rows k:3:nrows.
    for k in 1:3
      wsum_comp_pl[k, pidx] = sum(abs2, view(sw, k:3:nrows))
    end
    for c in 1:ncand
      cols  = cand_cols[c]
      Awc   = length(cols) == ncols_all ? Aw : Aw[:, cols]
      th    = pinv(Awc) * bw
      thetas[c][:, pidx] = th
      # Residual of the same fit, scored with and without the point weights. The
      # unweighted value says how well the field itself is reproduced; the weighted
      # one is the quantity the fit actually minimized. The weighted residual is
      # normalized by Σ weight rather than by the row count so that it stays a
      # weighted mean of the squared deviations (and so reduces to the unweighted
      # value when every weight is 1).
      resw = Awc * th - bw
      rmsw_c[c][pidx] = norm(resw) / sqrt(wsum_pl[pidx])
      for k in 1:3
        rmsw_comp_c[c][k, pidx] = norm(view(resw, k:3:nrows)) / sqrt(wsum_comp_pl[k, pidx])
      end
      rmsu_c[c][pidx] = norm(view(A, :, cols) * th - bvec) / sqrt(nrows)
    end

    # Average |B| over the base plane alone (unweighted): the field scale against
    # which the residuals above are to be judged. Over the fitted points only, so
    # that it stays comparable with them when a fit radius is set.
    field_ave_plane[pidx] = sum(norm(mag[p[3], p[4], iz0]) for p in xy_pts) / length(xy_pts)
  end

  return (; thetas, rmsw_c, rmsu_c, rmsw_comp_c, nrow_pl, wsum_pl, wsum_comp_pl,
            field_ave_plane)
end

#---------------------------------------------------------------------------------------------------

"""
    _fit_candidates(spec, table_max) -> Vector{Int}

Resolve an `m_max`/`nd_max` input-parameter setting into the list of values
`gg_calc_fit` should try. An `Int` pins that value; a vector or range is taken
as-is. Values are clamped to `table_max` (the largest the coefficient table
supports) and deduplicated, so an over-wide request such as `0:100` does not
refit the same top fit repeatedly.
"""
function _fit_candidates(spec::Union{Int,AbstractVector{Int}}, table_max::Int)
  vals = spec isa Int ? [spec] : collect(spec)
  isempty(vals) && error("Empty m_max/nd_max candidate list.")
  any(v -> v < 0, vals) && error("Negative m_max/nd_max candidate in $spec.")
  return sort!(unique!([min(v, table_max) for v in vals]))
end

"""
    _fit_score(criterion, rss, ndata, nparam) -> Float64

Selection score for one scanned fit; the lowest score wins. `rss` is the pooled
weighted sum of squared residuals over `ndata` field-component values, and
`nparam` the total number of fitted coefficients (per-plane count times the
number of base planes). With `RSS = rss`, `N = ndata` and `k = nparam`:

```
:aic    N*log(RSS/N) + 2k              Akaike information criterion
:bic    N*log(RSS/N) + k*log(N)        Bayesian information criterion
```

`N` in the log term is the count of data values entering the Gaussian
log-likelihood, not a weight total.

`:aic` and `:bic` share the leading term — minus twice the maximized Gaussian
log-likelihood, dropping an additive constant that is common to every candidate
and so cannot change the ranking — and differ only in the cost of one
coefficient, `2` versus `log(N)`. See the `gg_calc_fit` docstring for how to read the
trade-off and for the caveats that apply when the residual is dominated by
systematic truncation error rather than by noise.
"""
function _fit_score(criterion::Symbol, rss::Float64, ndata::Int, nparam::Int)
  # Guard the log against an (effectively) exact fit, which a fit with as many
  # coefficients as data points can produce.
  ll = ndata * log(max(rss, floatmin(Float64)) / ndata)
  criterion == :bic && return ll + nparam * log(ndata)
  return ll + 2 * nparam                                  # :aic
end

#---------------------------------------------------------------------------------------------------

"""
    _gg_field_contributions(gg_fit::GGFit, field::FieldGridTable) -> (rows, b_ave)

How much field each fitted GG function actually produces, measured over every
fitted transverse grid point of every base plane — that is, over
`gg_fit.fit_radius_max` when one is set, and over the whole grid otherwise.

`rows` holds one `(typ, m, ave, max)` per GG function — `(:a, m)` and `(:b, m)`
for each multipole order `m` present in the fit, plus `(:bs, 0)` — where `ave`
and `max` are the mean and the largest `|B|` [T] that function generates with
every other GG coefficient set to zero. All derivative orders `nd` of the
function are included, since they too contribute to the field at the plane. The
field is linear in the GG coefficients, so a row is exactly what dropping that
function from the fit would remove from the modeled field. `b_ave` is the mean
`|B|` of the field table itself, the scale the rows are to be read against.

Both `gg_show_fit_results` and the pruning pass of `gg_calc_fit` read these rows —
which is why they carry the `(typ, m)` group rather than a printable label.

This is the useful form of "how big is this coefficient": the raw `a`/`b` values
are not comparable across `m`, since the basis function each multiplies carries a
different power of `r` and so a different size over the grid.
"""
function _gg_field_contributions(gg_fit::GGFit, field::FieldGridTable)
  mag = field.magnetic
  ixs = first(axes(mag, 1)):last(axes(mag, 1))
  iys = first(axes(mag, 2)):last(axes(mag, 2))
  # Coordinates relative to the GG expansion axis, as the coefficient tables want.
  xs  = [field.r0[1] + field.dr[1] * ix - gg_fit.origin[1] for ix in ixs]
  ys  = [field.r0[2] + field.dr[2] * iy - gg_fit.origin[2] for iy in iys]
  # Restricted to the fit region: a function's field outside it is not something
  # the fit ever tried to produce, and at the grid corners it is large enough to
  # decide the pruning on its own.
  rfit2 = gg_fit.fit_radius_max^2
  pts   = [(i, j) for i in eachindex(xs), j in eachindex(ys)
           if rfit2 == 0 || xs[i]^2 + ys[j]^2 <= rfit2 * (1 + 1e-12)]
  nplanes = length(gg_fit.z_base)
  npts    = nplanes * length(pts)

  groups = vcat([(:a, m) for m in sort!(unique(k[1] for k in keys(gg_fit.a)))],
                [(:b, m) for m in sort!(unique(k[1] for k in keys(gg_fit.b)))],
                isempty(gg_fit.bs) ? Tuple{Symbol,Int}[] : [(:bs, 0)])

  rows = Tuple{Symbol,Int,Float64,Float64}[]
  for (typ, m) in groups
    tot = 0.0
    mx  = 0.0
    for ip in 1:nplanes
      # Value getters masked to this one group: every coefficient outside it reads
      # as zero, so the polynomials below are this function's share of the field.
      aval(mm, nd) = typ === :a && mm == m ? get(gg_fit.a, (mm, nd), nothing) : nothing
      bval(mm, nd) = typ === :b && mm == m ? get(gg_fit.b, (mm, nd), nothing) : nothing
      bsval(nd)    = typ === :bs ? get(gg_fit.bs, nd, nothing) : nothing
      at(v) = v === nothing ? 0.0 : v[ip]
      KBx = _comp_array(Bx_a, Bx_b, Bx_bs, (mm, nd) -> at(aval(mm, nd)),
                        (mm, nd) -> at(bval(mm, nd)), nd -> at(bsval(nd)), gg_fit.g_ref)
      KBy = _comp_array(By_a, By_b, By_bs, (mm, nd) -> at(aval(mm, nd)),
                        (mm, nd) -> at(bval(mm, nd)), nd -> at(bsval(nd)), gg_fit.g_ref)
      KBs = _comp_array(Bs_a, Bs_b, Bs_bs, (mm, nd) -> at(aval(mm, nd)),
                        (mm, nd) -> at(bval(mm, nd)), nd -> at(bsval(nd)), gg_fit.g_ref)
      for (i, j) in pts
        x, y = xs[i], ys[j]
        nb = sqrt(_polyval(KBx, x, y)[1]^2 + _polyval(KBy, x, y)[1]^2 +
                  _polyval(KBs, x, y)[1]^2)
        tot += nb
        mx   = max(mx, nb)
      end
    end
    push!(rows, (typ, m, npts == 0 ? NaN : tot / npts, mx))
  end

  # The scale the rows are read against, over the same points they cover.
  bsum = 0.0
  for iz in axes(mag, 3), (i, j) in pts
    bsum += norm(mag[ixs[i], iys[j], iz])
  end
  b_ave = npts == 0 ? NaN : bsum / (length(pts) * size(mag, 3))
  return rows, b_ave
end

#---------------------------------------------------------------------------------------------------

"""
    gg_show_fit_results(gg_fit::GGFit, field::FieldGridTable, params::GGFitInputParams)

Print a human-readable summary of a `gg_calc_fit` result `gg_fit`: the fit settings, the
`(m_max, nd_max)` scan table if a scan was run, a per-plane table of the weighted
and unweighted RMS residuals alongside the average field magnitude of the plane,
and the leading multipoles at the central plane as a quick sanity check.

The scan table breaks the weighted residual out by field component (`wRMS Bx`,
`wRMS By`, `wRMS Bs`) next to the pooled `wRMS resid`. When a fit is poor, the
split says whether all three components are equally bad — pointing at a model
too small, or at a grid the GG expansion cannot represent — or whether one
component alone carries the error.
"""
function gg_show_fit_results(gg_fit::GGFit, field::FieldGridTable, params::GGFitInputParams)
  println("="^72)
  println("GG fit:")
  println("  field grid        : ", join(size(field.magnetic), " x "), "  (ix, iy, iz)")
  # Grid extent from the index ranges, which need not start at 0 or 1:
  # a grid point (ix, iy, iz) sits at r0 + dr .* (ix, iy, iz).
  ax   = axes(field.magnetic)
  gmin = [field.r0[k] + field.dr[k] * first(ax[k]) for k in 1:3]
  gmax = [field.r0[k] + field.dr[k] * last(ax[k]) for k in 1:3]
  @printf("  grid spacing [m]  : %.6g, %.6g, %.6g   (dx, dy, dz)\n", field.dr...)
  for (k, nam) in enumerate(("x", "y", "z"))
    @printf("  grid %s range [m]  : %.6g to %.6g\n", nam, gmin[k], gmax[k])
  end
  println("  g_ref             : ", field.g_ref)
  println("  origin (x,y)      : ", params.origin)
  println("  n_planes_add      : ", params.n_planes_add)
  println("  m_max, nd_max     : ", gg_fit.m_max, ", ", gg_fit.nd_max,
          isempty(gg_fit.scan) ? "" : "   (selected by the scan below)")
  if !isempty(params.nd_max_for_m)
    println("  nd_max per m      : ",
            join((m == 0 ? "b_s = $nd" : "a_$m/b_$m = $nd"
                  for (m, nd) in sort!(collect(params.nd_max_for_m))), ", "),
            "   (overrides nd_max)")
  end
  println("  fit_radius_max    : ", params.fit_radius_max,
          params.fit_radius_max > 0 ? "   (points beyond this radius are not fitted)" :
                                      "   (0 = every grid point is fitted)")
  println("  core_weight       : ", params.core_weight)
  println("  outer_plane_weight: ", params.outer_plane_weight)
  if !isempty(params.exclude_functions)
    println("  excluded functions: ",
            join((_gg_label(typ, m) for (typ, m) in
                  sort!(collect(_gg_exclude_set(params.exclude_functions)))), ", "),
            "   (never fitted)")
  end
  println("  prune ave, max    : ", params.prune_ave_limit, ", ", params.prune_max_limit,
          "   (fraction of <|B|>; 0 = off)")
  if !isempty(gg_fit.pruned)
    println("  pruned functions  : ",
            join((_gg_label(typ, m) for (typ, m) in gg_fit.pruned), ", "),
            "   (", length(gg_fit.pruned), " dropped, the rest refit)")
  end
  println("  # GG coefficients : ", length(gg_fit.params), " per plane")
  println("  # base planes     : ", length(gg_fit.z_base))

  # ---- (m_max, nd_max) scan table, when a scan was run -------------------
  if !isempty(gg_fit.scan)
    crit = gg_fit.scan[1].criterion
    println("-"^104)
    println("Model scan (", length(gg_fit.scan), " combinations, best by ", crit, " marked *):")
    println("The three per-component columns are the wRMS residual of Bx, By and Bs alone.")
    @printf("%-2s %-5s  %-6s  %-7s  %-11s  %-11s  %-11s  %-11s  %-11s  %-12s\n",
            "", "m_max", "nd_max", "# coefs", "wRMS resid", "wRMS Bx", "wRMS By", "wRMS Bs",
            "RMS resid", string(crit))
    bestscore = minimum(s.score for s in gg_fit.scan)
    for s in gg_fit.scan
      @printf("%-2s %-5d  %-6d  %-7d  %-11.4e  %-11.4e  %-11.4e  %-11.4e  %-11.4e  %-12.6g\n",
              s.score == bestscore ? "*" : "", s.m_max, s.nd_max, s.n_coef,
              s.rms_weighted, s.rms_weighted_comp[1], s.rms_weighted_comp[2],
              s.rms_weighted_comp[3], s.rms_unweighted, s.score)
    end
  end

  println("-"^72)
  @printf("%-6s  %-12s  %-12s  %-12s  %-12s\n",
          "plane", "z [m]", "wRMS resid", "RMS resid", "<|B|> [T]")
  # A `GGFit` built by hand may leave these diagnostics unset; blank the column.
  cell(v, i) = length(v) == length(gg_fit.z_base) ? @sprintf("%-12.4e", v[i]) :
                                                     @sprintf("%-12s", "-")
  for i in eachindex(gg_fit.z_base)
    @printf("%-6d  %-12.6g  %-12.4e  %s  %s\n", i, gg_fit.z_base[i], gg_fit.rms_weighted_plane[i],
            cell(gg_fit.rms_unweighted_plane, i), cell(gg_fit.field_ave_plane, i))
  end
  println("-"^72)

  # How much field each GG function actually produces. This, rather than the raw
  # coefficient values, is what says whether a function is pulling its weight:
  # the values themselves are not comparable across m, each multiplying a basis
  # function with a different power of r and so a different size over the grid.
  rows, b_ave = _gg_field_contributions(gg_fit, field)
  println("Field contribution of each GG function, over all grid points of all base planes.")
  println("|B| from that function alone, every other GG coefficient zeroed:")
  @printf("  %-9s %-14s %-14s %10s\n", "function", "ave |B| [T]", "max |B| [T]", "ave/<|B|>")
  for (typ, m, ave, mx) in rows
    @printf("  %-9s %-14.4e %-14.4e %8.2f %%\n", _gg_label(typ, m), ave, mx, 100 * ave / b_ave)
  end
  @printf("  <|B|> of the field table = %.4e T\n", b_ave)
  println("="^72)
end

#---------------------------------------------------------------------------------------------------

"""
    write_gg_fit(gg_fit::GGFit, field::FieldGridTable, params::GGFitInputParams) -> output_file_path

Write a `gg_calc_fit` result `gg_fit` to an HDF5 file (readable by `read_gg_fit`).

Stores the fitted GG coefficients plus enough metadata to reproduce and
interpret the fit later. The (large) input field table is NOT stored. The file
is written to `params.output_file` and its path is returned.

## HDF5 schema

    root datasets   : z_base, rms_weighted_plane, rms_unweighted_plane,
                      field_ave_plane, origin                     (Float64[])
    root attributes : g_ref, dz_grid (Float64); m_max, nd_max, n_planes_add (Int);
                      fit_radius_max, core_weight, outer_plane_weight (Float64)
    groups a, b     : m (Int[]), nd (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{(m,nd) => values[i,:]}
    group  bs       : nd (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{nd => values[i,:]}
"""
function write_gg_fit(gg_fit::GGFit, field::FieldGridTable, params::GGFitInputParams)
  outfile = params.output_file
  h5open(outfile, "w") do f
    f["z_base"]    = collect(Float64, gg_fit.z_base)
    f["rms_weighted_plane"]   = collect(Float64, gg_fit.rms_weighted_plane)
    f["rms_unweighted_plane"] = collect(Float64, gg_fit.rms_unweighted_plane)
    f["field_ave_plane"]      = collect(Float64, gg_fit.field_ave_plane)
    f["origin"]    = collect(Float64, params.origin)
    attributes(f)["m_max"]              = Int(gg_fit.m_max)
    attributes(f)["nd_max"]             = Int(gg_fit.nd_max)
    attributes(f)["fit_radius_max"]     = Float64(gg_fit.fit_radius_max)
    attributes(f)["g_ref"]              = Float64(gg_fit.g_ref)
    attributes(f)["dz_grid"]            = Float64(field.dr[3])
    # Fit-control parameters, retained for later reference / reproducibility.
    attributes(f)["n_planes_add"]       = Int(params.n_planes_add)
    attributes(f)["core_weight"]        = Float64(params.core_weight)
    attributes(f)["outer_plane_weight"] = Float64(params.outer_plane_weight)
    _write_coef_group(f, "a", gg_fit.a)
    _write_coef_group(f, "b", gg_fit.b)
    _write_coef_group(f, "bs", gg_fit.bs; single = true)
  end
  println("Results written to ", outfile)
  return outfile
end
