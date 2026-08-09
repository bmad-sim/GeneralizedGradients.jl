@enumx GridAnchorPt Beginning Center End
@enumx GridGeometry XYZ 

"""
    mutable struct FieldGridTable{T}

Holds an electric and/or magnetic field sampled on a 3D grid.

`magnetic` and `electric` are 3D `OffsetArray`s whose elements are field
3-vectors: `magnetic[ix,iy,iz] == [Bx, By, Bz]` (and likewise `[Ex, Ey, Ez]`).
The grid indices `(ix, iy, iz)` need not start at 0 or 1; a grid point is at
position `r0 + dr .* (ix, iy, iz)` relative to the anchor.

## Fields

- `magnetic::OffsetArray{Vector{T}}` — magnetic field 3-vectors `[Bx, By, Bz]` [T].
- `electric::OffsetArray{Vector{T}}` — electric field 3-vectors `[Ex, Ey, Ez]` [V/m].
- `r0::Vector{T}` — grid origin offset `(x0, y0, z0)` [m].
- `dr::Vector{T}` — grid spacing `(dx, dy, dz)` [m].
- `g_ref::T` — curvilinear-coordinate bending strength `1/bending_radius`, in 1/m
  (`0` for a straight reference curve).
- `scale::T` — overall field scale factor.
- `RF_frequency::T` — RF frequency in Hz (`0` for a static field).
- `RF_phase::T` — RF phase [rad].
- `anchor_pt::GridAnchorPt.T` — grid anchor point: `Beginning`, `Center`, or `End`.
- `geometry::GridGeometry.T` — grid geometry: `XYZ`.

`FieldGridTable()` builds an empty table with `T = Float64`; read one from a
file with `read_field_grid_hdf5`.
"""
@kwdef mutable struct FieldGridTable{T}
  magnetic::OffsetArray{Vector{T}} = OffsetArray(Array{Vector{Float64}}(undef, 0, 0, 0), 1:0, 1:0, 1:0)
  electric::OffsetArray{Vector{T}} = OffsetArray(Array{Vector{Float64}}(undef, 0, 0, 0), 1:0, 1:0, 1:0)
  r0::Vector{T} = [0.0, 0.0, 0.0]
  dr::Vector{T} = [0.0, 0.0, 0.0]
  g_ref::T = 0.0
  scale::T = 1.0
  RF_frequency::T = 0.0
  RF_phase::T = 0.0
  anchor_pt::GridAnchorPt.T = GridAnchorPt.Center
  geometry::GridGeometry.T = GridGeometry.XYZ
end

# For a, b that are structs, the default `a == b` will only be true if `a` and `b` are equivalent (a === b).
# Here redefine `==` for `FieldGridTable` to return true if all fields are equal.

Base.:(==)(a::FieldGridTable, b::FieldGridTable) = all(getfield(a,f) == getfield(b,f) for f in fieldnames(FieldGridTable))

#---------------------------------------------------------------------------------------------------
# GGFitInputParams

"""
    mutable struct GGFitInputParams

Input parameters controlling a `gg_calc_fit` fit.
See the documentation of `gg_calc_fit` for more documentation.

## Fields

- `origin::Vector{Float64}` — Defines the line `[x0, y0, z]` about which the  generalized gradient 
  coefficients are computed. If h (1/bending_radius) is non-zero, origin must be `[0, 0]`.

- `n_planes_add::Int` — Number of z-planes added to either side of the base
  z-plane to be used in the analysis of the derivatives at any given base z-plane.
  For example, for `n_planes_add = 2`, two planes would be added to either side of the base plane
  making the total number of planes used in the analysis equal to five.

- `core_weight::Float64` — Merit function weight for "core" points (field table
  points whose transverse (x,y) position is near (0,0)). Default is 1.0 which
  gives an equal weight for all points of a given z-plane.

- `outer_plane_weight::Float64` — Default is 1.0. Merit function weight for z-planes away from
  the base z-plane when `n_planes_add` is non-zero.

- `m_max::Union{Int,AbstractVector{Int}}` — Maximum multipole order `m` of the
  `a_m`/`b_m` functions used in a fit. `m_max` can be a single integer or a
  range. For example, `m_max = 2:8` makes `gg_calc_fit` try each value in turn
  and keep the one that fits best (see `fit_criterion`). The default is `4:8`.
  The `bs(nd)` (that is, `a_0` derivative) unknowns carry no `m` and
  are never removed by `m_max`.

- `nd_max::Union{Int,AbstractVector{Int}}` — Maximum derivative order `nd`
  used in the fit. `nd_max` can be a single integer or a range. 
  For example, `nd_max = 2:8` makes `gg_calc_fit` try each value in turn
  and keep the one that fits best (see `fit_criterion`). The default is `3:7`

- `nd_max_for_m::Dict{Int,Int}` — Per-multipole order override of `nd_max`, mapping a
  multipole order `m` to the highest derivative order kept for `a_m` and `b_m`.
  Key `0` sets the limit for `b_s`, which carries no multipole order. This is used
  to speed up fitting.

  Example: `nd_max_for_m = Dict(4 => 2, 5 => 1)` maps `a_4`/`b_4` to max `nd` = 2 and 
  `a_5`/`b_5` to max `nd` = 1. Generally a good rule of thumb is that `m` + max `nd`
  should be roughly constant.

- `fit_radius_max::Float64` — Fit only the field points that are within this radius
  transversely of the GG expansion axis origin (not necessarily `(0, 0)`). 
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

- `fit_criterion::Symbol` — How a scan picks its winner. Each candidate model is
  given a score and the lowest score wins. Enlarging the model can only lower the
  residual (the smaller model's unknowns are a subset of the larger one's), so a
  usable criterion has to charge for coefficients. Writing `RSS` for the weighted
  sum of squared residuals pooled over all base planes, `N` for the number of
  fitted field-component values, and `k` for the total number of fitted
  coefficients (per-plane count times the number of base planes):

  ```
  :aic    score = N * ln(RSS/N) + 2 * k         # Akaike information criterion
  :bic    score = N * ln(RSS/N) + k * ln(N)     # Bayesian, and the default
  ```

  `:aic` and `:bic` share the same first term — minus twice the maximized
  Gaussian log-likelihood, up to a constant common to all candidates — and differ
  only in what one coefficient costs: `2` versus `ln(N)`. `ln(N) > 2` for any
  `N > 7`, so `:bic` always penalizes size at least as hard as `:aic` and never
  selects a larger model.

  See the "Choosing between models" section of the `gg_calc_fit` docstring for
  how to read the trade-off quantitatively and for why, on a field grid, these
  criteria are better treated as a ranking heuristic than as a probability
  statement.

- `exclude_functions::Vector{Tuple{Symbol,Int}}` — GG functions to leave out of
  the fit entirely, named as `(:a, m)`, `(:b, m)` or `(:bs, 0)` tuples — the same
  form the `pruned` field of the result uses.

  ```
  exclude_functions = [(:a, 2), (:a, 4), (:b, 2), (:b, 4), (:bs, 0)]
  ```

  An excluded function is dropped when the list of unknowns is assembled, so it
  never gets a design-matrix column: it is not fitted, cannot influence the
  coefficients that are kept, and is absent from the result. `b_s` carries no
  multipole order, so the `m` of a `:bs` entry is ignored. Naming a function the
  model does not contain anyway is a harmless no-op.

  Use this where the answer is known in advance — a magnet whose symmetry forbids
  the even multipoles, say — since `m_max` can only cut at the top of the range
  while this removes orders from anywhere in it. Where the answer is not known in
  advance, `prune_ave_limit`/`prune_max_limit` below decide the same question
  from the fit itself.

- `prune_ave_limit::Float64`, `prune_max_limit::Float64` — Drop GG functions that
  produce negligible field, so they are neither fitted nor stored. A function
  here is a whole `a_m`, a whole `b_m`, or `b_s` — all of its derivative orders
  `nd` together — and its contribution is the `|B|` it alone produces, every
  other GG coefficient set to zero, measured over every transverse grid point of
  every base plane.

  Both limits are fractions of the field table's mean `|B|`, so they carry over
  unchanged between magnets of different strength. `prune_ave_limit` is compared
  against the function's average contribution and `prune_max_limit` against its
  largest. A function is dropped only when it falls below **every** limit that is
  in force; a limit of `0` (the default for both) switches that test off, and
  with both off no pruning is done at all.

  ```
  prune_ave_limit = 1e-4      # drop if the average contribution is under 0.01% of <|B|>
  prune_max_limit = 1e-3      # ... and the largest contribution is under 0.1% of <|B|>
  ```

  Setting only `prune_max_limit` is the conservative choice: a function survives
  if it matters anywhere on the grid, even if it averages to little. Setting only
  `prune_ave_limit` prunes harder, and can discard a function that is small on
  average but significant near the aperture, where the max lives.

  Pruning is applied after a scan has picked its `(m_max, nd_max)` winner, and
  the surviving functions are then **refit**. The stored coefficients are
  therefore the least-squares solution of the model that was kept, not the
  leftovers of a larger fit. The functions removed are listed in the `pruned`
  field of the result.

- `output_file::String` — Name of the output file.
"""
@kwdef mutable struct GGFitInputParams
  origin::Vector{Float64} = [0.0, 0.0]   # (x, y) origin about which the generalized gradients coefs are computed
  n_planes_add::Int = 1                  # Number of z-planes added.
  m_max::Union{Int,AbstractVector{Int}} =  4:8  # Max multipole order m.
  nd_max::Union{Int,AbstractVector{Int}} = 3:7  # Max derivative order nd. 
  nd_max_for_m::Dict{Int,Int} = Dict{Int,Int}()  # Per-order override of nd_max: m => nd_max for a_m and b_m (m = 0 => b_s).
  fit_criterion::Symbol = :bic           # Scan selection criterion: :bic or :aic.
  fit_radius_max::Float64 = 0.0          # Only fit field points within this radius of the GG axis. 0 = use every point.
  core_weight::Float64 = 1.0             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
  outer_plane_weight::Float64 = 1.0      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
  exclude_functions::Vector{Tuple{Symbol,Int}} = Tuple{Symbol,Int}[]  # GG functions to leave out of the fit: (:a,m), (:b,m), (:bs,0).
  prune_ave_limit::Float64 = 0.0         # Drop a GG function whose ave contribution is below this fraction of <|B|>. 0 = test off.
  prune_max_limit::Float64 = 0.0         # Drop a GG function whose max contribution is below this fraction of <|B|>. 0 = test off.
  output_file::String = "gg_fit_results.h5"
end

#---------------------------------------------------------------------------------------------------
"""
    struct GGFitScanPoint

One row of a `gg_calc_fit` `(m_max, nd_max)` scan: the model tried and how it scored.
Collected in the `scan` field of the returned `GGFit` and printed by
`gg_show_fit_results`.

## Fields

- `m_max::Int`, `nd_max::Int` — the model this row is for.
- `n_coef::Int` — number of fitted coefficients per base plane.
- `rms_weighted::Float64` — weighted RMS residual pooled over all base planes,
  `sqrt(Σ w·δ² / Σ w)` over every fitted field-component value.
- `rms_weighted_comp::NTuple{3,Float64}` — the same quantity restricted to one
  field component at a time, as the 3-tuple `(Bx, By, Bs)`. Each entry is
  normalized by that component's own weight sum, so the three are directly
  comparable to each other and to `rms_weighted`, which is their weighted
  quadrature mean. A fit that is bad in only one component shows up here and
  nowhere else.
- `rms_unweighted::Float64` — the same residual with all point weights set to 1.
- `score::Float64` — value of the selection criterion; the scanned model with the lowest
  score is the one `gg_calc_fit` returns. It is `N*ln(RSS/N)` plus a penalty of
  `2` (`:aic`) or `ln(N)` (`:bic`) per fitted
  coefficient, where `RSS` is the pooled weighted sum of squared residuals and
  `N` the number of fitted field-component values. Only differences between
  candidates are meaningful — the absolute value is not. See the `fit_criterion`
  entry of `GGFitInputParams`.
- `criterion::Symbol` — which criterion `score` was computed with: `:bic` or
  `:aic`. Scores from different criteria are not comparable.
"""
@kwdef struct GGFitScanPoint
  m_max::Int = 0
  nd_max::Int = 0
  n_coef::Int = 0
  rms_weighted::Float64 = NaN
  rms_weighted_comp::NTuple{3,Float64} = (NaN, NaN, NaN)   # (Bx, By, Bs)
  rms_unweighted::Float64 = NaN
  score::Float64 = NaN
  criterion::Symbol = :bic
end

"""
    GGEvalPlan{VF,TWS,CPS,NG,NP,NQ}

Compiled, type-stable evaluation plan built once per `fit` (see low_level.jl).
Holds the interpolation `towers` (an `NTuple` of [`_Tower`](@ref)) and the
per-component monomial term lists `comps` (order `Bx By Bs  Ax Ay As  dAx dAy
dAs`).

The scratch-sizing constants are carried as `Val` **fields** (`ng = ngvals`,
`np = pmax+1`, `nq = qmax+1`) rather than plain integers so that (a) they are
compile-time constants inside the evaluator — sizing the stack-resident
`SVector`s with no heap allocation — and (b) they pass through `Adapt.adapt`
unchanged. Every array field is generic over its backing type, and the whole
plan is `Adapt.@adapt_structure`d, so `adapt(CuArray, plan)` (or whatever backend
`Adapt` targets) yields a plan whose evaluation runs inside a GPU kernel. The GG
value getters (`_interp_gvals`) and component evaluators (`_comp_value` /
`_comp_full`) are all allocation-free and generic over the coordinate type, so
the same plan tracks in `Float64`, `Float32`, or `ForwardDiff.Dual`.

## Fields

- `origin::NTuple{2,Float64}` — `(x, y)` line the GG expansion is written about.
- `z::VF` — base-plane positions [m], in increasing order.
- `towers::TWS` — the interpolation towers, an `NTuple{NT,_Tower}` (one per
  multipole `m` present, plus the single `bs` tower).
- `comps::CPS` — per-component monomial term lists, an `NTuple{9,_CompTerms}` in
  the order `Bx By Bs  Ax Ay As  dAx dAy dAs`.
- `ng::Val{NG}` — number of `gvals` slots, as a compile-time constant.
- `np::Val{NP}` — `pmax + 1`, the `x` power-table length.
- `nq::Val{NQ}` — `qmax + 1`, the `y` power-table length.
"""
struct GGEvalPlan{VF,TWS,CPS,NG,NP,NQ}
  origin::NTuple{2,Float64}
  z::VF
  towers::TWS                   # NTuple{NT,_Tower}
  comps::CPS                    # NTuple{9,_CompTerms}: Bx By Bs  Ax Ay As  dAx dAy dAs
  ng::Val{NG}                   # ngvals            (gvals length)
  np::Val{NP}                   # pmax + 1          (x power-table length)
  nq::Val{NQ}                   # qmax + 1          (y power-table length)
end

Adapt.@adapt_structure GGEvalPlan

#---------------------------------------------------------------------------------------------------
# GGFit

"""
    mutable struct GGFit

Holds the result of a `gg_calc_fit` fit: the fitted generalized-gradient (GG)
coefficient functions sampled at the base planes plus per-plane diagnostics.
Returned by `gg_calc_fit` and consumed by `gg_show_fit_results` and
`write_gg_fit`.

## Fields

- `z_base::Vector{Float64}` — `z` position of each base plane [m].
- `params::Vector{Tuple{Symbol,Int,Int}}` — list of fitted unknowns as
  `(type, m, nd)` tuples, where `type` is one of `:a`, `:b`, `:bs` (`bs` uses
  `m = 0`).
- `a::Dict{Tuple{Int,Int},Vector{Float64}}` — fitted `a(m,nd)` functions,
  `(m,nd) => values_over_planes`.
- `b::Dict{Tuple{Int,Int},Vector{Float64}}` — fitted `b(m,nd)` functions,
  `(m,nd) => values_over_planes`.
- `bs::Dict{Int,Vector{Float64}}` — fitted `bs(nd)` functions,
  `nd => values_over_planes`.
- `rms_weighted_plane::Vector{Float64}` — weighted RMS fit residual at each base
  plane, `sqrt(Σ w·δ² / Σ w)` over the points of that plane's fit region.
- `rms_unweighted_plane::Vector{Float64}` — RMS fit residual at each base plane
  over the same points as `rms_weighted_plane` but with all point weights set to
  1. Equal to `rms_weighted_plane` when `core_weight = outer_plane_weight = 1`.
- `field_ave_plane::Vector{Float64}` — average field magnitude `|B|` over the
  fitted grid points of each base plane [T]. Unweighted, and taken from the base
  plane alone (not the added planes), so it gives the field profile along `z` and
  a scale against which `rms_weighted_plane` can be judged.
- `fit_radius_max::Float64` — radius about `origin` the fit was restricted to
  [m], or `0` if every grid point was used. Recorded because the residuals and
  the field contributions above cover that region only, so reading them, or
  measuring anything else against the fit, needs to know where the fit applies.
- `m_max::Int` — highest multipole order `m` retained by the fit. Lower than the
  cutoff the scan chose if pruning removed every function at the top orders.
- `nd_max::Int` — highest derivative order `nd` retained by the fit.
- `scan::Vector{GGFitScanPoint}` — one `GGFitScanPoint` per `(m_max, nd_max)`
  combination tried, in the order tried. Empty when no scan was requested.
- `pruned::Vector{Tuple{Symbol,Int}}` — GG functions dropped for producing
  negligible field, as `(:a, m)` / `(:b, m)` / `(:bs, 0)` tuples. Empty unless
  `prune_ave_limit` or `prune_max_limit` was set. A pruned function is absent
  from `a`/`b`/`bs` entirely — the dicts are sparse in `m`, and every consumer
  treats a missing key as an identically zero function. Not stored in the HDF5
  file: which functions are present is already evident from the keys that were
  written.
- `g_ref::Float64` — reference-frame bending strength = `1/bending_radius`, in
  1/m (`0` for a straight reference frame).
- `origin::Vector{Float64}` — `(x, y)` line about which the GG coefficients are
  computed.
- `dz_grid::Float64` — spacing between base planes [m].
- `eval_plan::Union{Nothing,GGEvalPlan}` — internal cache: the compiled
  `GGEvalPlan` used by `field_and_potential_evaluate_at`, built lazily on first
  evaluation. Not part of the fit data (not serialized); assumes the other fields
  are not mutated afterward.
"""
@kwdef mutable struct GGFit
  z_base::Vector{Float64} = Float64[]
  params::Vector{Tuple{Symbol,Int,Int}} = Tuple{Symbol,Int,Int}[]
  a::Dict{Tuple{Int,Int},Vector{Float64}} = Dict{Tuple{Int,Int},Vector{Float64}}()
  b::Dict{Tuple{Int,Int},Vector{Float64}} = Dict{Tuple{Int,Int},Vector{Float64}}()
  bs::Dict{Int,Vector{Float64}} = Dict{Int,Vector{Float64}}()
  rms_weighted_plane::Vector{Float64} = Float64[]
  rms_unweighted_plane::Vector{Float64} = Float64[]
  field_ave_plane::Vector{Float64} = Float64[]
  fit_radius_max::Float64 = 0.0          # radius the fit was restricted to [m]. 0 = every grid point.
  m_max::Int = 0
  nd_max::Int = 0
  scan::Vector{GGFitScanPoint} = GGFitScanPoint[]
  pruned::Vector{Tuple{Symbol,Int}} = Tuple{Symbol,Int}[]
  g_ref::Float64 = 0.0
  origin::Vector{Float64} = [0.0, 0.0]   # (x, y) origin about which the generalized gradients coefs are computed
  dz_grid::Float64 = 0.0                 # spacing between base planes [m]
  eval_plan::Union{Nothing,GGEvalPlan} = nothing  # lazily built fast-eval plan (internal cache)
end

#---------------------------------------------------------------------------------------------------
# _CompTerms{VI,VF}

# Compiled fast-evaluation plan for `field_and_potential_evaluate_at`. Built once
# per `GGFit` and cached in its `eval_plan` field; the build and per-call
# evaluation live in low_level.jl.

"""
    _CompTerms{VI,VF}

One output component's monomial terms. Evaluating the component accumulates
`value += Σ w[t] * gvals[slot[t]] * x^p[t] * y^q[t]` over all terms `t`.

Generic over its backing-array types (`VI` for the integer arrays, `VF` for the
weights) so the same struct is `Vector`-backed on the host and device-array
backed after `Adapt.adapt` — see [`GGEvalPlan`](@ref).

## Fields

- `slot::VI` — index into `gvals` of the GG value each term multiplies.
- `w::VF` — coefficient of each term.
- `p::VI` — power of `x` of each term.
- `q::VI` — power of `y` of each term.
"""
struct _CompTerms{VI,VF}
  slot::VI
  w::VF
  p::VI
  q::VI
end

Base.length(ct::_CompTerms) = length(ct.slot)
Adapt.@adapt_structure _CompTerms

#---------------------------------------------------------------------------------------------------
# _Tower{VI,VF,MF}

"""
    _Tower{VI,VF,MF}

One GG derivative tower (a fixed multipole `m`, or the single `bs` tower).
`poly[d+1, pair]` is the coefficient of `u^d` (with `u = s - zref[pair]`) of the
interpolant on plane-pair `pair`; interpolating gives `H⁽ⁿᵈ⁾(s)` for the tower's
orders `nd = 0..N`, scattered into `gvals` at `slots[nd+1]`. Non-contiguous
orders (`nd > N`) are taken from the nearest (left) plane via `extra_vals[e, pair]`.

Generic over its backing-array types (`VI`/`VF`/`MF` for the integer vectors,
float vectors and float matrices) so it survives `Adapt.adapt` to the GPU.

## Fields

- `N::Int` — highest contiguous derivative order the tower interpolates.
- `deg::Int` — polynomial degree of the interpolant: `2N+1` for Hermite, `N` for
  Taylor.
- `slots::VI` — `gvals` slot for each order `nd = 0..N`.
- `poly::MF` — interpolant coefficients, `(deg+1) x npairs`.
- `zref::VF` — left-plane position of each plane-pair [m], length `npairs`.
- `extra_slots::VI` — `gvals` slots for the non-contiguous orders `nd > N` (rare).
- `extra_vals::MF` — value of each extra order at each plane, `n_extra x P`.
"""
struct _Tower{VI,VF,MF}
  N::Int
  deg::Int                          # polynomial degree (2N+1 Hermite, or N Taylor)
  slots::VI                         # gvals slot for order 0..N
  poly::MF                          # (deg+1) x npairs
  zref::VF                          # left-plane position per pair (length npairs)
  extra_slots::VI                   # non-contiguous orders nd > N (rare)
  extra_vals::MF                    # (n_extra x P) value of each extra order per plane
end

Adapt.@adapt_structure _Tower

