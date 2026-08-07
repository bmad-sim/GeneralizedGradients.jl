@enumx GridAnchorPt Beginning Center End
@enumx GridGeometry XYZ 

"""
    mutable struct FieldGridTable{T}

Holds an electric and/or magnetic field sampled on a 3D grid.

`magnetic` and `electric` are 3D `OffsetArray`s whose elements are field
3-vectors: `magnetic[ix,iy,iz] == [Bx, By, Bz]` (and likewise `[Ex, Ey, Ez]`).
The grid indices `(ix, iy, iz)` need not start at 0 or 1; a grid point is at
position `r0 + dr .* (ix, iy, iz)` relative to the anchor.

Fields:
- `magnetic` — magnetic field 3-vectors `[Bx, By, Bz]` [T].
- `electric` — electric field 3-vectors `[Ex, Ey, Ez]` [V/m].
- `r0` — grid origin offset `(x0, y0, z0)` [m].
- `dr` — grid spacing `(dx, dy, dz)` [m].
- `g_ref` — curvilinear-coordinate bending strength `1/bending_radius`, in 1/m
  (`0` for a straight reference curve).
- `scale` — overall field scale factor.
- `RF_frequency` — RF frequency in Hz (`0` for a static field).
- `RF_phase` — RF phase [rad].
- `anchor_pt` — grid anchor point, a `GridAnchorPt.T` (`Beginning`, `Center`, or `End`).
- `geometry` — grid geometry, a `GridGeometry.T` (`XYZ`).

`FieldGridTable()` builds an empty table with `T = Float64`; read one from a
file with `read_field_grid_hdf5`.
"""
@kwdef mutable struct FieldGridTable{T}
  # `magnetic`/`electric` are 3D grids whose elements are field 3-vectors:
  # `magnetic[ix,iy,iz] == [Bx, By, Bz]`.  Defaults use concrete Float64 (not `T`)
  # so that the parameterless `FieldGridTable()` works: it infers T = Float64.
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

@kwdef mutable struct GGTerm{T}
  coef::T = 0.0
  expn::Vector{Int} = [0,0]
end


@kwdef mutable struct GGTaylor{T}
  term::Vector{GGTerm{T}} = [GGTerm{T}()]
end

# GGTaylor() = GGTaylor{Float64}()

#---------------------------------------------------------------------------------------------------
"""
    origin = [x0, y0]

Defines the line [x0, y0, z] about which the generalized gradient coefficients are computed.
If h is non-zero, origin must be [0, 0].

    n_planes_add = Int

This parameter sets the number of z-planes added to either side of the base z-plane to
be used in the analysis of the derivatives at any given base z-plane (see "How the GG Calculation
Works" section). For example, for n_planes_add = 2, two planes would be added to either side of the
base plane making the total number of planes used in the analysis equal to five.

  core_weight = Float

Merit function weight for "core" points (field table points whose transverse (x,y)
position is near (0,0)). Default is 1.0 which gives an equal weight for all points of a given
z-plane. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  outer_plane_weight = Float

Merit function weight for z-planes away from the base z-plane when n_planes_add
is non-zero. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  m_max = Int or vector of Int

Maximum multipole order `m` of the `a_m`/`b_m` functions retained in the fit. A
single `Int` fixes it. A vector or range (for example `m_max = 1:8`) makes
`gg_fit` try each value in turn and keep the one that fits best (see
`fit_criterion`). The default `-1` means "use every `m` present in the
coefficient table", which is the historical behavior. The `bs(nd)` (that is,
`a_0` derivative) unknowns carry no `m` and are never removed by `m_max`.

  nd_max = Int or vector of Int

Maximum derivative order `nd` retained in the fit, with the same `Int`-or-vector
meaning as `m_max`. The default `-1` means `2 * n_planes_add`, the number of
derivative orders a stencil of `2 * n_planes_add + 1` planes resolves by
longitudinal differencing alone. Higher values are legitimate — the transverse
structure of the field also carries derivative information — but become
progressively worse conditioned, which is what the scan is for.

`m_max` and `nd_max` apply to every base plane; they never vary plane to plane.
If either is given as a vector, `gg_fit` scans the full grid of combinations and
records the outcome of each in the `scan` field of the returned `GGCoefs`.

  fit_criterion = Symbol

How a scan picks its winner. Each candidate model is given a score and the lowest
score wins. Enlarging the model can only lower the residual (the smaller model's
unknowns are a subset of the larger one's), so a usable criterion has to charge
for coefficients. Writing `RSS` for the weighted sum of squared residuals pooled
over all base planes, `N` for the number of fitted field-component values, `W`
for the sum of the weights of those values (`W = N` for uniform weighting), and
`k` for the total number of fitted coefficients (per-plane count times the number
of base planes):

```
:rms    score = sqrt(RSS / W)          # goodness of fit only, no charge for k
:aic    score = N * ln(RSS/N) + 2 * k         # Akaike information criterion
:bic    score = N * ln(RSS/N) + k * ln(N)     # Bayesian, and the default
```

`:aic` and `:bic` share the same first term — minus twice the maximized Gaussian
log-likelihood, up to a constant common to all candidates — and differ only in
what one coefficient costs: `2` versus `ln(N)`. `ln(N) > 2` for any `N > 7`, so
`:bic` always penalizes size at least as hard as `:aic` and never selects a
larger model. `:rms` charges nothing and so will pick the largest model offered;
it is useful for inspecting the raw residual ranking, not for choosing a model.

See the "Choosing between models" section of the `gg_fit` docstring for how to
read the trade-off quantitatively and for why, on a field grid, these criteria
are better treated as a ranking heuristic than as a probability statement.

  prune_ave_limit = Float64
  prune_max_limit = Float64

Drop GG functions that produce negligible field, so they are neither fitted nor
stored. A function here is a whole `a_m`, a whole `b_m`, or `b_s` — all of its
derivative orders `nd` together — and its contribution is the `|B|` it alone
produces, every other GG coefficient set to zero, measured over every transverse
grid point of every base plane.

Both limits are fractions of the field table's mean `|B|`, so they carry over
unchanged between magnets of different strength. `prune_ave_limit` is compared
against the function's average contribution and `prune_max_limit` against its
largest. A function is dropped only when it falls below **every** limit that is
in force; a limit of `0` (the default for both) switches that test off, and with
both off no pruning is done at all.

```
prune_ave_limit = 1e-4      # drop if the average contribution is under 0.01% of <|B|>
prune_max_limit = 1e-3      # ... and the largest contribution is under 0.1% of <|B|>
```

Setting only `prune_max_limit` is the conservative choice: a function survives if
it matters anywhere on the grid, even if it averages to little. Setting only
`prune_ave_limit` prunes harder, and can discard a function that is small on
average but significant near the aperture, where the max lives.

Pruning is applied after a scan has picked its `(m_max, nd_max)` winner, and the
surviving functions are then **refit**. The stored coefficients are therefore the
least-squares solution of the model that was kept, not the leftovers of a larger
fit. The functions removed are listed in the `pruned` field of the result.

  output_file

Name of the output file.
"""
@kwdef mutable struct GGFitInputParams
  origin::Vector{Float64} = [0.0, 0.0]   # (x, y) origin about which the generalized gradients coefs are computed
  n_planes_add::Int = 1                  # Number of z-planes added.
  m_max::Union{Int,AbstractVector{Int}} = -1   # Max multipole order m. -1 = all in table. Vector => scan.
  nd_max::Union{Int,AbstractVector{Int}} = -1  # Max derivative order nd. -1 = 2*n_planes_add. Vector => scan.
  fit_criterion::Symbol = :bic           # Scan selection criterion: :bic, :aic, or :rms.
  core_weight::Float64 = 1.0             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
  outer_plane_weight::Float64 = 1.0      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
  prune_ave_limit::Float64 = 0.0         # Drop a GG function whose ave contribution is below this fraction of <|B|>. 0 = test off.
  prune_max_limit::Float64 = 0.0         # Drop a GG function whose max contribution is below this fraction of <|B|>. 0 = test off.
  output_file::String = "gg_fit_results.h5"
end

#---------------------------------------------------------------------------------------------------
"""
    struct GGFitScanPoint

One row of a `gg_fit` `(m_max, nd_max)` scan: the model tried and how it scored.
Collected in the `scan` field of the returned `GGCoefs` and printed by
`gg_fit_show_results`.

Fields:
- `m_max`, `nd_max` — the model this row is for.
- `n_coef` — number of fitted coefficients per base plane.
- `rms_weighted` — weighted RMS residual pooled over all base planes,
  `sqrt(Σ w·δ² / Σ w)` over every fitted field-component value.
- `rms_weighted_comp` — the same quantity restricted to one field component at a
  time, as the 3-tuple `(Bx, By, Bs)`. Each entry is normalized by that
  component's own weight sum, so the three are directly comparable to each other
  and to `rms_weighted`, which is their weighted quadrature mean. A fit that is
  bad in only one component shows up here and nowhere else.
- `rms_unweighted` — the same residual with all point weights set to 1.
- `score` — value of the selection criterion; the scanned model with the lowest
  score is the one `gg_fit` returns. For `:rms` this is just `rms_weighted`. For `:aic`
  and `:bic` it is `N*ln(RSS/N)` plus a penalty of `2` or `ln(N)` per fitted
  coefficient, where `RSS` is the pooled weighted sum of squared residuals and
  `N` the number of fitted field-component values. Only differences between
  candidates are meaningful — the absolute value is not. See the `fit_criterion`
  entry of `GGFitInputParams`.
- `criterion` — which criterion `score` was computed with: `:bic`, `:aic`, or
  `:rms`. Scores from different criteria are not comparable.
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

#---------------------------------------------------------------------------------------------------
# Compiled fast-evaluation plan for `field_and_potential_evaluate_at`. Built once
# per `GGCoefs` and cached in its `eval_plan` field; the build and per-call
# evaluation live in low_level.jl.

"""
    _CompTerms{VI,VF}

One output component's monomial terms. Evaluating the component accumulates
`value += Σ w[t] * gvals[slot[t]] * x^p[t] * y^q[t]` over all terms `t`.

Generic over its backing-array types (`VI` for the integer arrays, `VF` for the
weights) so the same struct is `Vector`-backed on the host and device-array
backed after `Adapt.adapt` — see [`GGEvalPlan`](@ref).
"""
struct _CompTerms{VI,VF}
  slot::VI
  w::VF
  p::VI
  q::VI
end

Base.length(ct::_CompTerms) = length(ct.slot)
Adapt.@adapt_structure _CompTerms

"""
    _Tower{VI,VF,MF}

One GG derivative tower (a fixed multipole `m`, or the single `bs` tower).
`poly[d+1, pair]` is the coefficient of `u^d` (with `u = s - zref[pair]`) of the
interpolant on plane-pair `pair`; interpolating gives `H⁽ⁿᵈ⁾(s)` for the tower's
orders `nd = 0..N`, scattered into `gvals` at `slots[nd+1]`. Non-contiguous
orders (`nd > N`) are taken from the nearest (left) plane via `extra_vals[e, pair]`.

Generic over its backing-array types (`VI`/`VF`/`MF` for the integer vectors,
float vectors and float matrices) so it survives `Adapt.adapt` to the GPU.
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
"""
    mutable struct GGCoefs

Holds the result of a `gg_fit` fit: the fitted generalized-gradient (GG)
coefficient functions sampled at the base planes plus per-plane diagnostics.
Returned by `gg_fit` and consumed by `gg_fit_show_results` and
`write_gg_fit`.

Fields:
- `z_base` — `z` position of each base plane [m].
- `params` — list of fitted unknowns as `(type, m, nd)` tuples, where `type` is
  one of `:a`, `:b`, `:bs` (`bs` uses `m = 0`).
- `a` — fitted `a(m,nd)` functions, `Dict (m,nd) => values_over_planes`.
- `b` — fitted `b(m,nd)` functions, `Dict (m,nd) => values_over_planes`.
- `bs` — fitted `bs(nd)` functions, `Dict nd => values_over_planes`.
- `rms_weighted_plane` — weighted RMS fit residual at each base plane,
  `sqrt(Σ w·δ² / Σ w)` over the points of that plane's fit region.
- `rms_unweighted_plane` — RMS fit residual at each base plane over the same
  points as `rms_weighted_plane` but with all point weights set to 1. Equal to
  `rms_weighted_plane` when `core_weight = outer_plane_weight = 1`.
- `field_ave_plane` — average field magnitude `|B|` over the grid points of each
  base plane [T]. Unweighted, and taken from the base plane alone (not the added
  planes), so it gives the field profile along `z` and a scale against which
  `rms_weighted_plane` can be judged.
- `m_max` — highest multipole order `m` retained by the fit. Lower than the
  cutoff the scan chose if pruning removed every function at the top orders.
- `nd_max` — highest derivative order `nd` retained by the fit.
- `scan` — one `GGFitScanPoint` per `(m_max, nd_max)` combination tried, in the
  order tried. Empty when no scan was requested.
- `pruned` — GG functions dropped for producing negligible field, as
  `(:a, m)` / `(:b, m)` / `(:bs, 0)` tuples. Empty unless `prune_ave_limit` or
  `prune_max_limit` was set. A pruned function is absent from `a`/`b`/`bs`
  entirely — the dicts are sparse in `m`, and every consumer treats a missing key
  as an identically zero function. Not stored in the HDF5 file: which functions
  are present is already evident from the keys that were written.
- `g_ref` — reference-frame bending strength = `1/bending_radius` [1/m] (`0` for a
  straight reference frame).
- `origin` — `(x, y)` line about which the GG coefficients are computed.
- `dz_grid` — spacing between base planes [m].
- `eval_plan` — internal cache: the compiled `GGEvalPlan` used by
  `field_and_potential_evaluate_at`, built lazily on first evaluation. Not part
  of the fit data (not serialized); assumes the other fields are not mutated
  afterward.
"""
@kwdef mutable struct GGCoefs
  z_base::Vector{Float64} = Float64[]
  params::Vector{Tuple{Symbol,Int,Int}} = Tuple{Symbol,Int,Int}[]
  a::Dict{Tuple{Int,Int},Vector{Float64}} = Dict{Tuple{Int,Int},Vector{Float64}}()
  b::Dict{Tuple{Int,Int},Vector{Float64}} = Dict{Tuple{Int,Int},Vector{Float64}}()
  bs::Dict{Int,Vector{Float64}} = Dict{Int,Vector{Float64}}()
  rms_weighted_plane::Vector{Float64} = Float64[]
  rms_unweighted_plane::Vector{Float64} = Float64[]
  field_ave_plane::Vector{Float64} = Float64[]
  m_max::Int = 0
  nd_max::Int = 0
  scan::Vector{GGFitScanPoint} = GGFitScanPoint[]
  pruned::Vector{Tuple{Symbol,Int}} = Tuple{Symbol,Int}[]
  g_ref::Float64 = 0.0
  origin::Vector{Float64} = [0.0, 0.0]   # (x, y) origin about which the generalized gradients coefs are computed
  dz_grid::Float64 = 0.0                 # spacing between base planes [m]
  eval_plan::Union{Nothing,GGEvalPlan} = nothing  # lazily built fast-eval plan (internal cache)
end