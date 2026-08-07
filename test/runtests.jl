using GeneralizedGradients
using Test
using OffsetArrays
using HDF5
using KernelAbstractions
using Adapt: adapt

# Self-contained test fixture (HDF5 gg_fit result).
const EXAMPLE = joinpath(@__DIR__, "data", "gg_fit_result.h5")

# Run `f` with stdout suppressed (the writer/converter functions print a banner).
quiet(f) = redirect_stdout(f, devnull)

# Build a synthetic FieldGridTable with a smooth analytic field over an offset
# (ix, iy, iz) grid. `magnetic[ix,iy,iz]` is the [Bx,By,Bz] 3-vector at the point
# (x, y, z) = r0 + dr .* (ix, iy, iz). The on-axis Bz varies in z so the fit
# produces a non-zero solenoid (b_s) term.
function make_field(; g_ref = 0.0, ixr = -3:3, iyr = -3:3, izr = 0:5,
                      r0 = [0.0, 0.0, 0.0], dr = [0.01, 0.01, 0.005])
  nx, ny, nz = length(ixr), length(iyr), length(izr)
  data = Array{Vector{Float64}}(undef, nx, ny, nz)
  for (a, ix) in enumerate(ixr), (b, iy) in enumerate(iyr), (c, iz) in enumerate(izr)
    x = r0[1] + dr[1] * ix
    y = r0[2] + dr[2] * iy
    z = r0[3] + dr[3] * iz
    data[a, b, c] = [ 0.10 + 0.20x - 0.30y + 0.05x * z,
                     -0.10 + 0.40y + 0.10x - 0.02y * z,
                      0.30 + 0.05x - 0.04y + 0.01z]
  end
  return FieldGridTable{Float64}(; magnetic = OffsetArray(data, ixr, iyr, izr),
      r0 = collect(float.(r0)), dr = collect(float.(dr)), g_ref = float(g_ref))
end

# Frenet–Serret curl of the vector potential.  Given A and its Jacobian
# dA[i,j] = ∂A_i/∂u_j with u = (x, y, s), reconstruct B; this must equal the
# B returned by the evaluator for ANY coefficient values (constant curvature g_ref).
function curl_from(A, dA, x, g_ref)
  g = 1 + g_ref * x
  Bx = dA[3, 2] - dA[2, 3] / g
  By = dA[1, 3] / g - (g_ref * A[3] + g * dA[3, 1]) / g
  Bs = dA[2, 1] - dA[1, 2]
  return [Bx, By, Bs]
end

# Build a synthetic (fit, meta) pair (same shape as read_gg_fit returns) so we
# can exercise finite curvature g_ref, which the example file (g_ref = 0) does not.
synth(z_base, a, b, bs, g_ref; nd_max, dz_grid) = (
  GGCoefs(; z_base = collect(float.(z_base)), a, b, bs,
                 nd_max, rms_weighted_plane = fill(NaN, length(z_base)), g_ref,
                 origin = [0.0, 0.0], dz_grid),
  (;))

const PTS = ((0.004, 0.003), (-0.005, 0.002), (0.003, -0.004), (0.0, 0.006), (0.007, 0.0))

# Top-level (type-stable) sweep used for the per-particle allocation check: no
# captured variables, so any allocation reported is the evaluator's own.
@noinline function _gg_sweep(plan, xs, ys, ss)
  A1, dA1 = potential_evaluate_at(plan, xs[1], ys[1], ss[1])
  acc = A1[1] + dA1[1, 1]
  @inbounds for i in 2:length(xs)
    A, dA = potential_evaluate_at(plan, xs[i], ys[i], ss[i])
    acc += A[1] + dA[1, 1]
  end
  return acc
end

# KernelAbstractions kernel: evaluate the (Adapt-ed) plan per particle. Defined at
# top level so `@kernel` expands in module scope.
@kernel function _gg_pot_kernel!(Aout, dAout, plan, xs, ys, ss)
  i = @index(Global, Linear)
  A, dA = potential_evaluate_at(plan, xs[i], ys[i], ss[i])
  @inbounds for k in 1:3
    Aout[k, i] = A[k]
  end
  @inbounds for c in 1:3, r in 1:3
    dAout[r, c, i] = dA[r, c]
  end
end

@testset "GeneralizedGradients" begin

  @testset "read_gg_fit" begin
    fit, meta = read_gg_fit(EXAMPLE)
    @test fit isa GGCoefs
    for k in (:z_base, :a, :b, :bs, :m_max, :nd_max, :rms_weighted_plane, :g_ref, :origin, :dz_grid)
      @test hasproperty(fit, k)
    end
    @test length(fit.z_base) == length(fit.rms_weighted_plane)
    @test fit.a isa Dict && fit.b isa Dict && fit.bs isa Dict
  end

  @testset "curl(A) == B at grid planes (example, g_ref=0)" begin
    fit, meta = read_gg_fit(EXAMPLE)
    for ip in 1:length(fit.z_base), (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(fit, ip, x, y)
      Bc = curl_from(A, dA, x - fit.origin[1], fit.g_ref)
      @test maximum(abs, B .- Bc) < 1e-12
    end
  end

  @testset "curl(A) == B, synthetic single plane (g_ref=0.6)" begin
    a  = Dict((1,0)=>0.7,(2,0)=>-0.4,(3,0)=>0.25,(1,1)=>0.3,(2,1)=>-0.15,(1,2)=>0.2)
    b  = Dict((1,0)=>0.5,(2,0)=>0.35,(3,0)=>-0.2,(1,1)=>0.1,(2,1)=>0.05)
    bs = Dict(0=>0.45, 1=>-0.12, 2=>0.08)
    va = Dict(k => [v] for (k, v) in a)
    vb = Dict(k => [v] for (k, v) in b)
    vbs = Dict(k => [v] for (k, v) in bs)
    fit, meta = synth([0.0], va, vb, vbs, 0.6; nd_max = 2, dz_grid = 0.1)
    for (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(fit, 1, x, y)
      Bc = curl_from(A, dA, x, fit.g_ref)
      @test maximum(abs, B .- Bc) < 1e-12
    end
  end

  # Physically consistent multi-plane result: each GG function is a known
  # polynomial in s, sampled with its true derivative tower at every plane.
  fa1(s) = 0.6 + 1.0s - 0.4s^2 + 0.2s^3
  fa2(s) = -0.3 + 0.5s + 0.25s^2
  fb1(s) = 0.4 - 0.7s + 0.3s^2
  fb2(s) = 0.2 + 0.15s - 0.35s^2
  fbs(s) = 0.07 + 0.5s - 0.2s^2 + 0.1s^3
  d1(f, s; δ=1e-6) = (f(s+δ) - f(s-δ)) / (2δ)
  d2(f, s; δ=1e-4) = (f(s+δ) - 2f(s) + f(s-δ)) / δ^2
  d3(f, s; δ=1e-3) = (f(s+2δ) - 2f(s+δ) + 2f(s-δ) - f(s-2δ)) / (2δ^3)
  mvec(f, s) = [f(s), d1(f, s), d2(f, s), d3(f, s)]
  zg = collect(0.0:0.05:0.5)
  amap = Dict((1,m) => [mvec(fa1, s)[m+1] for s in zg] for m in 0:3)
  merge!(amap, Dict((2,m) => [mvec(fa2, s)[m+1] for s in zg] for m in 0:2))
  bmap = Dict((1,m) => [mvec(fb1, s)[m+1] for s in zg] for m in 0:2)
  merge!(bmap, Dict((2,m) => [mvec(fb2, s)[m+1] for s in zg] for m in 0:2))
  bsmap = Dict(m => [mvec(fbs, s)[m+1] for s in zg] for m in 0:3)
  fitM, metaM = synth(zg, amap, bmap, bsmap, 0.5; nd_max = 3, dz_grid = 0.05)

  @testset "curl(A) == B, multi-plane (g_ref=0.5)" begin
    for ip in 1:length(zg), (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(fitM, ip, x, y)
      Bc = curl_from(A, dA, x, fitM.g_ref)
      @test maximum(abs, B .- Bc) < 1e-12
    end
  end

  @testset "∂A/∂s matches finite difference" begin
    ip = 6; sc = zg[ip]; xq, yq = 0.006, -0.004
    _, _, dA = field_and_potential_evaluate(fitM, ip, xq, yq)
    δ = 1e-5
    planM = eval_plan(fitM)
    _, Ap, _ = field_and_potential_evaluate_at(planM, xq, yq, sc + δ)
    _, Am, _ = field_and_potential_evaluate_at(planM, xq, yq, sc - δ)
    dAs_fd = (Ap .- Am) ./ (2δ)
    @test maximum(abs, dA[:, 3] .- dAs_fd) < 1e-6
  end

  @testset "gg_coefficients_at_plane matches direct indexing" begin
    fit, meta = read_gg_fit(EXAMPLE)
    ip = 3
    a, b, bs = gg_coefficients_at_plane(fit, ip)
    @test a isa Dict{Tuple{Int,Int},Float64}
    @test bs isa Dict{Int,Float64}
    for (nm, v) in fit.a; @test a[nm] == v[ip]; end
    for (nm, v) in fit.b; @test b[nm] == v[ip]; end
    for (m, v) in fit.bs; @test bs[m] == v[ip]; end
  end

  @testset "_at_s reproduces plane values at a grid plane" begin
    fit, meta = read_gg_fit(EXAMPLE)
    ip = 4; s = fit.z_base[ip]

    a, b, bs = gg_coefficients_at_plane(fit, ip)
    as, bsd, bss = gg_coefficients_at_s(fit, s)
    @test as == a && bsd == b && bss == bs

    CBx, CBy, CBs = field_coefficients_at_plane(fit, ip)
    CBxs, CBys, CBss = field_coefficients_at_s(fit, s)
    @test CBxs == CBx && CBys == CBy && CBss == CBs

    B, A, dA = field_and_potential_evaluate(fit, ip, 0.004, 0.003)
    Bs, As, dAs = field_and_potential_evaluate_at(eval_plan(fit), 0.004, 0.003, s)
    @test Bs ≈ B && As ≈ A && dAs ≈ dA
  end

  @testset "fast evaluate_at matches reference path" begin
    # The compiled plan (low_level.jl) must reproduce the generic reference
    # composition field_and_potential_evaluate(_interp_gg_fit(...)). And
    # potential_evaluate_at must return exactly the A/dA of the full evaluator.
    fit, _ = read_gg_fit(EXAMPLE)
    reference(f, x, y, s) =
      field_and_potential_evaluate(GeneralizedGradients._interp_gg_fit(f, s), 1, x, y)
    approx(P, Q) = maximum(abs, P .- Q) ≤ 1e-9 * max(1.0, maximum(abs, Q))
    for f in (fit, fitM)
      plan = eval_plan(f)
      zlo, zhi = first(f.z_base), last(f.z_base)
      ss = (zlo, zhi, (zlo+zhi)/2, zlo + 0.37(zhi-zlo),
            f.z_base[min(3, length(f.z_base))], zlo - 0.01, zhi + 0.01)  # incl. extrapolation
      for (x, y) in PTS, s in ss
        Bf, Af, dAf = field_and_potential_evaluate_at(plan, x, y, s)
        Br, Ar, dAr = reference(f, x, y, s)
        @test approx(Bf, Br) && approx(Af, Ar) && approx(dAf, dAr)
        Ap, dAp = potential_evaluate_at(plan, x, y, s)
        @test Ap == Af && dAp == dAf     # identical: same code path minus B
        Bp = field_evaluate_at(plan, x, y, s)
        @test Bp == Bf                   # identical: same code path, B only
      end
    end
  end

  @testset "plan evaluator is GPU-capable (Adapt + KA kernel, generic T)" begin
    # The compiled GGEvalPlan is Adapt-able and its evaluator is allocation-free
    # and generic over the coordinate type, so tracking with a GG fit can run on
    # the GPU. We check that here against a real fit, on the portable KA CPU
    # backend (which exercises the same code path a GPU backend would compile):
    #   * Adapt round-trips the plan and gives identical results,
    #   * potential_evaluate_at runs inside a KA kernel and equals the host,
    #   * for both Float64 and Float32 inputs,
    #   * with zero per-particle heap allocation.
    fit, _ = read_gg_fit(EXAMPLE)
    plan = eval_plan(fit)
    zlo, zhi = first(fit.z_base), last(fit.z_base)
    backend = CPU()

    # Adapt round-trip: adapt(Array, plan) reproduces the host result bit-for-bit.
    aplan = adapt(Array, plan)
    for (x, y) in PTS
      s = (zlo + zhi) / 2
      @test potential_evaluate_at(aplan, x, y, s) == potential_evaluate_at(plan, x, y, s)
    end

    for T in (Float64, Float32)
      pts = [(T(x), T(y), T(zlo + f * (zhi - zlo))) for (x, y) in PTS for f in range(0, 1; length = 5)]
      N  = length(pts)
      xs = T[p[1] for p in pts]; ys = T[p[2] for p in pts]; ss = T[p[3] for p in pts]

      Aref  = zeros(T, 3, N)
      dAref = zeros(T, 3, 3, N)
      for i in 1:N
        A, dA = potential_evaluate_at(plan, xs[i], ys[i], ss[i])
        Aref[:, i]     .= A
        dAref[:, :, i] .= dA
      end

      Aout  = KernelAbstractions.zeros(backend, T, 3, N)
      dAout = KernelAbstractions.zeros(backend, T, 3, 3, N)
      _gg_pot_kernel!(backend)(Aout, dAout, aplan, xs, ys, ss; ndrange = N)
      KernelAbstractions.synchronize(backend)

      @test Array(Aout)  == Aref        # exact: kernel runs the same code as host
      @test Array(dAout) == dAref
    end

    # Zero per-particle heap: total allocation must not grow with the particle
    # count (a real per-call heap of even 8 bytes would add ~144 KB over the gap).
    mk(n) = (collect(range(-0.005, 0.005; length = n)),
             collect(range(-0.004, 0.004; length = n)),
             collect(range(zlo, zhi; length = n)))
    xa, ya, sa = mk(2_000); xb, yb, sb = mk(20_000)
    _gg_sweep(plan, xa, ya, sa)                       # compile
    a1 = @allocated _gg_sweep(plan, xa, ya, sa)
    a2 = @allocated _gg_sweep(plan, xb, yb, sb)
    @test abs(a2 - a1) < 4096
  end

  @testset "gg_fit + show + write round-trip" begin
    field = make_field()
    p = GGFitInputParams()
    p.n_planes_add = 1
    res = gg_fit(field, p)
    @test res isa GGCoefs
    @test length(res.z_base) == size(field.magnetic, 3)
    @test res.nd_max == 2
    @test length(res.rms_weighted_plane) == length(res.z_base)
    @test all(isfinite, res.rms_weighted_plane)
    @test !isempty(res.params)
    quiet(() -> gg_fit_show_results(res, field, p))

    mktempdir() do dir
      p.output_file = joinpath(dir, "fit.h5")
      out = quiet(() -> write_gg_fit(res, field, p))
      @test out == p.output_file && isfile(out)
      fit, meta = read_gg_fit(out)
      @test fit.m_max == res.m_max
      @test fit.nd_max == res.nd_max
      @test fit.dz_grid ≈ field.dr[3]
      @test length(fit.z_base) == length(res.z_base)
      @test Set(keys(fit.a)) == Set(keys(res.a))
      @test Set(keys(fit.bs)) == Set(keys(res.bs))
      for (k, v) in res.a; @test fit.a[k] ≈ v; end
      for (k, v) in res.b; @test fit.b[k] ≈ v; end
      for (k, v) in res.bs; @test fit.bs[k] ≈ v; end
    end
  end

  @testset "gg_fit weighting and n_planes_add=0 branches" begin
    field = make_field()
    # Non-default core/outer weights exercise the weighting branches.
    p = GGFitInputParams()
    p.n_planes_add = 1
    p.core_weight = 2
    p.outer_plane_weight = 2
    res = gg_fit(field, p)
    @test all(isfinite, res.rms_weighted_plane)
    # n_planes_add = 0 (single-plane, nd_max = 0) exercises the dzmax == 0 branch.
    p0 = GGFitInputParams()
    p0.n_planes_add = 0
    res0 = gg_fit(field, p0)
    @test res0.nd_max == 0
    @test all(isfinite, res0.rms_weighted_plane)
  end

  @testset "public docstrings are attached" begin
    # A blank line between a docstring and its definition silently detaches it,
    # which once dropped the whole GGFitInputParams parameter reference.
    undocumented(d) = occursin("No documentation found", string(d))
    @test !undocumented(@doc GGFitInputParams)
    @test !undocumented(@doc GGCoefs)
    @test !undocumented(@doc GGFitScanPoint)
    @test !undocumented(@doc gg_fit)
    @test !undocumented(@doc gg_fit_show_results)
    @test !undocumented(@doc read_gg_fit)
    @test !undocumented(@doc write_gg_fit)
    # The three fit criteria must be spelled out where a user will look for them.
    for probe in ("fit_criterion", ":rms", ":aic", ":bic", "sqrt(RSS / W)")
      @test occursin(probe, string(@doc GGFitInputParams))
    end
  end

  @testset "gg_fit m_max/nd_max scan" begin
    field = make_field()

    # Scalar m_max/nd_max pin the model exactly and run no scan.
    p = GGFitInputParams()
    p.n_planes_add = 1
    p.m_max  = 4
    p.nd_max = 3
    res = gg_fit(field, p)
    @test isempty(res.scan)
    @test res.m_max == 4 && res.nd_max == 3
    @test maximum(k[1] for k in keys(res.b)) <= 4
    @test maximum(k[2] for k in keys(res.b)) <= 3
    # bs unknowns describe a_0 and are bounded by nd_max only, never by m_max.
    @test maximum(keys(res.bs)) <= 3

    # A vector on either cutoff scans the full grid of combinations.
    p.m_max  = 2:4
    p.nd_max = [1, 2]
    res = gg_fit(field, p)
    @test length(res.scan) == 6
    @test Set((s.m_max, s.nd_max) for s in res.scan) ==
          Set([(m, nd) for m in 2:4 for nd in 1:2])
    @test all(s -> s.n_coef > 0 && isfinite(s.rms_weighted) && isfinite(s.score), res.scan)
    # Every point weight applies to all three components alike, so each component
    # carries exactly a third of the total weight and the pooled weighted RMS is
    # the plain quadrature mean of the three per-component values.
    @test all(s -> all(isfinite, s.rms_weighted_comp) &&
                   s.rms_weighted ≈ sqrt(sum(abs2, s.rms_weighted_comp) / 3), res.scan)
    # The winner is the reported model, and its coefficient count matches its row.
    best = res.scan[argmin([s.score for s in res.scan])]
    @test (res.m_max, res.nd_max) == (best.m_max, best.nd_max)
    @test length(res.params) == best.n_coef
    @test length(res.rms_weighted_plane) == length(res.z_base)
    @test all(isfinite, res.rms_weighted_plane)
    quiet(() -> gg_fit_show_results(res, field, p))

    # Field contribution of each GG function: one row per a_m / b_m actually
    # fitted, plus b_s, each a magnitude with ave no larger than max.
    rows, b_ave = GeneralizedGradients._gg_field_contributions(res, field)
    @test [r[1] for r in rows] ==
          vcat(["a_$m" for m in sort!(unique(k[1] for k in keys(res.a)))],
               ["b_$m" for m in sort!(unique(k[1] for k in keys(res.b)))],
               isempty(res.bs) ? String[] : ["b_s"])
    @test all(r -> 0 <= r[2] <= r[3], rows)
    @test isfinite(b_ave) && b_ave > 0

    # More coefficients can only reduce the residual, so :rms takes the top model.
    p.fit_criterion = :rms
    res_rms = gg_fit(field, p)
    @test (res_rms.m_max, res_rms.nd_max) == (4, 2)
    p.fit_criterion = :aic
    @test gg_fit(field, p) isa GGCoefs

    # Candidates beyond what the table holds are clamped and deduplicated.
    p.fit_criterion = :bic
    p.m_max  = 11:40
    p.nd_max = 0:1
    res = gg_fit(field, p)
    @test length(res.scan) == 6            # m_max 11,12,13 x nd_max 0,1
    @test maximum(s.m_max for s in res.scan) == 13

    @test_throws ErrorException (p.nd_max = Int[]; gg_fit(field, p))
    p.nd_max = 0:1
    @test_throws ErrorException (p.fit_criterion = :bogus; gg_fit(field, p))
  end

  @testset "field grid HDF5 round-trip (mag + elec, curvature, RF)" begin
    mktempdir() do dir
      field = make_field(g_ref = 0.5)
      ax = axes(field.magnetic)
      edata = [field.magnetic[ix, iy, iz] .* 2.0 for ix in ax[1], iy in ax[2], iz in ax[3]]
      field.electric = OffsetArray(edata, ax...)
      field.RF_frequency = 1.3e9
      field.RF_phase = 0.25
      field.anchor_pt = GridAnchorPt.End

      path = joinpath(dir, "grid.h5")
      @test write_field_grid_hdf5(path, field) == path
      fg = read_field_grid_hdf5(path)
      @test fg.magnetic == field.magnetic
      @test fg.electric == field.electric
      @test fg.g_ref ≈ field.g_ref
      @test fg.dr ≈ field.dr && fg.r0 ≈ field.r0
      @test fg.RF_frequency ≈ field.RF_frequency
      @test fg.RF_phase ≈ field.RF_phase
      @test fg.anchor_pt == GridAnchorPt.End
    end
  end

  @testset "field grid HDF5 errors" begin
    mktempdir() do dir
      @test_throws ErrorException write_field_grid_hdf5(joinpath(dir, "empty.h5"), FieldGridTable())
      bad = joinpath(dir, "bad.h5")
      h5open(bad, "w") do f
        f["x"] = [1.0, 2.0]
      end
      @test_throws ErrorException read_field_grid_hdf5(bad)
    end
  end

  @testset "field grid Julia-source write" begin
    mktempdir() do dir
      field = make_field()
      path = joinpath(dir, "grid.jl")
      @test write_field_grid(path, field) == path
      src = read(path, String)
      @test occursin("fg = FieldGridTable()", src)
      @test occursin("fg.magnetic", src)
    end
  end

  @testset "write_bmad_field_grid_element (string/FieldGridTable, text/hdf5, em_field/sbend)" begin
    mktempdir() do dir
      field = make_field()                       # g_ref = 0 -> em_field
      gpath = joinpath(dir, "grid.h5")
      write_field_grid_hdf5(gpath, field)

      # String input, hdf5 = true (default).
      stem = joinpath(dir, "grid_out")
      ele, gfile = quiet(() -> write_bmad_field_grid_element(gpath; output_base = stem))
      @test ele == stem * ".bmad" && isfile(ele)
      @test gfile == stem * "_grid.h5" && isfile(gfile)
      @test occursin("em_field", read(ele, String))

      # String input, text grid (hdf5 = false).
      base = joinpath(dir, "out_text")
      ele2, gfile2 = quiet(() -> write_bmad_field_grid_element(gpath; output_base = base, hdf5 = false))
      @test gfile2 == base * "_grid.bmad" && isfile(gfile2)
      @test occursin("em_field", read(ele2, String))

      # FieldGridTable input (curved frame -> sbend).
      bfield = make_field(g_ref = 0.4)
      base3 = joinpath(dir, "bend")
      ele3, gfile3 = quiet(() -> write_bmad_field_grid_element(bfield; output_base = base3))
      @test isfile(base3 * ".bmad") && isfile(base3 * "_grid.h5")
      @test occursin("sbend", read(base3 * ".bmad", String))
    end
  end

  @testset "write_bmad_gg_fit (straight + bend/solenoid)" begin
    mktempdir() do dir
      # Straight reference (g_ref = 0) from the example fit file.
      base = joinpath(dir, "gg_straight")
      ele = quiet(() -> write_bmad_gg_fit(EXAMPLE; output_base = base))
      @test ele == base * ".bmad" && isfile(ele)
      @test isfile(base * "_gg.bmad")
      @test occursin("em_field", read(ele, String))

      fit, meta = read_gg_fit(EXAMPLE)
      cs, cc, c0c, npl, ndmax, kmax = GeneralizedGradients.gg_to_bmad_curves(fit)
      @test npl == length(fit.z_base)
      @test ndmax == fit.nd_max
      @test kmax >= 1

      # In-memory GGCoefs method writes the same element.
      basem = joinpath(dir, "gg_mem")
      elem = write_bmad_gg_fit(fit; output_base = basem)
      @test elem == basem * ".bmad" && isfile(elem) && isfile(basem * "_gg.bmad")

      # Curved reference (g_ref ≠ 0) with a solenoid term: fit a synthetic field,
      # write it, then convert -> exercises the sbend + solenoid + cutoff paths.
      field = make_field(g_ref = 0.3)
      p = GGFitInputParams()
      p.n_planes_add = 1
      p.output_file = joinpath(dir, "curved_fit.h5")
      res = gg_fit(field, p)
      quiet(() -> write_gg_fit(res, field, p))
      base2 = joinpath(dir, "gg_bend")
      ele2 = quiet(() -> write_bmad_gg_fit(p.output_file; output_base = base2, cutoff = 1e-6))
      @test isfile(ele2) && isfile(base2 * "_gg.bmad")
      @test occursin("sbend", read(ele2, String))
    end
  end
end
