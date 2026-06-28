using GeneralizedGradients
using Test
using OffsetArrays
using HDF5

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

# Build a synthetic gg_fit NamedTuple (same shape as gg_load_fit returns) so we
# can exercise finite curvature g_ref, which the example file (g_ref = 0) does not.
synth(z_base, a, b, bs, g_ref; m_max, dz_grid) = (;
  z_base = collect(float.(z_base)),
  a = a, b = b, bs = bs,
  g_ref = g_ref, origin = [0.0, 0.0], dz_grid = dz_grid,
  m_max = m_max, rms_plane = fill(NaN, length(z_base)))

const PTS = ((0.004, 0.003), (-0.005, 0.002), (0.003, -0.004), (0.0, 0.006), (0.007, 0.0))

@testset "GeneralizedGradients" begin

  @testset "gg_load_fit" begin
    gg = gg_load_fit(EXAMPLE)
    for k in (:z_base, :a, :b, :bs, :g_ref, :origin, :dz_grid, :m_max, :rms_plane)
      @test hasproperty(gg, k)
    end
    @test length(gg.z_base) == length(gg.rms_plane)
    @test gg.a isa Dict && gg.b isa Dict && gg.bs isa Dict
  end

  @testset "curl(A) == B at grid planes (example, g_ref=0)" begin
    gg = gg_load_fit(EXAMPLE)
    for ip in 1:length(gg.z_base), (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(gg, ip, x, y)
      Bc = curl_from(A, dA, x - gg.origin[1], gg.g_ref)
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
    gg = synth([0.0], va, vb, vbs, 0.6; m_max = 2, dz_grid = 0.1)
    for (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(gg, 1, x, y)
      Bc = curl_from(A, dA, x, gg.g_ref)
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
  ggM = synth(zg, amap, bmap, bsmap, 0.5; m_max = 3, dz_grid = 0.05)

  @testset "curl(A) == B, multi-plane (g_ref=0.5)" begin
    for ip in 1:length(zg), (x, y) in PTS
      B, A, dA = field_and_potential_evaluate(ggM, ip, x, y)
      Bc = curl_from(A, dA, x, ggM.g_ref)
      @test maximum(abs, B .- Bc) < 1e-12
    end
  end

  @testset "∂A/∂s matches finite difference" begin
    ip = 6; sc = zg[ip]; xq, yq = 0.006, -0.004
    _, _, dA = field_and_potential_evaluate(ggM, ip, xq, yq)
    δ = 1e-5
    _, Ap, _ = field_and_potential_evaluate_at(ggM, xq, yq, sc + δ)
    _, Am, _ = field_and_potential_evaluate_at(ggM, xq, yq, sc - δ)
    dAs_fd = (Ap .- Am) ./ (2δ)
    @test maximum(abs, dA[:, 3] .- dAs_fd) < 1e-6
  end

  @testset "gg_coefficients_at_plane matches direct indexing" begin
    gg = gg_load_fit(EXAMPLE)
    ip = 3
    a, b, bs = gg_coefficients_at_plane(gg, ip)
    @test a isa Dict{Tuple{Int,Int},Float64}
    @test bs isa Dict{Int,Float64}
    for (nm, v) in gg.a; @test a[nm] == v[ip]; end
    for (nm, v) in gg.b; @test b[nm] == v[ip]; end
    for (m, v) in gg.bs; @test bs[m] == v[ip]; end
  end

  @testset "_at_s reproduces plane values at a grid plane" begin
    gg = gg_load_fit(EXAMPLE)
    ip = 4; s = gg.z_base[ip]

    a, b, bs = gg_coefficients_at_plane(gg, ip)
    as, bsd, bss = gg_coefficients_at_s(gg, s)
    @test as == a && bsd == b && bss == bs

    CBx, CBy, CBs = field_coefficients_at_plane(gg, ip)
    CBxs, CBys, CBss = field_coefficients_at_s(gg, s)
    @test CBxs == CBx && CBys == CBy && CBss == CBs

    B, A, dA = field_and_potential_evaluate(gg, ip, 0.004, 0.003)
    Bs, As, dAs = field_and_potential_evaluate_at(gg, 0.004, 0.003, s)
    @test Bs ≈ B && As ≈ A && dAs ≈ dA
  end

  @testset "gg_fit + show + write round-trip" begin
    field = make_field()
    p = GGFitParams()
    p.n_planes_add = 1
    res = gg_fit(field, p)
    @test res isa GGFitResults
    @test length(res.z_base) == size(field.magnetic, 3)
    @test res.m_max == 2
    @test length(res.rms_plane) == length(res.z_base)
    @test all(isfinite, res.rms_plane)
    @test !isempty(res.params)
    quiet(() -> gg_fit_show_results(res, field, p))

    mktempdir() do dir
      p.output_file = joinpath(dir, "fit.h5")
      out = quiet(() -> gg_fit_write_results(res, field, p))
      @test out == p.output_file && isfile(out)
      gg = gg_load_fit(out)
      @test gg.m_max == res.m_max
      @test gg.dz_grid ≈ field.dr[3]
      @test length(gg.z_base) == length(res.z_base)
      @test Set(keys(gg.a)) == Set(keys(res.res_a))
      @test Set(keys(gg.bs)) == Set(keys(res.res_bs))
      for (k, v) in res.res_a; @test gg.a[k] ≈ v; end
      for (k, v) in res.res_b; @test gg.b[k] ≈ v; end
      for (k, v) in res.res_bs; @test gg.bs[k] ≈ v; end
    end
  end

  @testset "gg_fit weighting and n_planes_add=0 branches" begin
    field = make_field()
    # Non-default core/outer weights exercise the weighting branches.
    p = GGFitParams()
    p.n_planes_add = 1
    p.core_weight = 2
    p.outer_plane_weight = 2
    res = gg_fit(field, p)
    @test all(isfinite, res.rms_plane)
    # n_planes_add = 0 (single-plane, m_max = 0) exercises the dzmax == 0 branch.
    p0 = GGFitParams()
    p0.n_planes_add = 0
    res0 = gg_fit(field, p0)
    @test res0.m_max == 0
    @test all(isfinite, res0.rms_plane)
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

  @testset "field_grid_to_bmad (string/FieldGridTable, text/hdf5, em_field/sbend)" begin
    mktempdir() do dir
      field = make_field()                       # g_ref = 0 -> em_field
      gpath = joinpath(dir, "grid.h5")
      write_field_grid_hdf5(gpath, field)

      # String input, default output_base (from file name) and default hdf5 = true.
      ele = quiet(() -> field_grid_to_bmad(gpath))
      stem = joinpath(dir, "grid")
      @test ele == stem * ".bmad" && isfile(ele)
      @test isfile(stem * "_grid.h5")
      @test occursin("em_field", read(ele, String))

      # String input, explicit output_base, text grid (hdf5 = false).
      base = joinpath(dir, "out_text")
      ele2 = quiet(() -> field_grid_to_bmad(gpath; output_base = base, hdf5 = false))
      @test isfile(base * "_grid.bmad")
      @test occursin("em_field", read(ele2, String))

      # FieldGridTable input (curved frame -> sbend), default output_base + hdf5.
      cd(dir) do
        bfield = make_field(g_ref = 0.4)
        ele3 = quiet(() -> field_grid_to_bmad(bfield))
        @test isfile("field_grid.bmad") && isfile("field_grid_grid.h5")
        @test occursin("sbend", read("field_grid.bmad", String))
      end
    end
  end

  @testset "gg_to_bmad (straight + bend/solenoid)" begin
    mktempdir() do dir
      # Straight reference (g_ref = 0) from the example fit file.
      base = joinpath(dir, "gg_straight")
      ele = quiet(() -> gg_to_bmad(EXAMPLE; output_base = base))
      @test ele == base * ".bmad" && isfile(ele)
      @test isfile(base * "_gg.bmad")
      @test occursin("em_field", read(ele, String))

      fit = gg_load_fit(EXAMPLE)
      cs, cc, c0c, npl, mmax, kmax = GeneralizedGradients.gg_to_bmad_curves(fit)
      @test npl == length(fit.z_base)
      @test mmax == fit.m_max
      @test kmax >= 1

      # Curved reference (g_ref ≠ 0) with a solenoid term: fit a synthetic field,
      # write it, then convert -> exercises the sbend + solenoid + cutoff paths.
      field = make_field(g_ref = 0.3)
      p = GGFitParams()
      p.n_planes_add = 1
      p.output_file = joinpath(dir, "curved_fit.h5")
      res = gg_fit(field, p)
      quiet(() -> gg_fit_write_results(res, field, p))
      base2 = joinpath(dir, "gg_bend")
      ele2 = quiet(() -> gg_to_bmad(p.output_file; output_base = base2, cutoff = 1e-6))
      @test isfile(ele2) && isfile(base2 * "_gg.bmad")
      @test occursin("sbend", read(ele2, String))
    end
  end
end
