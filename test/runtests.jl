using GeneralizedGradients
using Test

# Self-contained test fixture (a copy of example/gg_fit_result.jld2).
const EXAMPLE = joinpath(@__DIR__, "data", "gg_fit_result.jld2")

# Frenet–Serret curl of the vector potential.  Given A and its Jacobian
# dA[i,j] = ∂A_i/∂u_j with u = (x, y, s), reconstruct B; this must equal the
# B returned by the evaluator for ANY coefficient values (constant curvature h).
function curl_from(A, dA, x, h)
    g = 1 + h * x
    Bx = dA[3, 2] - dA[2, 3] / g
    By = dA[1, 3] / g - (h * A[3] + g * dA[3, 1]) / g
    Bs = dA[2, 1] - dA[1, 2]
    return [Bx, By, Bs]
end

# Build a synthetic gg_fit NamedTuple (same shape as gg_load_fit returns) so we
# can exercise finite curvature h, which the example file (h = 0) does not.
synth(z_base, a, b, bs, h; m_max, dz_grid) = (;
    z_base = collect(float.(z_base)),
    a = a, b = b, bs = bs,
    h = h, origin = [0.0, 0.0], dz_grid = dz_grid,
    m_max = m_max, rms_plane = fill(NaN, length(z_base)))

const PTS = ((0.004, 0.003), (-0.005, 0.002), (0.003, -0.004), (0.0, 0.006), (0.007, 0.0))

@testset "GeneralizedGradients" begin

    @testset "gg_load_fit" begin
        gg = gg_load_fit(EXAMPLE)
        for k in (:z_base, :a, :b, :bs, :h, :origin, :dz_grid, :m_max, :rms_plane)
            @test hasproperty(gg, k)
        end
        @test length(gg.z_base) == length(gg.rms_plane)
        @test gg.a isa Dict && gg.b isa Dict && gg.bs isa Dict
    end

    @testset "curl(A) == B at grid planes (example, h=0)" begin
        gg = gg_load_fit(EXAMPLE)
        for ip in 1:length(gg.z_base), (x, y) in PTS
            B, A, dA = field_and_potential_evaluate(gg, ip, x, y)
            Bc = curl_from(A, dA, x - gg.origin[1], gg.h)
            @test maximum(abs, B .- Bc) < 1e-12
        end
    end

    @testset "curl(A) == B, synthetic single plane (h=0.6)" begin
        a  = Dict((1,0)=>0.7,(2,0)=>-0.4,(3,0)=>0.25,(1,1)=>0.3,(2,1)=>-0.15,(1,2)=>0.2)
        b  = Dict((1,0)=>0.5,(2,0)=>0.35,(3,0)=>-0.2,(1,1)=>0.1,(2,1)=>0.05)
        bs = Dict(0=>0.45, 1=>-0.12, 2=>0.08)
        va = Dict(k => [v] for (k, v) in a)
        vb = Dict(k => [v] for (k, v) in b)
        vbs = Dict(k => [v] for (k, v) in bs)
        gg = synth([0.0], va, vb, vbs, 0.6; m_max = 2, dz_grid = 0.1)
        for (x, y) in PTS
            B, A, dA = field_and_potential_evaluate(gg, 1, x, y)
            Bc = curl_from(A, dA, x, gg.h)
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

    @testset "curl(A) == B, multi-plane (h=0.5)" begin
        for ip in 1:length(zg), (x, y) in PTS
            B, A, dA = field_and_potential_evaluate(ggM, ip, x, y)
            Bc = curl_from(A, dA, x, ggM.h)
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
end
