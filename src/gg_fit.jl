#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# gg_fit.jl
#
# Fit a 3D magnetic field table to generalized-gradient (GG) coefficients
# a_n(z), b_n(z), b_s(z) and their z-derivatives, plane by plane.
#
# Usage:
#   julia src/gg_fit.jl [input_file]
# where input_file defaults to example/fit_params.jl.  The input file defines
# the field grid `field`, the bending strength `h`, the transverse `origin`, and
# the fit-control parameters `n_planes_add`, `core_weight`, `outer_plane_weight`
# (see example/fit_params.jl for full documentation).
#
# The z-axis and the s-axis (used in the TUPS09 field expansion) are identical;
# z is just the field-map notation for s.
#
# How the fit works
# -----------------
# The field expansion (tables/field_function_table.jl) is linear in the GG
# functions and their s-derivatives:
#
#   B_c(x,y,z) = Σ_{(n,m)}  CS_c,b(n,m; x,y) · b(n,m)(z)
#              + Σ_{(n,m)}  CS_c,a(n,m; x,y) · a(n,m)(z)
#              + Σ_{m}      CS_c,bs(m; x,y)  · bs(m)(z)
#
# for each field component c ∈ {Bx, By, Bs}, where
#   CS_c,f(n,m; x,y) = Σ (coeff · h^k · x^p · y^q)
# is the sum of the table entries c_f[(n,m)] = [(coeff,p,q,k), ...], and
#   b(n,m) = dᵐb_n/dzᵐ,  a(n,m) = dᵐa_n/dzᵐ,  bs(m) = dᵐ⁺¹a_0/dzᵐ⁺¹.
#
# The unknowns at a base plane z0 are the function values and their derivatives
# f(n,m)(z0), m = 0 … m_max.  The field on a neighbouring plane at offset
# dz = z - z0 is obtained by Taylor-extrapolating each derivative:
#   f(n,m)(z0+dz) = Σ_{j≥m} dz^(j-m)/(j-m)! · f(n,j)(z0).
# Substituting makes the model linear in the base-plane unknowns f(n,j)(z0):
#   design entry for unknown f(n,j) =
#       Σ_{m=0}^{j} CS_c,f(n,m; x,y) · dz^(j-m)/(j-m)!.
#
# Each base plane is then solved by weighted linear least squares over all
# field points lying within `n_planes_add` planes of the base plane.
# ---------------------------------------------------------------------------

using JLD2, OffsetArrays, LinearAlgebra, Printf

# ---------------------------------------------------------------------------
# Load input parameters and the GG coefficient table
# ---------------------------------------------------------------------------

const INPUT_FILE = length(ARGS) >= 1 ? ARGS[1] :
                   joinpath(@__DIR__, "..", "example", "fit_params.jl")
const TABLE_FILE = joinpath(@__DIR__, "..", "tables", "field_function_table.jl")

include(INPUT_FILE)   # defines: field, h, origin, n_planes_add, core_weight, outer_plane_weight
include(TABLE_FILE)   # defines: Bx_a By_a Bs_a  Bx_b By_b Bs_b  Bx_bs By_bs Bs_bs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# CoefSum: Σ coeff·h^k·x^p·y^q over the table entries for one (component,function).
function coefsum(terms, x::Float64, y::Float64, h)
    s = 0.0
    for (c, p, q, k) in terms
        hk = k == 0 ? 1.0 : float(h)^k
        s += float(c) * hk * x^p * y^q
    end
    return s
end

# Float factorial (m_max is small, but keep it overflow-proof).
ffact(k::Int) = k <= 1 ? 1.0 : prod(2.0:float(k))

# ---------------------------------------------------------------------------
# Main fit routine
# ---------------------------------------------------------------------------

function run_fit(pt, r0, dr, h, origin, n_planes_add, core_weight, outer_plane_weight,
                 a_dicts, b_dicts, bs_dicts)

    npa     = n_planes_add
    m_max   = 2 * npa                 # highest derivative order the planes can resolve
    dz_grid = dr[3]

    ix_lo, ix_hi = first(axes(pt, 1)), last(axes(pt, 1))
    iy_lo, iy_hi = first(axes(pt, 2)), last(axes(pt, 2))
    iz_lo, iz_hi = first(axes(pt, 3)), last(axes(pt, 3))
    ixs = ix_lo:ix_hi
    iys = iy_lo:iy_hi

    # Transverse coordinates relative to the GG origin (the expansion axis).
    xs = [r0[1] + dr[1] * ix - origin[1] for ix in ixs]
    ys = [r0[2] + dr[2] * iy - origin[2] for iy in iys]
    rmax2 = maximum(xs[i]^2 + ys[j]^2 for i in eachindex(xs), j in eachindex(ys))

    # ---- Assemble the parameter (unknown) list from the table keys --------
    # a/b indexed by (n,m); bs indexed by m (stored as (0,m)).
    pset = Set{Tuple{Symbol,Int,Int}}()
    for d in a_dicts, k in keys(d)
        k[2] <= m_max && push!(pset, (:a, k[1], k[2]))
    end
    for d in b_dicts, k in keys(d)
        k[2] <= m_max && push!(pset, (:b, k[1], k[2]))
    end
    for d in bs_dicts, m in keys(d)
        m <= m_max && push!(pset, (:bs, 0, m))
    end
    params = sort!(collect(pset))
    ncols  = length(params)

    # ---- Precompute CoefSum grids: (comp,type,n,m) => matrix over (ix,iy) --
    # comp: 1=Bx, 2=By, 3=Bs.   type: :a,:b,:bs.
    comp_dicts = Dict(
        (1, :a) => a_dicts[1], (2, :a) => a_dicts[2], (3, :a) => a_dicts[3],
        (1, :b) => b_dicts[1], (2, :b) => b_dicts[2], (3, :b) => b_dicts[3],
        (1, :bs) => bs_dicts[1], (2, :bs) => bs_dicts[2], (3, :bs) => bs_dicts[3],
    )
    CS = Dict{Tuple{Int,Symbol,Int,Int},Matrix{Float64}}()
    for ((comp, typ), d) in comp_dicts
        for (key, terms) in d
            n, m = typ == :bs ? (0, key) : (key[1], key[2])
            m <= m_max || continue
            grid = [coefsum(terms, xs[i], ys[j], h) for i in eachindex(xs), j in eachindex(ys)]
            CS[(comp, typ, n, m)] = grid
        end
    end

    # ---- Result containers ------------------------------------------------
    nplanes = length(iz_lo:iz_hi)
    z_base  = [r0[3] + dr[3] * iz for iz in iz_lo:iz_hi]
    res_a   = Dict{Tuple{Int,Int},Vector{Float64}}()
    res_b   = Dict{Tuple{Int,Int},Vector{Float64}}()
    res_bs  = Dict{Int,Vector{Float64}}()
    for (typ, n, m) in params
        typ == :a  && (res_a[(n, m)] = fill(NaN, nplanes))
        typ == :b  && (res_b[(n, m)] = fill(NaN, nplanes))
        typ == :bs && (res_bs[m]     = fill(NaN, nplanes))
    end
    rms_plane = fill(NaN, nplanes)

    # ---- Loop over base planes -------------------------------------------
    for (pidx, iz0) in enumerate(iz_lo:iz_hi)
        izs   = max(iz_lo, iz0 - npa):min(iz_hi, iz0 + npa)
        dzs   = [(iz - iz0) * dz_grid for iz in izs]
        dzmax = maximum(abs, dzs)

        npts  = length(ixs) * length(iys) * length(izs)
        nrows = 3 * npts
        A     = zeros(nrows, ncols)
        bvec  = zeros(nrows)
        sw    = zeros(nrows)          # sqrt of point weight

        row = 0
        for (zi, iz) in enumerate(izs)
            dz  = dzs[zi]
            wpl = (npa == 0 || dzmax == 0) ? 1.0 :
                  1 + (outer_plane_weight - 1) * abs(dz) / dzmax
            for (iix, ix) in enumerate(ixs), (iiy, iy) in enumerate(iys)
                r2  = xs[iix]^2 + ys[iiy]^2
                wco = core_weight == 1 ? 1.0 :
                      core_weight * rmax2 / (rmax2 + r2 * (core_weight - 1))
                w   = wco * wpl
                B3  = pt[ix, iy, iz]                      # [Bx, By, Bs]
                for comp in 1:3
                    row += 1
                    bvec[row] = B3[comp]
                    sw[row]   = sqrt(w)
                    for (col, (typ, n, j)) in enumerate(params)
                        val = 0.0
                        for mm in 0:j
                            grid = get(CS, (comp, typ, n, mm), nothing)
                            grid === nothing && continue
                            val += grid[iix, iiy] * dz^(j - mm) / ffact(j - mm)
                        end
                        A[row, col] = val
                    end
                end
            end
        end

        # Weighted least squares (pinv = stable min-norm solution if rank-deficient).
        Aw    = A .* sw
        bw    = bvec .* sw
        theta = pinv(Aw) * bw

        # Store coefficients and weighted RMS residual.
        for (col, (typ, n, m)) in enumerate(params)
            typ == :a  && (res_a[(n, m)][pidx] = theta[col])
            typ == :b  && (res_b[(n, m)][pidx] = theta[col])
            typ == :bs && (res_bs[m][pidx]     = theta[col])
        end
        rms_plane[pidx] = norm(Aw * theta - bw) / sqrt(nrows)
    end

    return (; z_base, params, res_a, res_b, res_bs, rms_plane, m_max)
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

pt = field["pt"]
r0 = field["r0_grid"]
dr = field["dr_grid"]

a_dicts  = (Bx_a, By_a, Bs_a)
b_dicts  = (Bx_b, By_b, Bs_b)
bs_dicts = (Bx_bs, By_bs, Bs_bs)

result = run_fit(pt, r0, dr, h, origin, n_planes_add, core_weight, outer_plane_weight,
                 a_dicts, b_dicts, bs_dicts)

# ---- Report --------------------------------------------------------------
println("="^72)
println("GG fit complete")
println("  input file        : ", INPUT_FILE)
println("  field grid        : ", join(length.(axes(pt)), " x "), "  (ix, iy, iz)")
println("  h                 : ", h)
println("  origin (x,y)      : ", origin)
println("  n_planes_add      : ", n_planes_add, "   (max derivative order m_max = ", result.m_max, ")")
println("  core_weight       : ", core_weight)
println("  outer_plane_weight: ", outer_plane_weight)
println("  # GG coefficients : ", length(result.params), " per plane")
println("  # base planes     : ", length(result.z_base))
println("-"^72)
@printf("%-6s  %-12s  %-12s\n", "plane", "z [m]", "wRMS resid")
for i in eachindex(result.z_base)
    @printf("%-6d  %-12.6g  %-12.4e\n", i, result.z_base[i], result.rms_plane[i])
end
println("-"^72)

# Show the leading multipoles at the central plane for a quick sanity check.
ic = cld(length(result.z_base), 2)
println("Leading coefficients at central plane (z = ",
        @sprintf("%.6g", result.z_base[ic]), "):")
for n in 1:6
    b00 = get(result.res_b, (n, 0), nothing)
    a00 = get(result.res_a, (n, 0), nothing)
    bstr = b00 === nothing ? "     -      " : @sprintf("% .6e", b00[ic])
    astr = a00 === nothing ? "     -      " : @sprintf("% .6e", a00[ic])
    @printf("  n=%-2d   b(n,0)=%s   a(n,0)=%s\n", n, bstr, astr)
end
if haskey(result.res_bs, 0)
    @printf("  bs(0) = % .6e\n", result.res_bs[0][ic])
end
println("="^72)

# ---- Save ----------------------------------------------------------------
outfile = joinpath(dirname(INPUT_FILE), "gg_fit_result.jld2")
jldsave(outfile;
        z_base    = result.z_base,
        a         = result.res_a,
        b         = result.res_b,
        bs        = result.res_bs,
        rms_plane = result.rms_plane,
        m_max     = result.m_max,
        h         = h,
        origin    = origin,
        r0_grid   = r0,
        dz_grid   = dr[3])
println("Results written to ", outfile)
