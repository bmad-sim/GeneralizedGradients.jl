#!/usr/bin/env julia

"""
    gg_fit.jl

Fit a 3D magnetic field table to generalized-gradient (GG) coefficients
a_n(z), b_n(z), b_s(z) and their z-derivatives, plane by plane.

## Usage

Run with command:
  julia src/gg_fit.jl [parameter_input_file]
The parameter input file defines the field grid `field`, the transverse 
`origin`, and the fit-control parameters `n_planes_add`, `core_weight`, `outer_plane_weight`. See below.

Besides the input file, this function will open the file (relative to this directory)
"../tables/gg_coef_table.jl" (this relative to this gg_fit.jl file) which contains 
coefficients needed for the fit.

## How the fit works

The field expansion (tables/gg_coef_table.jl) is linear in the GG
functions and their s-derivatives:

  B_c(x,y,z) = Σ_{(n,m)}  CS_c,b(n,m; x,y) · b(n,m)(z)
             + Σ_{(n,m)}  CS_c,a(n,m; x,y) · a(n,m)(z)
             + Σ_{m}      CS_c,bs(m; x,y)  · bs(m)(z)

for each field component c ∈ {Bx, By, Bs}, where
  CS_c,f(n,m; x,y) = Σ (coeff · g_ref^k · x^p · y^q)
is the sum of the table entries c_f[(n,m)] = [(coeff,p,q,k), ...], and
  b(n,m) = dᵐb_n/dzᵐ,  a(n,m) = dᵐa_n/dzᵐ,  bs(m) = dᵐ⁺¹a_0/dzᵐ⁺¹.

The unknowns at a base plane z0 are the function values and their derivatives
f(n,m)(z0), m = 0 … m_max.  The field on a neighbouring plane at offset
dz = z - z0 is obtained by Taylor-extrapolating each derivative:
  f(n,m)(z0+dz) = Σ_{j≥m} dz^(j-m)/(j-m)! · f(n,j)(z0).
Substituting makes the model linear in the base-plane unknowns f(n,j)(z0):
  design entry for unknown f(n,j) =
      Σ_{m=0}^{j} CS_c,f(n,m; x,y) · dz^(j-m)/(j-m)!.

Each base plane is then solved by weighted linear least squares over all
field points lying within `n_planes_add` planes of the base plane.

## Input parameter file

The input parameter file defines a number of parameters. 
Example parameter file is at "example/fit_params.jl".

### Example input file

using GeneralizedGradients

field = read_field_grid("wsnk_field.h5")   # Field table dict (HDF5).
origin = [-0.001, 0.0]      # (x, y) origin about which the generalized gradients coefs are computed
n_planes_add = 1            # Number of z-planes added.
core_weight = 1             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
outer_plane_weight = 1      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
output_file = "gg_fit_result.h5"

### origin = [x0, y0]

Defines the line [x0, y0, z] about which the generalized gradient coefficients are computed.
If g_ref is non-zero, origin must be [0, 0].

### n_planes_add = Int

This parameter sets the number of z-planes added to either side of the base z-plane to
be used in the analysis of the derivatives at any given base z-plane (see "How the GG Calculation
Works" section). For example, for n_planes_add = 2, two planes would be added to either side of the
base plane making the total number of planes used in the analysis equal to five.

### core_weight = Float

Merit function weight for "core" points (field table points whose transverse (x,y)
position is near (0,0)). Default is 1.0 which gives an equal weight for all points of a given
z-plane. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

### outer_plane_weight = Float

Merit function weight for z-planes away from the base z-plane when n_planes_add
is non-zero. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

### output_file

Name of the output file.

### field

A `FieldGridTable` holding the field table and associated parameters
(in the (x, y, z) curvilinear coordinate system):
  field.magnetic[c, ix, iy, iz]  Field components (c = 1,2,3 -> Bx, By, Bz). An
                                 OffsetArray; the grid indices (ix, iy, iz) need
                                 not start at 0 or 1.
  field.r0                        Grid origin 3-vector
  field.dr                        Grid spacing 3-vector
  field.g_ref                     Bending strength = 1 / bending_radius
A grid point (ix, iy, iz) has (x, y, z) position r0 + dr .* (ix, iy, iz).

Note: To construct a field file to be read in use the following:
  using GeneralizedGradients, OffsetArrays
  write_field_grid("this_file.h5", FieldGridTable{Float64}(; magnetic = B, r0 = r0,
                                                             dr = dr, g_ref = g_ref))
where `B` is a (3, nx, ny, nz) OffsetArray with axes (1:3, ix_lo:ix_hi, …).
""" gg_fit

using OffsetArrays, LinearAlgebra, Printf, GeneralizedGradients

# ---------------------------------------------------------------------------
# Load input parameters and the GG coefficient table
# ---------------------------------------------------------------------------

const INPUT_FILE = joinpath(pwd(), ARGS[1])
const TABLE_FILE = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")

include(INPUT_FILE)   # defines: field, origin, n_planes_add, core_weight, outer_plane_weight
include(TABLE_FILE)   # defines: Bx_a By_a Bs_a  Bx_b By_b Bs_b  Bx_bs By_bs Bs_bs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# CB coefficient sum: Σ coeff·g_ref^k·x^p·y^q over the table entries for one
# (component, function) — one entry of the CB grids built in run_fit.
function coefsum(terms, x::Float64, y::Float64, g_ref)
    s = 0.0
    for (c, p, q, k) in terms
        hk = k == 0 ? 1.0 : float(g_ref)^k
        s += float(c) * hk * x^p * y^q
    end
    return s
end

# Float factorial (m_max is small, but keep it overflow-proof).
ffact(k::Int) = k <= 1 ? 1.0 : prod(2.0:float(k))

# ---------------------------------------------------------------------------
# Main fit routine
# ---------------------------------------------------------------------------

function run_fit(field::FieldGridTable, origin, n_planes_add, core_weight, outer_plane_weight,
                 a_dicts, b_dicts, bs_dicts)

    mag   = field.magnetic            # OffsetArray: mag[comp, ix, iy, iz]
    r0    = field.r0                   # grid origin (gridOriginOffset)
    dr    = field.dr
    g_ref = field.g_ref

    npa     = n_planes_add
    m_max   = 2 * npa                 # highest derivative order the planes can resolve
    dz_grid = dr[3]

    # Grid index ranges, taken from the field arrays (not assumed to start at 0/1).
    # Use plain UnitRanges (not the OffsetArray axes) so the `xs`/`ys`/`z_base`
    # comprehensions below stay 1-based while `mag` is still indexed by real index.
    ixs = first(axes(mag, 2)):last(axes(mag, 2))
    iys = first(axes(mag, 3)):last(axes(mag, 3))
    izs_grid = first(axes(mag, 4)):last(axes(mag, 4))

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

    # ---- Precompute CB (field-coefficient) grids: (comp,type,n,m) => matrix over (ix,iy) --
    # comp: 1=Bx, 2=By, 3=Bs.   type: :a,:b,:bs.
    comp_dicts = Dict(
        (1, :a) => a_dicts[1], (2, :a) => a_dicts[2], (3, :a) => a_dicts[3],
        (1, :b) => b_dicts[1], (2, :b) => b_dicts[2], (3, :b) => b_dicts[3],
        (1, :bs) => bs_dicts[1], (2, :bs) => bs_dicts[2], (3, :bs) => bs_dicts[3],
    )
    CB = Dict{Tuple{Int,Symbol,Int,Int},Matrix{Float64}}()
    for ((comp, typ), d) in comp_dicts
        for (key, terms) in d
            n, m = typ == :bs ? (0, key) : (key[1], key[2])
            m <= m_max || continue
            grid = [coefsum(terms, xs[i], ys[j], g_ref) for i in eachindex(xs), j in eachindex(ys)]
            CB[(comp, typ, n, m)] = grid
        end
    end

    # ---- Result containers ------------------------------------------------
    nplanes = length(izs_grid)
    z_base  = [r0[3] + dr[3] * iz for iz in izs_grid]
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
    for (pidx, iz0) in enumerate(izs_grid)
        izs   = max(first(izs_grid), iz0 - npa):min(last(izs_grid), iz0 + npa)
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
                B3  = @view mag[:, ix, iy, iz]            # [Bx, By, Bs]
                for comp in 1:3
                    row += 1
                    bvec[row] = B3[comp]
                    sw[row]   = sqrt(w)
                    for (col, (typ, n, j)) in enumerate(params)
                        val = 0.0
                        for mm in 0:j
                            grid = get(CB, (comp, typ, n, mm), nothing)
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

# `field` is a FieldGridTable (from read_field_grid in the parameter file).
dr    = field.dr
g_ref = field.g_ref

a_dicts  = (Bx_a, By_a, Bs_a)
b_dicts  = (Bx_b, By_b, Bs_b)
bs_dicts = (Bx_bs, By_bs, Bs_bs)

result = run_fit(field, origin, n_planes_add, core_weight, outer_plane_weight,
                 a_dicts, b_dicts, bs_dicts)

# ---- Report --------------------------------------------------------------
println("="^72)
println("GG fit complete")
println("  input file        : ", INPUT_FILE)
println("  field grid        : ", join(size(field.magnetic)[2:4], " x "), "  (ix, iy, iz)")
println("  g_ref             : ", g_ref)
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
# Stores the fitted GG coefficients plus enough metadata to reproduce and
# interpret the fit later.  The (large) input field table is deliberately NOT
# stored; only its grid geometry (r0_grid, dz_grid) is kept.
outfile = joinpath(dirname(INPUT_FILE), output_file)
gg_save_fit(outfile;
        z_base             = result.z_base,
        a                  = result.res_a,
        b                  = result.res_b,
        bs                 = result.res_bs,
        rms_plane          = result.rms_plane,
        m_max              = result.m_max,
        g_ref              = g_ref,
        origin             = origin,
        dz_grid            = dr[3],
        # Fit-control parameters, retained for later reference / reproducibility.
        n_planes_add       = n_planes_add,
        core_weight        = core_weight,
        outer_plane_weight = outer_plane_weight,
        input_file         = INPUT_FILE)
println("Results written to ", outfile)
