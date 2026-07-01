# ---------------------------------------------------------------------------
# gg_fit.jl
#
# Fit a 3D magnetic field grid to generalized-gradient (GG) coefficients.
# The GG coefficient table (Bx_a … Bs_bs) is a module global brought in by
# gg_eval.jl's include of tables/gg_coef_table.jl, so it is used directly here.
# ---------------------------------------------------------------------------

"""
    gg_fit(field::FieldGridTable, params::GGFitInputParams) -> GGCoefs

Fit a 3D magnetic field grid to generalized-gradient (GG) coefficients
`a_n(z)`, `b_n(z)`, `b_s(z)` and their `z`-derivatives, plane by plane.

The returned `GGCoefs` holds the fitted coefficients and per-plane
diagnostics. Use `gg_fit_show_results` to print a summary and
`write_gg_fit` to save the result to an HDF5 file (readable by
`read_gg_fit`).

See `examples/run_gg_fit.jl` for a complete, runnable example.

## Arguments

- `field` — a `FieldGridTable`.
  `field.magnetic[ix,iy,iz]` is the `[Bx,By,Bz]` 3-vector at the grid point,
  whose `(x, y, z)` position is `field.r0 + field.dr .* (ix, iy, iz)`.
- `params` — a `GGFitInputParams` holding the fit parameters (`origin`,
  `n_planes_add`, `core_weight`, `outer_plane_weight`, `output_file`).

## How the fit works

The GG coefficients are computed at the equally spaced `z`-positions of the
field-table planes. The fit is done plane by plane: the coefficients of a given
plane (the "base plane") are computed independently of every other plane, and
all coefficients of a base plane are solved for simultaneously by minimizing a
merit function

```
Merit = Σ  weight · (field_from_table - field_from_GG_coefs)^2
```

The sum runs over all field points lying in a plane within `n_planes_add` of the
base plane. For example, `n_planes_add = 2` adds two planes on either side, so
five planes are used in total. Near the ends of the table the count is reduced —
a base plane at the very end of the table uses only three planes when
`n_planes_add = 2`.

The field expansion (`tables/gg_coef_table.jl`) is linear in the GG functions
and their `s`-derivatives:

```
B_c(x,y,z) = Σ_{(n,m)}  CS_c,b(n,m; x,y) · b(n,m)(z)
           + Σ_{(n,m)}  CS_c,a(n,m; x,y) · a(n,m)(z)
           + Σ_{m}      CS_c,bs(m; x,y)  · bs(m)(z)
```

for each field component `c ∈ {Bx, By, Bs}`, where

```
CS_c,f(n,m; x,y) = Σ (coeff · g_ref^k · x^p · y^q)
```

is the sum of the table entries `c_f[(n,m)] = [(coeff,p,q,k), ...]`, and
`b(n,m) = dᵐb_n/dzᵐ`, `a(n,m) = dᵐa_n/dzᵐ`, `bs(m) = dᵐ⁺¹a_0/dzᵐ⁺¹`.

The unknowns at a base plane `z0` are the function values and their derivatives
`f(n,m)(z0)`, `m = 0 … m_max`. The field on a neighbouring plane at offset
`dz = z - z0` is obtained by Taylor-extrapolating each derivative:

```
f(n,m)(z0+dz) = Σ_{j≥m} dz^(j-m)/(j-m)! · f(n,j)(z0)
```
where `f` is either `a`, `b`, or `bs`.
Substituting makes the model linear in the base-plane unknowns `f(n,j)(z0)`:

```
design entry for unknown f(n,j) = Σ_{m=0}^{j} CS_c,f(n,m; x,y) · dz^(j-m)/(j-m)!
```

Each base plane is then solved by weighted linear least squares over all field
points lying within `n_planes_add` planes of the base plane.

Adding extra planes smooths the computed values, but the approximation that the
GG curve is well represented by the base-plane Taylor polynomial becomes less
accurate as more planes are added. Past some limit, using more planes makes the
fit *less* accurate.

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
  used to resolve derivatives (`m_max = 2*n_planes_add`).
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
  r0    = field.r0                   # grid origin (gridOriginOffset)
  dr    = field.dr
  g_ref = field.g_ref

  npa     = n_planes_add
  m_max   = 2 * npa                 # highest derivative order the planes can resolve
  dz_grid = dr[3]

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
  params_list = sort!(collect(pset))
  ncols  = length(params_list)

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
      grid = [_coefsum(terms, xs[i], ys[j], g_ref) for i in eachindex(xs), j in eachindex(ys)]
      CB[(comp, typ, n, m)] = grid
    end
  end

  # ---- Result containers ------------------------------------------------
  nplanes = length(izs_grid)
  z_base  = [r0[3] + dr[3] * iz for iz in izs_grid]
  a   = Dict{Tuple{Int,Int},Vector{Float64}}()
  b   = Dict{Tuple{Int,Int},Vector{Float64}}()
  bs  = Dict{Int,Vector{Float64}}()
  for (typ, n, m) in params_list
    typ == :a  && (a[(n, m)] = fill(NaN, nplanes))
    typ == :b  && (b[(n, m)] = fill(NaN, nplanes))
    typ == :bs && (bs[m]     = fill(NaN, nplanes))
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
        B3  = mag[ix, iy, iz]                     # [Bx, By, Bs]
        for comp in 1:3
          row += 1
          bvec[row] = B3[comp]
          sw[row]   = sqrt(w)
          for (col, (typ, n, j)) in enumerate(params_list)
            val = 0.0
            for mm in 0:j
              grid = get(CB, (comp, typ, n, mm), nothing)
              grid === nothing && continue
              val += grid[iix, iiy] * dz^(j - mm) / _ffact(j - mm)
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
    for (col, (typ, n, m)) in enumerate(params_list)
      typ == :a  && (a[(n, m)][pidx] = theta[col])
      typ == :b  && (b[(n, m)][pidx] = theta[col])
      typ == :bs && (bs[m][pidx]     = theta[col])
    end
    rms_plane[pidx] = norm(Aw * theta - bw) / sqrt(nrows)
  end

  return GGCoefs(; z_base, params = params_list, a, b, bs, rms_plane, m_max,
                    g_ref = field.g_ref, origin, dz_grid)
end

#---------------------------------------------------------------------------------------------------

"""
    gg_fit_show_results(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)

Print a human-readable summary of a `gg_fit` `results`: the fit settings, the
per-plane weighted RMS residuals, and the leading multipoles at the central
plane as a quick sanity check.
"""
function gg_fit_show_results(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)
  println("="^72)
  println("GG fit:")
  println("  field grid        : ", join(size(field.magnetic), " x "), "  (ix, iy, iz)")
  println("  g_ref             : ", field.g_ref)
  println("  origin (x,y)      : ", params.origin)
  println("  n_planes_add      : ", params.n_planes_add, "   (max derivative order m_max = ", results.m_max, ")")
  println("  core_weight       : ", params.core_weight)
  println("  outer_plane_weight: ", params.outer_plane_weight)
  println("  # GG coefficients : ", length(results.params), " per plane")
  println("  # base planes     : ", length(results.z_base))
  println("-"^72)
  @printf("%-6s  %-12s  %-12s\n", "plane", "z [m]", "wRMS resid")
  for i in eachindex(results.z_base)
    @printf("%-6d  %-12.6g  %-12.4e\n", i, results.z_base[i], results.rms_plane[i])
  end
  println("-"^72)

  # Show the leading multipoles at the central plane for a quick sanity check.
  ic = cld(length(results.z_base), 2)
  println("Leading coefficients at central plane (z = ",
      @sprintf("%.6g", results.z_base[ic]), "):")
  for n in 1:6
    b00 = get(results.b, (n, 0), nothing)
    a00 = get(results.a, (n, 0), nothing)
    bstr = b00 === nothing ? "     -      " : @sprintf("% .6e", b00[ic])
    astr = a00 === nothing ? "     -      " : @sprintf("% .6e", a00[ic])
    @printf("  n=%-2d   b(n,0)=%s   a(n,0)=%s\n", n, bstr, astr)
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

    root datasets   : z_base, rms_plane, origin            (Float64[])
    root attributes : g_ref, dz_grid (Float64); m_max, n_planes_add (Int);
                      core_weight, outer_plane_weight (Float64)
    groups a, b     : n (Int[]), m (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{(n,m) => values[i,:]}
    group  bs       : m (Int[]), values (Float64[nkeys, nplanes])
                      -- reconstruct Dict{m => values[i,:]}
"""
function write_gg_fit(results::GGCoefs, field::FieldGridTable, params::GGFitInputParams)
  outfile = params.output_file
  h5open(outfile, "w") do f
    f["z_base"]    = collect(Float64, results.z_base)
    f["rms_plane"] = collect(Float64, results.rms_plane)
    f["origin"]    = collect(Float64, params.origin)
    attributes(f)["m_max"]              = Int(results.m_max)
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
