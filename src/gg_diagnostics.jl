#---------------------------------------------------------------------------------------------------
# Residual diagnostics for a gg_fit result.
#
# The per-plane wRMS printed by gg_fit_show_results says how large the residual
# is. It does not say what the residual is made of, which is the question that
# decides what to do about it: a residual the GG expansion could represent but
# the fit was not given enough terms for is fixed by enlarging the model, while a
# residual that is not a Maxwellian field at all is not fixable by any GG model.
#
# Three independent measurements separate those cases; see the docstring of
# `gg_fit_show_residuals` for how to read them together.
#---------------------------------------------------------------------------------------------------

"""
    _gg_taylor_getters(results, ip, dz) -> (aval, bval, bsval)

GG coefficient getters for base plane `ip`, Taylor-shifted to the longitudinal
offset `dz`: `aval(m, nd)` returns `a_m^[nd]` at `z_base[ip] + dz` as the fit
models it,

```
Σ_{j ≥ nd}  a(m,j)[ip] · dz^(j-nd) / (j-nd)!
```

and likewise for `b` and `b_s`. At `dz = 0` this is just the stored coefficient.

This is the same Taylor extrapolation the `gg_fit` design matrix uses to carry a
base plane's coefficients onto its neighbouring planes, so a residual built from
these getters is the residual the fit actually minimized.
"""
function _gg_taylor_getters(results::GGCoefs, ip::Integer, dz::Real)
  ndtop = max(results.nd_max,
              maximum((k[2] for k in keys(results.a)), init = 0),
              maximum((k[2] for k in keys(results.b)), init = 0),
              maximum(keys(results.bs), init = 0))

  function tsum(at, nd)
    nd < 0 && return 0.0
    dz == 0 && return at(nd)
    s = 0.0
    for j in nd:ndtop
      s += at(j) * dz^(j - nd) / factorial(j - nd)
    end
    return s
  end

  aval(m, nd) = tsum(j -> get(results.a, (m, j), nothing) === nothing ? 0.0 :
                          results.a[(m, j)][ip], nd)
  bval(m, nd) = tsum(j -> get(results.b, (m, j), nothing) === nothing ? 0.0 :
                          results.b[(m, j)][ip], nd)
  bsval(nd)   = tsum(j -> get(results.bs, j, nothing) === nothing ? 0.0 :
                          results.bs[j][ip], nd)
  return aval, bval, bsval
end

#---------------------------------------------------------------------------------------------------

"""
    gg_fit_residual_map(results::GGCoefs, field::FieldGridTable, plane::Integer;
                        dplane::Integer = 0) -> NamedTuple

Field table minus GG fit over the transverse grid of one plane — the data behind
one plane's RMS residual, and what to plot to see the shape of a bad fit.

- `plane` — 1-based index into `results.z_base`.
- `dplane` — grid planes away from that base plane, so `dplane = 0` (the default)
  is the base plane itself. A non-zero offset evaluates the fit by the same
  Taylor extrapolation `gg_fit` used when it fitted that neighbouring plane, so
  the map is comparable with the stored residual for any `n_planes_add`.

Returns `(; x, y, z, plane, dplane, origin, r_fit, B_table, B_fit, dB)`. `x` and
`y` are the absolute grid coordinates (vectors, whatever the grid's own index
range) and `z` the plane's longitudinal position. The three field arrays are
indexed `[ix, iy, component]` with component `1, 2, 3` = `Bx, By, Bs`, and
`dB = B_table - B_fit`. `r_fit` is the radius the diagnostics treat as the edge
of the fit region: `results.fit_radius_max` if the fit set one, otherwise the
largest circle inside the grid.

The map covers the whole grid either way — a residual outside the fit region is
worth seeing, it just is not something the fit was asked to make small.

```julia
r = gg_fit_residual_map(results, field, 12)
surface(r.x, r.y, r.dB[:, :, 1])       # Bx residual over the plane
```

Note that with `n_planes_add > 0` a base plane's stored `rms_weighted_plane`
covers its neighbouring planes too, so it will not equal the RMS of this one map.
"""
function gg_fit_residual_map(results::GGCoefs, field::FieldGridTable, plane::Integer;
                             dplane::Integer = 0)
  mag  = field.magnetic
  izs  = first(axes(mag, 3)):last(axes(mag, 3))
  1 <= plane <= length(izs) ||
      error("Plane $plane out of range: the fit has $(length(izs)) base planes.")
  iz = izs[plane] + dplane
  iz in izs || error("Plane $plane offset by dplane = $dplane leaves the grid, " *
                     "whose z indices are $izs.")

  ixs = first(axes(mag, 1)):last(axes(mag, 1))
  iys = first(axes(mag, 2)):last(axes(mag, 2))
  x   = [field.r0[1] + field.dr[1] * ix for ix in ixs]
  y   = [field.r0[2] + field.dr[2] * iy for iy in iys]
  dz  = dplane * field.dr[3]
  z   = results.z_base[plane] + dz

  # The fitted field at this plane is a fixed (x,y) polynomial per component, so
  # build the three coefficient arrays once and evaluate them over the grid.
  aval, bval, bsval = _gg_taylor_getters(results, plane, dz)
  g_ref = results.g_ref
  KB = (_comp_array(Bx_a, Bx_b, Bx_bs, aval, bval, bsval, g_ref),
        _comp_array(By_a, By_b, By_bs, aval, bval, bsval, g_ref),
        _comp_array(Bs_a, Bs_b, Bs_bs, aval, bval, bsval, g_ref))

  B_table = Array{Float64}(undef, length(x), length(y), 3)
  B_fit   = similar(B_table)
  for (i, ix) in enumerate(ixs), (j, iy) in enumerate(iys)
    B3 = mag[ix, iy, iz]
    xr = x[i] - results.origin[1]           # the expansion is written about the GG axis
    yr = y[j] - results.origin[2]
    for k in 1:3
      B_table[i, j, k] = B3[k]
      B_fit[i, j, k]   = _polyval(KB[k], xr, yr)[1]
    end
  end

  # r_fit is the boundary the diagnostics split on: the radius the fit was
  # restricted to, or else the largest circle the grid holds, which is as far out
  # as the expansion is well posed even when every point was fitted.
  ox, oy = results.origin[1], results.origin[2]
  r_fit  = results.fit_radius_max > 0 ? results.fit_radius_max :
           min(maximum(x) - ox, ox - minimum(x), maximum(y) - oy, oy - minimum(y))

  return (; x, y, z, plane, dplane, origin = copy(results.origin), r_fit,
            B_table, B_fit, dB = B_table .- B_fit)
end

#---------------------------------------------------------------------------------------------------

"""
    _radial_split(map) -> (r_fit, rms, rms_in, out_frac)

Split a residual map by radius about the GG expansion axis, at `map.r_fit`.

Field grids are rectangular and the GG expansion is a series in `r`, so the two
do not match: the corners of a square grid sit at `r = √2 · r_in`, where every
multipole is at its largest and the series is at its least convergent — a corner
point of a `±a` grid weighs a term of order `m` by `2^(m/2)` against the same term
at `r = a`. Those points are also the majority of a rectangular grid's area, so a
plane's RMS residual can be almost entirely made of them while the fit is good
everywhere the expansion is meant to be used. That is the case `fit_radius_max`
exists to remove, and when the fit set one, this splits there instead.

Returns the split radius, the RMS residual over the whole plane, the RMS over the
points inside it, and `out_frac`, the share of the total squared residual
contributed by the points outside it.
"""
function _radial_split(map)
  ox, oy = map.origin[1], map.origin[2]
  r_in = map.r_fit
  sin_ = sout = 0.0
  nin = nout = 0
  for i in eachindex(map.x), j in eachindex(map.y)
    v = sum(abs2, view(map.dB, i, j, :))
    if hypot(map.x[i] - ox, map.y[j] - oy) <= r_in * (1 + 1e-9)
      sin_ += v; nin += 1
    else
      sout += v; nout += 1
    end
  end
  tot = sin_ + sout
  return (r_in, sqrt(tot / (3 * (nin + nout))), nin == 0 ? NaN : sqrt(sin_ / (3nin)),
          tot == 0 ? 0.0 : sout / tot)
end

"""
    _resid_rough(d) -> Float64

Scale of the point-to-point irregular part of a residual map `d[ix, iy]`.

For a residual that is white noise of standard deviation `σ`, the second
difference `d[i-1] - 2d[i] + d[i+1]` has variance `6σ²`, so `mean(D²)/6` recovers
`σ²`. For a smooth residual the same second difference is `h²·∂²d` and is small
by two powers of the grid spacing. The estimate is taken along `x` and along `y`
and the smaller kept, so a residual that varies rapidly in one direction only is
not mistaken for noise.

Smooth structure does leak in — `h²·∂²d` is not zero — so this is an upper bound
on the noise, which is the safe direction: it can only understate how much of the
residual a larger model could remove.
"""
function _resid_rough(d)
  nx, ny = size(d)
  est = Float64[]
  if nx >= 3
    s = 0.0
    for j in 1:ny, i in 2:nx-1
      s += (d[i-1, j] - 2d[i, j] + d[i+1, j])^2
    end
    push!(est, s / (6 * (nx - 2) * ny))
  end
  if ny >= 3
    s = 0.0
    for i in 1:nx, j in 2:ny-1
      s += (d[i, j-1] - 2d[i, j] + d[i, j+1])^2
    end
    push!(est, s / (6 * nx * (ny - 2)))
  end
  return isempty(est) ? 0.0 : sqrt(minimum(est))
end

"""
    _interp2(v, x, y, px, py) -> Float64

Bilinear interpolation of `v[ix, iy]` sampled on the evenly spaced coordinate
vectors `x` and `y`, at the point `(px, py)`. Points outside the grid are clamped
to its edge; the callers only ask for points inside it.
"""
function _interp2(v, x, y, px, py)
  nx, ny = length(x), length(y)
  fx = nx == 1 ? 1.0 : 1 + (px - x[1]) / (x[2] - x[1])
  fy = ny == 1 ? 1.0 : 1 + (py - y[1]) / (y[2] - y[1])
  i  = clamp(floor(Int, fx), 1, max(nx - 1, 1))
  j  = clamp(floor(Int, fy), 1, max(ny - 1, 1))
  tx = nx == 1 ? 0.0 : clamp(fx - i, 0.0, 1.0)
  ty = ny == 1 ? 0.0 : clamp(fy - j, 0.0, 1.0)
  i2 = min(i + 1, nx)
  j2 = min(j + 1, ny)
  return (1 - tx) * (1 - ty) * v[i, j]  + tx * (1 - ty) * v[i2, j] +
         (1 - tx) * ty       * v[i, j2] + tx * ty       * v[i2, j2]
end

"""
    _azimuthal_harmonics(map, mmax; nth = 64, nrad = 5) -> NamedTuple or nothing

Azimuthal Fourier decomposition of a residual map, on circles centred on the GG
expansion axis.

The residual's transverse part is resolved into radial and azimuthal components
`B_r`, `B_θ` before the transform, because those are what a single multipole
makes clean. A missing multipole of order `m` has a scalar potential going as
`r^m·sin(mθ)`, hence

```
B_r, B_θ  ~  r^(m-1) · {sin,cos}(mθ)        B_s  ~  r^m · {sin,cos}(mθ)
```

— one azimuthal harmonic `m`, with a definite radial power. Noise has no
preferred harmonic and no radial growth. So both the harmonic *and* its measured
radial exponent have to line up before a residual can be blamed on a missing GG
term, and the exponent is the part that is hard to fake.

Returns `(; radii, amp, expo)`, where `amp[k][m+1, ir]` is the amplitude [T] of
harmonic `m` of kind `k` (`1, 2, 3` = `B_r, B_θ, B_s`) on the circle of radius
`radii[ir]`, and `expo[k][m+1]` is the exponent of the power law fitted through
`amp` by least squares in log-log, over the outer half of the radii. Returns
`nothing` when no circle centred on the axis fits inside the grid.

The circles stop at `map.r_fit`, so this sees only the part of the plane the
expansion is well posed on; `_radial_split` covers what happens outside it.
"""
function _azimuthal_harmonics(map, mmax::Integer; nth::Integer = 64, nrad::Integer = 5)
  x, y, dB = map.x, map.y, map.dB
  ox, oy   = map.origin[1], map.origin[2]
  # Stay inside both the fit region and the grid: outside either, the sampled
  # residual is not something the harmonics can be read as model error.
  rmax = min(map.r_fit, maximum(x) - ox, ox - minimum(x),
             maximum(y) - oy, oy - minimum(y))
  rmax > 0 || return nothing

  radii = [rmax * k / nrad for k in 1:nrad]
  amp   = [zeros(mmax + 1, nrad) for _ in 1:3]
  for (ir, r) in enumerate(radii)
    # Sample the residual around the circle, transverse part in (r, θ) components.
    vals = [zeros(nth) for _ in 1:3]
    for t in 1:nth
      th = 2π * (t - 1) / nth
      px, py = ox + r * cos(th), oy + r * sin(th)
      dx = _interp2(view(dB, :, :, 1), x, y, px, py)
      dy = _interp2(view(dB, :, :, 2), x, y, px, py)
      vals[1][t] = dx * cos(th) + dy * sin(th)
      vals[2][t] = -dx * sin(th) + dy * cos(th)
      vals[3][t] = _interp2(view(dB, :, :, 3), x, y, px, py)
    end
    # Direct transform: mmax is small, and this avoids an FFT dependency.
    for k in 1:3, m in 0:mmax
      c = sum(vals[k][t] * cis(-m * 2π * (t - 1) / nth) for t in 1:nth)
      amp[k][m+1, ir] = abs(c) * (m == 0 ? 1 : 2) / nth
    end
  end

  # Power law through each harmonic's radial profile, fitted over the outer half
  # of the circles only. On the inner ones a high-order multipole has fallen to
  # the level of everything else in the table, and including them biases the
  # exponent down towards 0 — exactly the direction that would make a real
  # multipole look like noise.
  expo = [fill(NaN, mmax + 1) for _ in 1:3]
  first_ir = max(1, cld(nrad, 2))
  for k in 1:3, m in 0:mmax
    use = [ir for ir in first_ir:nrad if amp[k][m+1, ir] > 0]
    length(use) >= 2 || continue
    lr = [log(radii[ir]) for ir in use]
    la = [log(amp[k][m+1, ir]) for ir in use]
    lrm, lam = sum(lr) / length(lr), sum(la) / length(la)
    den = sum((v - lrm)^2 for v in lr)
    den > 0 && (expo[k][m+1] = sum((lr[i] - lrm) * (la[i] - lam) for i in eachindex(lr)) / den)
  end

  return (; radii, amp, expo)
end

"""
    _maxwell_stats(Bm, B0, Bp, r0, dr, g_ref) -> (divB, curlB, zjump)

Measure a three-plane stack of field values against Maxwell's equations.
`Bm`, `B0`, `Bp` are the planes at `iz-1`, `iz`, `iz+1`, each indexed
`[ix, iy, component]`, `r0`/`dr` the grid origin and spacing.

`divB` and `curlB` are the RMS over the middle plane's interior points of `|∇·B|`
and `|∇×B|` as a fraction of the RMS of the field's own gradient there
(`sqrt(Σ_ij (∂B_i/∂u_j)²)`). Dividing by the gradient makes the numbers
dimensionless and comparable from plane to plane, but **the absolute level means
nothing on its own**: the same centred differences that measure the violation
truncate at `O(h²·∂³B)`, so a perfectly Maxwellian field that varies rapidly
across a cell produces a large ratio too. It has to be read against the same
statistic computed on a field known to be Maxwellian — which is what
`gg_fit_show_residuals` does, running this on the fitted GG field tabulated over
the same grid and reporting the table's level as a multiple of it.

`zjump` is the RMS over the plane of `|B[iz-1] - 2B[iz] + B[iz+1]|` [T], the
second difference along `z`. It is left unnormalized because it is read down the
column: a smoothly varying map gives `h²·∂²B` on every plane, so a plane whose
value stands orders of magnitude above its neighbours' is a seam between two
maps, a wrong `z` step, or a dropped plane.

The derivatives are taken in the curvilinear frame set by `g_ref` (scale factors
`1, 1, g` with `g = 1 + g_ref·x`):

```
∇·B    = (1/g)[∂(g·Bx)/∂x + ∂(g·By)/∂y + ∂Bs/∂s]
(∇×B)x = ∂Bs/∂y - (1/g)·∂By/∂s
(∇×B)y = (1/g)·∂Bx/∂s - (1/g)(g_ref·Bs + g·∂Bs/∂x)
(∇×B)s = ∂By/∂x - ∂Bx/∂y
```

In a current-free region both vanish for any physical field. A GG expansion is
Maxwellian by construction and so can only ever fit the part of a table that
obeys them: whatever the table violates them by is error no model can remove. A
large `∇×B` in particular says the region carries current — a coil inside the
grid — which the expansion has no way to represent.
"""
function _maxwell_stats(Bm, B0, Bp, r0, dr, g_ref)
  nx, ny = size(B0, 1), size(B0, 2)
  (nx >= 3 && ny >= 3) && return _maxwell_stats_inner(Bm, B0, Bp, r0, dr, g_ref)
  return (NaN, NaN, NaN)
end

function _maxwell_stats_inner(Bm, B0, Bp, r0, dr, g_ref)
  nx, ny = size(B0, 1), size(B0, 2)
  dx, dy, dz = dr
  sdiv = scurl = sjmp = sgrad = 0.0
  n = 0
  for i in 2:nx-1, j in 2:ny-1
    B  = @view B0[i, j, :]
    g  = 1 + g_ref * (r0[1] + dx * (i - 1))
    dBdx = (view(B0, i+1, j, :) .- view(B0, i-1, j, :)) ./ (2dx)
    dBdy = (view(B0, i, j+1, :) .- view(B0, i, j-1, :)) ./ (2dy)
    dBds = (view(Bp, i, j, :)   .- view(Bm, i, j, :))   ./ (2dz)

    # ∂/∂x of (g·B) picks up g' = g_ref; g does not depend on y or s.
    div  = (g_ref * B[1] + g * dBdx[1] + g * dBdy[2] + dBds[3]) / g
    curl = (dBdy[3] - dBds[2] / g,
            dBds[1] / g - (g_ref * B[3] + g * dBdx[3]) / g,
            dBdx[2] - dBdy[1])
    sdiv  += div^2
    scurl += curl[1]^2 + curl[2]^2 + curl[3]^2
    sjmp  += sum(abs2, view(Bp, i, j, :) .- 2 .* B .+ view(Bm, i, j, :))
    # The scale the violations are read against: the field's own gradient, which
    # is also what sets the truncation error of the differences above.
    sgrad += sum(abs2, dBdx) + sum(abs2, dBdy) + sum(abs2, dBds)
    n += 1
  end
  grad = sqrt(sgrad / n)
  grad > 0 || return (NaN, NaN, sqrt(sjmp / n))
  return (sqrt(sdiv / n) / grad, sqrt(scurl / n) / grad, sqrt(sjmp / n))
end

#---------------------------------------------------------------------------------------------------

"""
    gg_fit_show_residuals(results::GGCoefs, field::FieldGridTable;
                          planes = eachindex(results.z_base), detail = Int[],
                          mmax::Integer = 8)

Print what each plane's residual is made of, to separate a fit that is too small
from a field table a GG expansion cannot represent.

The residual is `field table - GG fit` over the transverse grid of a base plane
(`gg_fit_residual_map`). `planes` selects which base planes to report and
`detail` those to additionally print a full harmonic table for. `mmax` is the
highest azimuthal harmonic examined.

## The measurements

**Where on the plane the residual is.** A field grid is rectangular and the GG
expansion is a series in `r`, so the grid's corners stick out well beyond the
largest circle the expansion is well posed on — `√2` further out on a square
grid, where every multipole is at its largest. `in-rms` is the residual inside
that circle (or inside `fit_radius_max`, when the fit set one) and `out%` the
share of the squared residual coming from outside it. **Read this column first**:
an `out%` near 100 means the plane's RMS is a statement about the corners and not
about the fit, and no amount of model adjustment will fix it because it is the
series expansion itself running out of convergence there. That is what
`fit_radius_max` is for — with it set, `out%` describes points the fit was never
asked about, and `in-rms` is the residual that was actually minimized.

**Rough versus smooth.** The residual is split into a point-to-point irregular
part (`rough`, estimated from second differences, see `_resid_rough`) and what is
left over (`smooth`, from `rms² = rough² + smooth²`), and `rough%` is the first
as a percentage of the plane's rms. Interpolation noise in the field table is
rough; a GG term the fit does not have is smooth. **A residual that is mostly
rough will not improve no matter how many GG terms are added**; a residual that
is mostly smooth is a model that is missing something. Both are measured over the
largest centred block inside the inscribed circle, so the corners cannot
dominate the answer.

**Which term is missing.** For the smooth case, the residual is decomposed into
azimuthal harmonics on circles about the GG axis, with the transverse part
resolved into `B_r` and `B_θ` (see `_azimuthal_harmonics`). A missing multipole
of order `m` shows up as harmonic `m` growing as `r^(m-1)` in `B_r`/`B_θ` and
`r^m` in `B_s`. The `top harmonic` column names the largest one and the exponent
`p` measured for it; when `p` matches that expectation, the residual really is a
multipole the fit does not have, and raising `m_max` (or lifting an
`nd_max_for_m` cap, or un-excluding a function) will remove it. An exponent near
`0` means the harmonic does not grow with radius and is not a multipole at all.

**Whether the table can be fitted at all.** `∇·B` and `∇×B` measure the field
table against Maxwell's equations by centred differences, with no reference to
the fit. A GG expansion is Maxwellian by construction, so whatever the table
violates by is error no model can remove — and a large `∇×B` in particular means
current inside the grid, which the expansion cannot represent at all.

The catch is that those same differences truncate at `O(h²·∂³B)`, so a rapidly
varying but perfectly Maxwellian field also shows a violation, and the raw number
means nothing. The columns are therefore **ratios to the same statistic computed
on the fitted GG field tabulated over the same grid** — a field that satisfies
Maxwell exactly and has much the same structure, so what it shows is the
truncation floor for this grid and this field. `1.0` means the table is as
Maxwellian as the differencing can tell; a value of several means a violation the
grid is fine enough to see.

`d²B/dz²` is the second difference of the table along `z` in tesla, read down the
column rather than in absolute terms: a plane whose value stands orders of
magnitude above its neighbours' does not belong with them, which is a defect in
the map and not something to fit.

Reading them together: a high `out%` is a fit region the expansion cannot cover,
whatever else the row says; rough, with `∇·B`/`∇×B` well above 1, is a
table-quality problem; smooth with a harmonic whose exponent matches its order is
a model that needs that term; a `d²B/dz²` far above the neighbouring planes' is a
seam or a bad plane in the map.

The check none of this replaces is the direct one: refit with a larger `m_max`
and see whether the plane's residual actually falls. A residual that is missing
GG terms drops; one that is not saturates, and the saturated level is the honest
floor for that plane.
"""
function gg_fit_show_residuals(results::GGCoefs, field::FieldGridTable;
                               planes = eachindex(results.z_base), detail = Int[],
                               mmax::Integer = 8)
  kinds = ("B_r", "B_th", "B_s")
  println("="^108)
  println("GG fit residual diagnostics:  residual = field table - GG fit, on each base plane.")
  println("  in-rms, out%   : residual inside the fit radius (or the largest circle the ",
          "grid holds), and the")
  println("                   share of the total squared residual from the corners ",
          "outside it, where the")
  println("                   series converges worst")
  println("  rough%         : irregular, unfittable share of the residual inside that ",
          "circle; the rest is")
  println("                   smooth, and smooth is what a bigger model can remove")
  println("  top harmonic   : largest azimuthal harmonic m of the residual, ",
          "p = its radial exponent")
  println("                   a missing multipole m has p = m-1 in B_r/B_th, ",
          "p = m in B_s; noise has p ~ 0")
  println("  divB, curlB    : the table's Maxwell violation as a multiple of the ",
          "fitted GG field's own,")
  println("                   that field being Maxwellian by construction.  ",
          "1 = as clean as this grid")
  println("                   can show; several = a real violation, and a floor ",
          "no GG fit beats")
  println("  d2B/dz2        : second difference of the table along z [T] -- ",
          "flags a plane out of line with its neighbours")
  println("-"^108)
  @printf("%-6s %-8s %-11s %-11s %-6s %-7s %-18s %-8s %-8s %-11s\n",
          "plane", "z [m]", "rms [T]", "in-rms [T]", "out%", "rough%", "top harmonic",
          "divB x", "curlB x", "d2B/dz2 [T]")

  # The Maxwell columns difference across the neighbouring planes, so every map
  # is needed; build them once rather than three times per reported plane.
  nplane = length(results.z_base)
  maps   = [gg_fit_residual_map(results, field, p) for p in 1:nplane]
  for p in planes
    map = maps[p]
    r_in, rms, rms_in, out_frac = _radial_split(map)
    # Roughness is measured on the largest centred block that fits inside the
    # inscribed circle, and reported against that block's own rms: the corners
    # would otherwise dominate both, and they are not the region in question.
    half = r_in / sqrt(2)
    bi = [i for i in eachindex(map.x) if abs(map.x[i] - map.origin[1]) <= half * (1 + 1e-9)]
    bj = [j for j in eachindex(map.y) if abs(map.y[j] - map.origin[2]) <= half * (1 + 1e-9)]
    blk = view(map.dB, bi, bj, :)
    rms_blk = sqrt(sum(abs2, blk) / length(blk))
    rough   = sqrt(sum(_resid_rough(view(blk, :, :, k))^2 for k in 1:3) / 3)

    har = _azimuthal_harmonics(map, mmax)
    top = "-"
    if har !== nothing
      # Largest harmonic on the outermost circle, across all three kinds.
      best = (0, 0, -Inf)
      for k in 1:3, m in 0:mmax
        har.amp[k][m+1, end] > best[3] && (best = (k, m, har.amp[k][m+1, end]))
      end
      k, m, _ = best
      top = @sprintf("%-5s m=%-2d p=%4.1f", kinds[k], m, har.expo[k][m+1])
    end

    # The fitted field, tabulated on the same grid and differenced the same way,
    # calibrates out the truncation error the differences themselves carry.
    dvr = clr = zj = NaN
    if 1 < p < nplane
      dvt, clt, zj = _maxwell_stats(maps[p-1].B_table, map.B_table, maps[p+1].B_table,
                                    field.r0, field.dr, field.g_ref)
      dvf, clf, _  = _maxwell_stats(maps[p-1].B_fit, map.B_fit, maps[p+1].B_fit,
                                    field.r0, field.dr, field.g_ref)
      dvr, clr = dvt / dvf, clt / clf
    end
    @printf("%-6d %-8.6g %-11.4e %-11.4e %-6.1f %-7.1f %-18s %-8.2f %-8.2f %-11.3e\n",
            p, map.z, rms, rms_in, 100out_frac, 100 * rough / rms_blk, top, dvr, clr, zj)
  end

  for p in detail
    map = maps[p]
    har = _azimuthal_harmonics(map, mmax)
    println("-"^108)
    r_in = map.r_fit
    @printf("Plane %d (z = %.6g): residual by radius (r = %.4g is the edge of the fit region).\n",
            p, map.z, r_in)
    @printf("  %-16s %-12s %-12s %-12s %-8s\n", "r band [m]", "rms Bx", "rms By", "rms Bs", "points")
    edges = [r_in * k / 4 for k in 0:4]
    push!(edges, hypot(maximum(abs, map.x), maximum(abs, map.y)) + 1)
    for ib in 1:length(edges)-1
      lo, hi = edges[ib], edges[ib+1]
      s = zeros(3); n = 0
      for i in eachindex(map.x), j in eachindex(map.y)
        rr = hypot(map.x[i] - map.origin[1], map.y[j] - map.origin[2])
        lo <= rr < hi || continue
        n += 1
        for k in 1:3; s[k] += map.dB[i, j, k]^2; end
      end
      n == 0 && continue
      @printf("  %-16s %-12.4e %-12.4e %-12.4e %-8d\n",
              ib == length(edges)-1 ? @sprintf("> %.4g (corners)", lo) :
                                      @sprintf("%.4g - %.4g", lo, hi),
              sqrt(s[1]/n), sqrt(s[2]/n), sqrt(s[3]/n), n)
    end
    println("  Azimuthal harmonics of the residual, amplitude [T] on circles about the GG axis.")
    har === nothing && (println("  No circle about the GG axis fits inside the grid."); continue)
    println("  p = radial exponent, fitted over the outer half of the radii.")
    for k in 1:3
      @printf("  %-5s %-4s %s  %8s\n", kinds[k], "m",
              join((@sprintf("%-11s", @sprintf("r=%.4g", r)) for r in har.radii), " "), "p")
      for m in 0:mmax
        maximum(har.amp[k][m+1, :]) == 0 && continue
        @printf("  %-5s %-4d %s  %8.2f\n", "", m,
                join((@sprintf("%-11.3e", har.amp[k][m+1, ir])
                      for ir in eachindex(har.radii)), " "), har.expo[k][m+1])
      end
    end
  end
  println("="^108)
  return nothing
end
