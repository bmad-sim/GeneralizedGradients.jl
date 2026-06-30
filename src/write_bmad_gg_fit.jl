# ---------------------------------------------------------------------------
# write_bmad_gg_fit.jl
#
# Convert generalized-gradient (GG) coefficients produced by `gg_fit` into Bmad
# `gen_grad_map` format (a lattice element with the GG map attached).
# `write_bmad_gg_fit` is the public function; programs/run_write_bmad_gg_fit.jl is a
# shell wrapper.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coefficient recursion: project a_n/b_n/b_s  ->  Bmad C_{m,α} derivative towers
# ---------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

"""
    write_bmad_gg_fit(fit::GGCoefs, meta; ele_name, output_base, cutoff) -> lattice_file_path
    write_bmad_gg_fit(input::AbstractString; output_base, cutoff) -> lattice_file_path

Convert generalized-gradient (GG) coefficients produced by `gg_fit` into Bmad
`gen_grad_map` format, producing a Bmad lattice element with the GG map attached.
Returns the path of the lattice-element file.

The GG fit is supplied either as a loaded result (`fit`, `meta` as returned by
`read_gg_fit`) or as the path to a gg_fit HDF5 file (output of `write_gg_fit`),
which is read with `read_gg_fit`.

## Usage

```
using GeneralizedGradients
write_bmad_gg_fit("gg_fit_result.h5")
```
From the shell (see `programs/run_write_bmad_gg_fit.jl`):
```
julia programs/run_write_bmad_gg_fit.jl <gg_fit_result.h5> [output_base] [cutoff]
```

Keyword arguments:
- `ele_name` — name of the Bmad lattice element. Default `"gen_grad_ele"` (or, for
  a file input, the `output_base` basename).
- `output_base` — base name for the two output files: `<output_base>.bmad` (the
  lattice element) and `<output_base>_gg.bmad` (the attached `gen_grad_map`).
  Defaults to the input file name without extension, or `ele_name` for an
  in-memory fit.
- `cutoff` — relative magnitude cutoff for pruning negligible multipole curves.
  A curve is dropped if its peak `|GG|` is below `cutoff * (largest peak |GG| of
  any curve)`. Default `0` (keep every non-zero curve).

The reference-coordinates bending "strength" `1/bend_radius` [1/m] is taken from
`fit.g_ref`; non-zero => the element is an `sbend` with `curved_ref_frame = T`,
otherwise an `em_field`.

## Background: the two GG conventions

This project (Van der Schueren / Sagan) characterizes the field by midplane-
derivative generalized gradients `a_n(s)`, `b_n(s)`, `b_s(s)`:

```
B_x(x,0,s) = Σ_{n≥1} a_n(s) x^{n-1}/(n-1)!     (skew / "cos" family)
B_y(x,0,s) = Σ_{n≥1} b_n(s) x^{n-1}/(n-1)!     (normal / "sin" family)
B_s(0,0,s) = b_s(s)                            (solenoidal, m = 0)
```

Bmad's `gen_grad_map` (Venturini-Dragt) instead uses azimuthal-harmonic
gradients `C_{m,α}(z)`, `α ∈ {sin, cos}`, where the field is (Sagan, IPAC23 Eq. 4)

```
B_ρ = Σ_{m≥1,n} f(m,n)(2n+m) ρ^{2n+m-1}[C^{[2n]}_{m,s} sin mθ + C^{[2n]}_{m,c} cos mθ]
      + Σ_{n≥1} f(0,n)(2n) ρ^{2n-1} C^{[2n]}_{0,c}
B_θ = Σ_{m≥1,n} f(m,n) m ρ^{2n+m-1}[C^{[2n]}_{m,s} cos mθ - C^{[2n]}_{m,c} sin mθ]
B_z = Σ_{m≥0,n} f(m,n) ρ^{2n+m}[C^{[2n+1]}_{m,s} sin mθ + C^{[2n+1]}_{m,c} cos mθ]
```

with `f(m,n) = (-1)^n m!/(4^n n!(n+m)!)`, sin = normal, cos = skew.

Equating the two on the midplane gives the exact relations used here. For each
azimuthal `m` and derivative order `j` (with `k ≡ m`):

```
C^{[j]}_{m,s} = (1/m!)[ b^{[j]}_m - (m-1)! Σ_{n≥1, m-2n≥1} Wn(m,n) C^{[j+2n]}_{m-2n,s} ]
C^{[j]}_{m,c} = (1/m!)[ a^{[j]}_m - (m-1)! Σ_{n≥1, m-2n≥1} Wc(m,n) C^{[j+2n]}_{m-2n,c}
                                 - (m even) (m-1)! Us(m) b_s^{[m+j-1]} ]
C^{[j]}_{0,c} = b_s^{[j-1]}                                            (j ≥ 1)

Wn(m,n) = (-1)^n (m-2n)!(m-2n)/(4^n n!(m-n)!)     (normal radial mixing)
Wc(m,n) = (-1)^n (m-2n)! m   /(4^n n!(m-n)!)       (skew radial mixing)
Us(m)   = (-1)^{m/2} m /(4^{m/2} ((m/2)!)^2)        (skew↔solenoid coupling)
```

where `x^{[j]} ≡ dʲx/dsʲ` is supplied directly by the fit (`a[(n,j)]`,
`b[(n,j)]`, `bs[j]`). These recursions are solved in order of increasing `m`,
reusing the lower-`m` towers. Truncation at the fit's maximum derivative order
`m_max` bounds the radial-correction sums exactly as the fit itself is bounded,
so the resulting `gen_grad_map` reproduces the project field to machine precision.

## Output

A Bmad `gen_grad_map` (`field_type = magnetic`) attached to a lattice element.
As with grid fields, the map is anchored at the entrance of the element
(`ele_anchor_pt = beginning`), z-positions run `0, dz, 2dz, …` and the element
length is `L = (n_planes - 1) * dz`. The transverse anchor `r0` is the GG
expansion axis (`origin`). For a curved reference (`g_ref ≠ 0`) the element is an
`sbend` with `g = g_ref` and `curved_ref_frame = T`; otherwise it is an `em_field`.
"""
function write_bmad_gg_fit(input::AbstractString;
                output_base::AbstractString =
                    joinpath(dirname(input), first(splitext(basename(input)))),
                cutoff::Real = 0.0)
  fit, meta = read_gg_fit(input)
  return write_bmad_gg_fit(fit, meta; ele_name = basename(output_base), output_base, cutoff)
end

function write_bmad_gg_fit(fit::GGCoefs, meta;
                ele_name::AbstractString = "gen_grad_ele",
                output_base::AbstractString = ele_name,
                cutoff::Real = 0.0)

  g_ref = fit.g_ref
  cs, cc, c0c, npl, mmax, kmax = gg_to_bmad_curves(fit, meta)
  dz = meta.dz_grid
  L  = (npl - 1) * dz
  is_bend = g_ref != 0

  # Decide which curves to emit (prune negligible multipoles).
  gpeak = maximum(vcat(0.0, [_peak(cs, m) for m in 1:kmax], [_peak(cc, m) for m in 1:kmax]))
  thresh = cutoff * gpeak
  has_sol = haskey(c0c, 0) && maximum(abs, c0c[0]) > 0 ||
            any(haskey(c0c, j) && maximum(abs, c0c[j]) > 0 for j in 1:mmax+1)
  keep(d, m) = (v = get(d, (m, 0), nothing); v !== nothing && maximum(abs, v) > thresh)

  map_file = output_base * "_gg.bmad"
  ele_file = output_base * ".bmad"
  map_name = basename(map_file)

  # Emit one derivs table.  tower(j) returns the per-plane vector for order j,
  # listed for derivative orders 0 … nder.  The solenoid (m = 0) carries one
  # extra order because C^{[j]}_{0,c} = b_s^{[j-1]} reaches index m_max + 1.
  function write_curve(io, m, kind, nder, tower)
    println(io, "  curve = {")
    println(io, "    m = ", m, ",")
    println(io, "    kind = ", kind, ",")
    println(io, "    derivs = {")
    for i in 1:npl
      vals = join((_gg_num(tower(j)[i]) for j in 0:nder), " ")
      @printf(io, "      %s: %s,\n", _gg_num((i - 1) * dz), vals)
    end
    println(io, "    }")
    println(io, "  },")
  end

  open(map_file, "w") do io
    println(io, "{")
    println(io, "  field_type = magnetic,")
    println(io, "  ele_anchor_pt = beginning,")
    is_bend && println(io, "  curved_ref_frame = T,")
    println(io, "  r0 = (", _gg_num(meta.origin[1]), ", ", _gg_num(meta.origin[2]), ", 0),")
    println(io, "  dz = ", _gg_num(dz), ",")

    # Solenoid first (m = 0, cos), then normal+skew for each m.
    if has_sol
      sol(j) = get(c0c, j, zeros(Float64, npl))
      write_curve(io, 0, "cos", mmax + 1, sol)
    end
    for m in 1:kmax
      keep(cs, m) && write_curve(io, m, "sin", mmax, j -> get(cs, (m, j), zeros(Float64, npl)))
      keep(cc, m) && write_curve(io, m, "cos", mmax, j -> get(cc, (m, j), zeros(Float64, npl)))
    end
    println(io, "}")
  end

  open(ele_file, "w") do io
    println(io, "! Bmad lattice element with attached generalized-gradient map.")
    println(io, "! Generated from gg_fit GG coefficients by write_bmad_gg_fit.")
    println(io, "!")
    if is_bend
      println(io, "! Reference curve is an arc (g = ", _gg_num(g_ref),
            " 1/m) => sbend; GGs are in the bend curvilinear frame.")
      println(io)
      println(io, ele_name, ": sbend,")
      println(io, "  l = ", _gg_num(L), ",")
      println(io, "  g = ", _gg_num(g_ref), ",")
    else
      println(io, "! Reference curve is straight => em_field.")
      println(io)
      println(io, ele_name, ": em_field,")
      println(io, "  l = ", _gg_num(L), ",")
    end
    println(io, "  field_calc = fieldmap,")
    println(io, "  tracking_method = runge_kutta,")
    println(io, "  mat6_calc_method = tracking,")
    println(io, "  gen_grad_map = call::", map_name)
  end

  return ele_file
end

#---------------------------------------------------------------------------------------------------

"""
    gg_to_bmad_curves(fit, meta) -> (cs, cc, c0c, nplanes, m_max, kmax)

Compute the Bmad azimuthal-harmonic GG derivative towers from a loaded
`gg_fit` result (`fit`, `meta` as returned by `read_gg_fit`). Returns

```
cs[(m,j)]  :: Vector  -- C^{[j]}_{m,sin}(plane)  (normal multipole m)
cc[(m,j)]  :: Vector  -- C^{[j]}_{m,cos}(plane)  (skew multipole m)
c0c[j]     :: Vector  -- C^{[j]}_{0,cos}(plane)  (solenoid, j ≥ 1)
```

each a per-plane vector, for `j = 0 … m_max`.
"""
function gg_to_bmad_curves(fit, meta)
  mmax = fit.m_max
  npl  = length(fit.z_base)
  kmax = maximum(first.(keys(fit.b)))
  Z()  = zeros(Float64, npl)

  # Fit getters (missing entries are treated as identically zero).
  bget(k, j)  = (0 <= j <= mmax && haskey(fit.b, (k, j)))  ? fit.b[(k, j)]  : nothing
  aget(k, j)  = (0 <= j <= mmax && haskey(fit.a, (k, j)))  ? fit.a[(k, j)]  : nothing
  bsget(j)    = (0 <= j <= mmax && haskey(fit.bs, j))      ? fit.bs[j]      : nothing

  Wn(k, n) = (-1.0)^n * _fac(k - 2n) * (k - 2n) / (4.0^n * _fac(n) * _fac(k - n))
  Wc(k, n) = (-1.0)^n * _fac(k - 2n) * k         / (4.0^n * _fac(n) * _fac(k - n))
  Us(k)    = (-1.0)^(k ÷ 2) * k / (4.0^(k ÷ 2) * _fac(k ÷ 2)^2)

  cs = Dict{Tuple{Int,Int},Vector{Float64}}()   # normal (sin)
  cc = Dict{Tuple{Int,Int},Vector{Float64}}()   # skew (cos)

  for k in 1:kmax, j in 0:mmax
    # Normal family.
    bj = bget(k, j)
    if bj !== nothing
      acc = copy(bj)
      n = 1
      while k - 2n >= 1
        lo = get(cs, (k - 2n, j + 2n), nothing)
        lo !== nothing && (acc .-= _fac(k - 1) * Wn(k, n) .* lo)
        n += 1
      end
      cs[(k, j)] = acc ./ _fac(k)
    end
    # Skew family.
    aj = aget(k, j)
    if aj !== nothing
      acc = copy(aj)
      n = 1
      while k - 2n >= 1
        lo = get(cc, (k - 2n, j + 2n), nothing)
        lo !== nothing && (acc .-= _fac(k - 1) * Wc(k, n) .* lo)
        n += 1
      end
      if iseven(k)
        bsd = bsget(k + j - 1)          # = C^{[k+j]}_{0,c}
        bsd !== nothing && (acc .-= _fac(k - 1) * Us(k) .* bsd)
      end
      cc[(k, j)] = acc ./ _fac(k)
    end
  end

  # Solenoid (m = 0, cos): derivatives are exact; the value column (j = 0) is a
  # cubic-Hermite cumulative integral of b_s (it does not affect the m = 0
  # field but is used by Bmad's interpolating spline, so it must be consistent).
  c0c = Dict{Int,Vector{Float64}}()
  if bsget(0) !== nothing
    for j in 1:mmax+1
      d = bsget(j - 1)
      d !== nothing && (c0c[j] = copy(d))
    end
    val = Z()
    bs0 = bsget(0)                          # b_s = C'_{0,c}
    bs1 = bsget(1)                          # b_s'
    dz  = meta.dz_grid
    for i in 2:npl
      # ∫ over one plane of the cubic-Hermite of C'_{0,c} = b_s.
      incr = 0.5 * dz * (bs0[i-1] + bs0[i])
      bs1 !== nothing && (incr += dz^2 / 12 * (bs1[i-1] - bs1[i]))
      val[i] = val[i-1] + incr
    end
    c0c[0] = val
  end

  return cs, cc, c0c, npl, mmax, kmax
end


