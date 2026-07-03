# ---------------------------------------------------------------------------
# low_level.jl
#
# Low-level, underscore-prefixed helper functions for the package: monomial
# coefficient assembly and polynomial evaluation, Hermite/Taylor interpolation
# of the GG derivative towers, the compiled fast-evaluation plan build/eval used
# by `field_and_potential_evaluate_at`, and the HDF5 / field-grid / Bmad
# read-write helpers.
#
# The `GGEvalPlan`, `_Tower` and `_CompTerms` types are defined in struct.jl (so
# `GGCoefs` can hold a plan in its `eval_plan` field); the functions below build
# and evaluate them (see the "Fast evaluation" section).
# ---------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

"""
    _accum(tdict, valfun, g_ref) -> K

Coefficient-array builder. `K[p+1,q+1]` = coefficient of `xᵖ yᵠ`.
`valfun(key)` returns the GG function value multiplying that table entry.
"""
function _accum(tdict, valfun, g_ref)
  K = zeros(Float64, _NMAX, _NMAX)
  for (key, terms) in tdict
    v = valfun(key)
    v == 0.0 && continue
    for (c, p, q, k) in terms
      K[p+1, q+1] += float(c) * (k == 0 ? 1.0 : float(g_ref)^k) * v
    end
  end
  return K
end

#---------------------------------------------------------------------------------------------------

"""
    _comp_array(Ta, Tb, Tbs, aval, bval, bsval, g_ref) -> K

Combined coefficient array of a component: sum of its `a`, `b` and `bs` parts.
`Ta`/`Tb` are keyed by `(n,m)` and `Tbs` by `m`.
"""
function _comp_array(Ta, Tb, Tbs, aval, bval, bsval, g_ref)
  return _accum(Ta, k -> aval(k...), g_ref) .+
         _accum(Tb, k -> bval(k...), g_ref) .+
         _accum(Tbs, m -> bsval(m), g_ref)
end

#---------------------------------------------------------------------------------------------------

"""
    _polyval(K, x, y) -> (val, dvx, dvy)

Value and `(x,y)` partials of the plain polynomial `Σ K[i,j] xⁱ yʲ`.
"""
function _polyval(K, x, y)
  val = 0.0; dvx = 0.0; dvy = 0.0
  for i in 0:_NMAX-1, j in 0:_NMAX-1
    c = K[i+1, j+1]
    c == 0.0 && continue
    xi = x^i; yj = y^j
    val += c * xi * yj
    i > 0 && (dvx += c * i * x^(i-1) * yj)
    j > 0 && (dvy += c * j * xi * y^(j-1))
  end
  return val, dvx, dvy
end

#---------------------------------------------------------------------------------------------------

"""
    _taylor_derivs(z0, f0, sq) -> Vector

Single-point Taylor tower: from `f` and its derivatives at `z0`, return
`[P⁽ᵐ⁾(sq) for m=0..N]` with `P` the Taylor series, i.e. `f` extrapolated to `sq`.
"""
function _taylor_derivs(z0, f0, sq)
  N = length(f0) - 1
  u = sq - z0
  out = zeros(Float64, N + 1)
  for m in 0:N
    acc = 0.0
    for j in m:N
      acc += f0[j+1] * u^(j-m) / factorial(j-m)
    end
    out[m+1] = acc
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _hermite_derivs(zL, zR, fL, fR, sq) -> Vector

Two-point Hermite tower: `fL[j+1] = f⁽ʲ⁾(zL)`, `fR[j+1] = f⁽ʲ⁾(zR)`, `j = 0..N`.
Returns `[H⁽ᵐ⁾(sq) for m=0..N]` where `H` is the degree-`(2N+1)` Hermite
interpolant. Built via confluent Newton divided differences in the local
coordinate `u = s - zL` (nodes: `0` with multiplicity `N+1`, `hstep` with
multiplicity `N+1`).
"""
function _hermite_derivs(zL, zR, fL, fR, sq)
  N = length(fL) - 1
  K = 2N + 1                       # polynomial degree = (#nodes) - 1
  hstep = zR - zL
  fval(i, j) = (i <= N ? fL : fR)[j+1]    # j-th derivative at node i's plane
  isR(i) = i > N                          # node i (0-based) in the R block?

  memo = Dict{Tuple{Int,Int},Float64}()
  function dd(i, k)                # divided difference f[t_i, …, t_{i+k}]
    haskey(memo, (i, k)) && return memo[(i, k)]
    v = isR(i) == isR(i + k) ? fval(i, k) / factorial(k) :       # one block: f⁽ᵏ⁾/k!
      (dd(i + 1, k - 1) - dd(i, k - 1)) / hstep           # spans both blocks
    memo[(i, k)] = v
    return v
  end

  c = [dd(0, k) for k in 0:K]                 # Newton coefficients
  tnode(i) = i <= N ? 0.0 : hstep             # node positions in u

  # Accumulate the Newton form into monomial coefficients in u.
  poly  = zeros(Float64, K + 1)               # poly[d+1] = coeff of u^d
  basis = [1.0]                               # current ∏ (u - t_j)
  for k in 0:K
    @inbounds for d in 1:length(basis)
      poly[d] += c[k+1] * basis[d]
    end
    if k < K                                # multiply basis by (u - t_k)
      tk = tnode(k)
      nb = zeros(Float64, length(basis) + 1)
      @inbounds for d in 1:length(basis)
        nb[d+1] += basis[d]
        nb[d]   -= tk * basis[d]
      end
      basis = nb
    end
  end

  # Evaluate H and its derivatives at uq.
  uq = sq - zL
  out = zeros(Float64, N + 1)
  for m in 0:N
    acc = 0.0
    for d in m:K
      ff = 1.0                            # falling factorial d·(d-1)···(d-m+1)
      for r in 0:m-1
        ff *= (d - r)
      end
      acc += poly[d+1] * ff * uq^(d-m)
    end
    out[m+1] = acc
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_tower(fL, fR, zL, zR, sq, single) -> Vector

Interpolate one GG function's derivative tower onto `sq` (Hermite, or Taylor if `single`).
"""
_interp_tower(fL, fR, zL, zR, sq, single) =
  single ? _taylor_derivs(zL, fL, sq) : _hermite_derivs(zL, zR, fL, fR, sq)

#---------------------------------------------------------------------------------------------------

"""
    _contiguous_order(orders) -> N

Largest `N` such that orders `0,1,…,N` are all present in `orders` (sorted).
"""
function _contiguous_order(orders)
  N = -1
  for (idx, m) in enumerate(orders)
    m == idx - 1 ? (N = m) : break
  end
  return N
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_nm_dict(d, iL, iR, zL, zR, sq, single) -> Dict

Interpolate an `(n,m)`-keyed dict (`a`, `b`): build one Hermite per multipole `n`.
"""
function _interp_nm_dict(d, iL, iR, zL, zR, sq, single)
  out = Dict{Tuple{Int,Int},Vector{Float64}}()
  byn = Dict{Int,Vector{Int}}()
  for (n, m) in keys(d)
    push!(get!(byn, n, Int[]), m)
  end
  for (n, ms) in byn
    sort!(ms)
    N  = _contiguous_order(ms)
    fL = [d[(n, j)][iL] for j in 0:N]
    fR = [d[(n, j)][iR] for j in 0:N]
    vals = _interp_tower(fL, fR, zL, zR, sq, single)
    for j in 0:N
      out[(n, j)] = [vals[j+1]]
    end
    for m in ms                              # any non-contiguous order: nearest plane
      m > N && (out[(n, m)] = [d[(n, m)][iL]])
    end
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_m_dict(d, iL, iR, zL, zR, sq, single) -> Dict

Interpolate an `m`-keyed dict (`bs`): a single Hermite tower.
"""
function _interp_m_dict(d, iL, iR, zL, zR, sq, single)
  out = Dict{Int,Vector{Float64}}()
  ms  = sort(collect(keys(d)))
  N   = _contiguous_order(ms)
  fL  = [d[j][iL] for j in 0:N]
  fR  = [d[j][iR] for j in 0:N]
  vals = _interp_tower(fL, fR, zL, zR, sq, single)
  for j in 0:N
    out[j] = [vals[j+1]]
  end
  for m in ms
    m > N && (out[m] = [d[m][iL]])
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _interp_gg_fit(fit, s::Real) -> fit::GGCoefs

Take GG fit results `fit` which give the GG functions at a set of planes and
return a similar `GGCoefs` but with one plane: the GG coefficients for that
plane are the interpolated GG coefficients at the given `s`-position.

- `fit` — GG coefficients for all planes.

Builds a single virtual plane at `s` by Hermite-interpolating every GG derivative
tower from the two straddling grid planes (one-plane Taylor if only one plane).
"""
function _interp_gg_fit(fit, s::Real)
  z  = fit.z_base
  P  = length(z)
  sq = float(s)

  if P == 1
    iL = iR = 1
  else
    i0 = searchsortedlast(z, sq)             # z[i0] <= s < z[i0+1]
    iL = clamp(i0, 1, P - 1)                 # straddling pair (extrapolates at ends)
    iR = iL + 1
  end
  single = iL == iR
  zL = z[iL]; zR = z[iR]

  a2  = _interp_nm_dict(fit.a,  iL, iR, zL, zR, sq, single)
  b2  = _interp_nm_dict(fit.b,  iL, iR, zL, zR, sq, single)
  bs2 = _interp_m_dict(fit.bs, iL, iR, zL, zR, sq, single)

  fit2 = GGCoefs(; z_base = [sq], a = a2, b = b2, bs = bs2,
                        m_max = fit.m_max, rms_plane = [NaN], g_ref = fit.g_ref,
                        origin = fit.origin, dz_grid = fit.dz_grid)
  return fit2
end

#---------------------------------------------------------------------------------------------------

"""
    _field_CB(fit, ip::Integer) -> (CBx, CBy, CBs)

Field-expansion coefficients `B_c(x,y,s) = Σ_{i,j} CB_{c,i,j}(s) xⁱ yʲ`.
Returns full `_NMAX×_NMAX` arrays summed over the `a`, `b`, `bs` parts.
"""
function _field_CB(fit, ip::Integer)
  g_ref = fit.g_ref
  aval(n, m) = (m >= 0 && haskey(fit.a, (n, m))) ? fit.a[(n, m)][ip] : 0.0
  bval(n, m) = (m >= 0 && haskey(fit.b, (n, m))) ? fit.b[(n, m)][ip] : 0.0
  bsval(m)   = (m >= 0 && haskey(fit.bs, m))     ? fit.bs[m][ip]     : 0.0
  CBx = _accum(Bx_a, k -> aval(k...), g_ref) .+ _accum(Bx_b, k -> bval(k...), g_ref) .+ _accum(Bx_bs, m -> bsval(m), g_ref)
  CBy = _accum(By_a, k -> aval(k...), g_ref) .+ _accum(By_b, k -> bval(k...), g_ref) .+ _accum(By_bs, m -> bsval(m), g_ref)
  CBs = _accum(Bs_a, k -> aval(k...), g_ref) .+ _accum(Bs_b, k -> bval(k...), g_ref) .+ _accum(Bs_bs, m -> bsval(m), g_ref)
  return CBx, CBy, CBs
end

#---------------------------------------------------------------------------------------------------

"""
    _trim3(CBx, CBy, CBs) -> (CBx, CBy, CBs)

Trim three coefficient arrays to the smallest `(x,y)` extent holding every
nonzero entry, so the returned matrices are indexed `CB[i+1, j+1] = CB_{c,i,j}`.
"""
function _trim3(CBx, CBy, CBs)
  pmax = 1; qmax = 1
  for K in (CBx, CBy, CBs), j in 1:_NMAX, i in 1:_NMAX
    if K[i, j] != 0.0
      pmax = max(pmax, i); qmax = max(qmax, j)
    end
  end
  return CBx[1:pmax, 1:qmax], CBy[1:pmax, 1:qmax], CBs[1:pmax, 1:qmax]
end

#---------------------------------------------------------------------------------------------------

"""
    _coefsum(terms, x::Float64, y::Float64, g_ref)

CB coefficient sum: `Σ coeff·g_ref^k·x^p·y^q` over the table entries for one
`(component, function)` — one entry of the CB grids built in `gg_fit`.
"""
function _coefsum(terms, x::Float64, y::Float64, g_ref)
  s = 0.0
  for (c, p, q, k) in terms
    hk = k == 0 ? 1.0 : float(g_ref)^k
    s += float(c) * hk * x^p * y^q
  end
  return s
end

#---------------------------------------------------------------------------------------------------

"""
    _gg_num(x::Real) -> String

Lossless, compact `Float64` text: `repr` emits the shortest string that parses
back to the identical `Float64` (Bmad's Fortran reader accepts the e-notation).
Without this, cancellation in `B_s` (which is a small difference of larger terms)
magnifies the rounding of a fixed-precision format.
"""
_gg_num(x::Real) = iszero(x) ? "0" : repr(float(x))

#---------------------------------------------------------------------------------------------------

"""
    _peak(d, m) -> Float64

Peak `|value|` of a derivative tower's value column (`j = 0`), used for cutoffs.
"""
_peak(d, m) = (v = get(d, (m, 0), nothing); v === nothing ? 0.0 : maximum(abs, v))

#---------------------------------------------------------------------------------------------------

"""
    _write_field_component_jl(io, name, field)

Write the `fg.<name>` OffsetArray of `[Bx,By,Bz]` 3-vectors as include-able Julia.
"""
function _write_field_component_jl(io, name, field)
  ax = axes(field)
  nx, ny, nz = length.(ax)
  ox, oy, oz = first(ax[1]) - 1, first(ax[2]) - 1, first(ax[3]) - 1
  println(io)
  println(io, "temp = Array{Vector{Float64}}(undef, $nx, $ny, $nz);")
  println(io, "fg.$name = OffsetArray(temp, $ox, $oy, $oz);")
  println(io)
  for ix in ax[1], iy in ax[2], iz in ax[3]
    b = field[ix, iy, iz]
    println(io, "fg.$name[$ix, $iy, $iz] = [", b[1], ", ", b[2], ", ", b[3], "]")
  end
end

#---------------------------------------------------------------------------------------------------

"""
    _write_fixed_str_array(parent, name, strs::AbstractVector{<:AbstractString})

Write a fixed-length (null-terminated, ASCII) string-array attribute, matching
Bmad's `hdf5_write_attribute_string` rank-1.  `HDF5.jl` writes String arrays as
variable-length strings by default, which Bmad's reader cannot convert into its
fixed `character` buffers (it aborts on `axisLabels`).
"""
function _write_fixed_str_array(parent, name, strs::AbstractVector{<:AbstractString})
  n = maximum(length, strs)
  dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
  HDF5.API.h5t_set_size(dt, n)
  HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLTERM)
  HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_ASCII)
  dspace = dataspace((length(strs),))
  attr = create_attribute(parent, name, dt, dspace)
  buf = zeros(UInt8, n * length(strs))
  for (i, s) in enumerate(strs)
    cu = codeunits(s)
    copyto!(buf, (i - 1) * n + 1, cu, 1, length(cu))
  end
  HDF5.API.h5a_write(attr, dt, buf)
  close(attr); close(dspace); close(dt)
end

#---------------------------------------------------------------------------------------------------

"""
    _anchor_to_str(a::GridAnchorPt.T) -> String

Map a `GridAnchorPt` enum value to its openPMD `eleAnchorPt` string.
"""
function _anchor_to_str(a::GridAnchorPt.T)
  a == GridAnchorPt.Beginning && return "beginning"
  a == GridAnchorPt.Center    && return "center"
  return "end"
end

#---------------------------------------------------------------------------------------------------

"""
    _anchor_from_str(s) -> GridAnchorPt.T

Parse an openPMD `eleAnchorPt` string into a `GridAnchorPt` enum value.
"""
function _anchor_from_str(s)
  ls = lowercase(strip(string(s)))
  ls == "beginning" && return GridAnchorPt.Beginning
  ls == "center"    && return GridAnchorPt.Center
  ls == "end"       && return GridAnchorPt.End
  error("Unrecognized eleAnchorPt: $s")
end

#---------------------------------------------------------------------------------------------------

"""
    _geometry_to_str(::GridGeometry.T) -> String

Map a `GridGeometry` enum value to its openPMD `gridGeometry` string (only XYZ
is supported).
"""
_geometry_to_str(::GridGeometry.T) = "rectangular"

#---------------------------------------------------------------------------------------------------

"""
    _geometry_from_str(s) -> GridGeometry.T

Parse an openPMD `gridGeometry` string into a `GridGeometry` enum value.
"""
function _geometry_from_str(s)
  s == "rectangular" && return GridGeometry.XYZ
  error("read_field_grid_hdf5 supports only 'rectangular' (xyz) grids, got: $s")
end

#---------------------------------------------------------------------------------------------------

"""
    _component_dataset(field, c)

Lay component `c` of a `(ix, iy, iz)` OffsetArray of 3-vectors out as a 1-based
`(nx, ny, nz)` complex array.  `HDF5.jl` reverses dims on write, so the dataset
lands on disk exactly like Bmad's own Fortran writer (`H5Screate_simple_f` with
Fortran dims `[nx,ny,nz]`): Bmad's reader gets `data_dim = (nx,ny,nz)` and, with
`data_order "F"`, reads the column-major buffer back into `pt[ix,iy,iz]` correctly.
"""
function _component_dataset(field, c)
  ax = axes(field)
  nx, ny, nz = length(ax[1]), length(ax[2]), length(ax[3])
  out = Array{ComplexPMD}(undef, nx, ny, nz)
  for (a, ix) in enumerate(ax[1]), (b, iy) in enumerate(ax[2]), (k, iz) in enumerate(ax[3])
    v = field[ix, iy, iz][c]
    out[a, b, k] = ComplexPMD(real(v), imag(v))
  end
  return out
end

#---------------------------------------------------------------------------------------------------

"""
    _write_field_group(g1, name, field, unit_dim, unit_sym)

Write one field group (`"magneticField"`/`"electricField"`) from an `(ix,iy,iz)`
OffsetArray of 3-vectors.
"""
function _write_field_group(g1, name, field, unit_dim, unit_sym)
  grp = create_group(g1, name)
  for (c, axis) in enumerate(("x", "y", "z"))
    grp[axis] = _component_dataset(field, c)
    da = attributes(grp[axis])
    da["gridDataOrder"] = "F"           # explicit; Bmad reader honors this first
    da["localName"]     = axis
    da["unitSI"]        = [1.0]
    da["unitDimension"] = unit_dim
    da["unitSymbol"]    = unit_sym
  end
end

#---------------------------------------------------------------------------------------------------

"""
    _write_field_grid_text(path, mag, r0, dr, is_bend, field_scale)

Write the plain-text field-grid block from an (ix, iy, iz) OffsetArray of
[Bx,By,Bz] 3-vectors, using the grid's own indices (origin `r0`, spacing `dr`,
anchor = beginning).
"""
function _write_field_grid_text(path, mag, r0, dr, is_bend, field_scale)
  ax = axes(mag)
  open(path, "w") do io
    println(io, "{")
    println(io, "  geometry = xyz,")
    println(io, "  field_type = magnetic,")
    println(io, "  ele_anchor_pt = beginning,")
    is_bend && println(io, "  curved_ref_frame = T,")
    field_scale != 1 && println(io, "  field_scale = ", string(field_scale), ",")
    println(io, "  r0 = (", string(r0[1]), ", ", string(r0[2]), ", ", string(r0[3]), "),")
    println(io, "  dr = (", string(dr[1]), ", ", string(dr[2]), ", ", string(dr[3]), "),")
    println(io, "  {")
    for iz in ax[3], iy in ax[2], ix in ax[1]
      B = mag[ix, iy, iz]
      @printf(io, "    %d %d %d: %s %s %s,\n",
          ix, iy, iz, string(B[1]), string(B[2]), string(B[3]))
    end
    println(io, "  }")
    println(io, "}")
  end
end

#---------------------------------------------------------------------------------------------------

"""
    _is_hdf5_path(path)

True if `path` should be treated as an HDF5 file (".h5" or ".hdf5" suffix).
"""
_is_hdf5_path(path) = lowercase(splitext(path)[2]) in (".h5", ".hdf5")

#---------------------------------------------------------------------------------------------------

"""
    _write_coef_group(parent, name, d; single::Bool = false)

Write a Dict keyed by (n,m) (or by m, if `single`) as index arrays + matrix.
"""
function _write_coef_group(parent, name, d; single::Bool = false)
  g = create_group(parent, name)
  ks = sort(collect(keys(d)))
  nplanes = isempty(ks) ? 0 : length(d[first(ks)])
  V = Array{Float64}(undef, length(ks), nplanes)
  for (i, k) in enumerate(ks)
    V[i, :] = d[k]
  end
  if single
    g["m"] = Int[k for k in ks]
  else
    g["n"] = Int[k[1] for k in ks]
    g["m"] = Int[k[2] for k in ks]
  end
  g["values"] = V
end

function _read_coef_group(parent, name; single::Bool = false)
  g = parent[name]
  m = Int.(read(g["m"]))
  V = read(g["values"])
  if single
    return Dict{Int,Vector{Float64}}(m[i] => V[i, :] for i in eachindex(m))
  else
    n = Int.(read(g["n"]))
    return Dict{Tuple{Int,Int},Vector{Float64}}((n[i], m[i]) => V[i, :] for i in eachindex(m))
  end
end

#---------------------------------------------------------------------------------------------------

"""
    _attr(obj, name, default)

Read an attribute if present, else return `default`.
"""
_attr(obj, name, default) = haskey(attributes(obj), name) ? read_attribute(obj, name) : default

#---------------------------------------------------------------------------------------------------

"""
    _read_field_group(g1, name, lb, nx, ny, nz)

Read a field group (`"magneticField"`/`"electricField"`) into an `(ix, iy, iz)`
`OffsetArray` of `[Bx,By,Bz]` 3-vectors indexed from `lb`, or `nothing` if absent.

In a Bmad `field_grid` file each component dataset is written Fortran-order
(logical dims `[nx,ny,nz]`; on-disk C-dims `(nz,ny,nx)`).  `HDF5.jl` reverses dims on
read, so it hands back a 1-based `(nx, ny, nz)` array that is already the field --
no transpose needed.
"""
function _read_field_group(g1, name, lb, nx, ny, nz)
  haskey(g1, name) || return nothing
  grp = g1[name]
  comps = ntuple(_ -> zeros(Float64, nx, ny, nz), 3)   # one (nx,ny,nz) array per component
  for (c, axis) in enumerate(("x", "y", "z"))
    haskey(grp, axis) || continue       # missing component => zero field
    comp = read(grp[axis])
    size(comp) == (nx, ny, nz) ||
      error("field_grid dataset $name/$axis has size $(size(comp)), expected ($nx, $ny, $nz) " *
            "-- not a Bmad-format (Fortran-order) field_grid file.")
    comps[c] .= real.(comp)
  end
  field = [Float64[comps[1][a, b, k], comps[2][a, b, k], comps[3][a, b, k]]
      for a in 1:nx, b in 1:ny, k in 1:nz]
  return OffsetArray(field, lb[1]:lb[1]+nx-1, lb[2]:lb[2]+ny-1, lb[3]:lb[3]+nz-1)
end

# ===================================================================================================
# Fast evaluation of `field_and_potential_evaluate_at`.
#
# The reference path (`field_and_potential_evaluate` + `_interp_gg_fit`) rebuilds
# everything per call through type-unstable closures over the abstract-typed
# monomial tables (`Vector{Tuple{Real,Int,Int,Int}}`): ~120k allocations and ~1 MB
# per evaluation. Instead we compile, once per `fit`, a fully concrete `GGEvalPlan`
# (cached in `fit.eval_plan`):
#
#   * `towers`: precomputed monomial coefficients of the Hermite (or single-plane
#     Taylor) interpolant on each straddling plane-pair. Interpolating at `s` is
#     then evaluating a small polynomial and its first N derivatives into a dense
#     `Float64` value vector `gvals` -- no divided differences, no `Dict`s.
#   * `comps`: for each output component, a flat list of monomial terms
#     `(slot, w, p, q)` with `w = coeff * g_ref^k` folded in and `slot` an index
#     into `gvals`. Evaluating a component is then a single monomorphic loop.
#
# Results are identical to the reference path up to floating-point summation order.
# ===================================================================================================
#---------------------------------------------------------------------------------------------------

"""
    _hermite_poly(zL, zR, fL, fR) -> Vector{Float64}

Monomial coefficients (in `u = s - zL`) of the degree-`(2N+1)` two-point Hermite
interpolant with `fL[j+1] = f⁽ʲ⁾(zL)`, `fR[j+1] = f⁽ʲ⁾(zR)`, `j = 0..N`. Same
confluent-Newton construction as `_hermite_derivs`, but returns the polynomial
rather than evaluating it.
"""
function _hermite_poly(zL, zR, fL, fR)
  N = length(fL) - 1
  K = 2N + 1
  hstep = zR - zL
  fval(i, j) = (i <= N ? fL : fR)[j+1]
  isR(i) = i > N

  DD = zeros(Float64, K + 1, K + 1)          # DD[i+1, k+1] = f[t_i, …, t_{i+k}]
  for i in 0:K
    DD[i+1, 1] = fval(i, 0)
  end
  for k in 1:K, i in 0:K-k
    DD[i+1, k+1] = isR(i) == isR(i + k) ? fval(i, k) / factorial(k) :
                   (DD[i+2, k] - DD[i+1, k]) / hstep
  end
  c = [DD[1, k+1] for k in 0:K]              # Newton coefficients
  tnode(i) = i <= N ? 0.0 : hstep

  poly  = zeros(Float64, K + 1)
  basis = [1.0]
  for k in 0:K
    for d in 1:length(basis)
      poly[d] += c[k+1] * basis[d]
    end
    if k < K
      tk = tnode(k)
      nb = zeros(Float64, length(basis) + 1)
      for d in 1:length(basis)
        nb[d+1] += basis[d]
        nb[d]   -= tk * basis[d]
      end
      basis = nb
    end
  end
  return poly
end

"""
    _taylor_poly(f0) -> Vector{Float64}

Monomial coefficients (in `u = s - z0`) of the single-plane Taylor series:
`poly[d+1] = f0[d+1] / d!`, so that `P⁽ᵐ⁾(u)` matches `_taylor_derivs`.
"""
_taylor_poly(f0) = [f0[d+1] / factorial(d) for d in 0:length(f0)-1]

#---------------------------------------------------------------------------------------------------

"""
    _make_tower(z, Fn, slots, extra_slots, extra_planevals) -> _Tower

Build one tower from a per-`n` value matrix `Fn[j+1, plane] = f⁽ʲ⁾` at that
plane, its `slots`, and any extras. `z` are the base planes. Precomputes the
interpolant's monomial coefficients on every straddling plane-pair (or the
single-plane Taylor series when there is one plane).
"""
function _make_tower(z, Fn, slots, extra_slots, extra_planevals)
  P = length(z)
  N = length(slots) - 1
  if P == 1
    deg = N
    poly = reshape(_taylor_poly(Fn[:, 1]), deg + 1, 1)
    zref = [z[1]]
  else
    deg = 2N + 1
    npair = P - 1
    poly = Matrix{Float64}(undef, deg + 1, npair)
    zref = Vector{Float64}(undef, npair)
    for ip in 1:npair
      fL = @view Fn[:, ip]
      fR = @view Fn[:, ip+1]
      poly[:, ip] = _hermite_poly(z[ip], z[ip+1], fL, fR)
      zref[ip] = z[ip]
    end
  end
  return _Tower(N, deg, slots, poly, zref, extra_slots, extra_planevals)
end

"""
    _build_towers_nm!(towers, z, d, slotmap, next) -> next

Assign `gvals` slots for the keys of an `(n,m)`-keyed dict `d` (`a` or `b`),
build one `_Tower` per multipole `n`, and append them to `towers`. Returns the
next free slot index.
"""
function _build_towers_nm!(towers, z, d::Dict{Tuple{Int,Int},Vector{Float64}}, slotmap, next)
  for k in keys(d)
    slotmap[k] = next
    next += 1
  end
  byn = Dict{Int,Vector{Int}}()
  for (n, m) in keys(d)
    push!(get!(byn, n, Int[]), m)
  end
  P = length(z)
  for (n, ms) in byn
    sort!(ms)
    N = _contiguous_order(ms)
    slots = [slotmap[(n, j)] for j in 0:N]
    Fn = Matrix{Float64}(undef, N + 1, P)
    for j in 0:N, ip in 1:P
      Fn[j+1, ip] = d[(n, j)][ip]
    end
    extra = [m for m in ms if m > N]
    extra_slots = [slotmap[(n, m)] for m in extra]
    push!(towers, _make_tower(z, Fn, slots, extra_slots, [d[(n, m)] for m in extra]))
  end
  return next
end

"""
    _build_towers_m!(towers, z, d, slotmap, next) -> next

Like `_build_towers_nm!` but for the `m`-keyed `bs` dict: a single `_Tower`.
"""
function _build_towers_m!(towers, z, d::Dict{Int,Vector{Float64}}, slotmap, next)
  for k in keys(d)
    slotmap[k] = next
    next += 1
  end
  isempty(d) && return next
  ms = sort(collect(keys(d)))
  N = _contiguous_order(ms)
  P = length(z)
  slots = [slotmap[j] for j in 0:N]
  Fn = Matrix{Float64}(undef, N + 1, P)
  for j in 0:N, ip in 1:P
    Fn[j+1, ip] = d[j][ip]
  end
  extra = [m for m in ms if m > N]
  extra_slots = [slotmap[m] for m in extra]
  push!(towers, _make_tower(z, Fn, slots, extra_slots, [d[m] for m in extra]))
  return next
end

#---------------------------------------------------------------------------------------------------

"""
    _build_comp(Ta, Tb, Tbs, bump, g_ref, slot_a, slot_b, slot_bs) -> _CompTerms

Flatten one output component's `(a, b, bs)` monomial tables into a flat term
list, folding `g_ref^k` into each weight and resolving every `(n,m)`/`m` key to a
`gvals` slot. `bump` shifts the GG derivative order by one (used for `∂A/∂s`).
Terms whose GG value is structurally zero (missing key) are dropped.
"""
function _build_comp(Ta, Tb, Tbs, bump::Bool, g_ref,
                     slot_a::Dict{Tuple{Int,Int},Int},
                     slot_b::Dict{Tuple{Int,Int},Int},
                     slot_bs::Dict{Int,Int})
  slot = Int[]; w = Float64[]; ps = Int[]; qs = Int[]
  wk(c, k) = float(c) * (k == 0 ? 1.0 : float(g_ref)^k)
  for (T, smap) in ((Ta, slot_a), (Tb, slot_b))
    for ((n, m), terms) in T
      s = get(smap, (n, bump ? m + 1 : m), 0)
      s == 0 && continue
      for (c, p, q, k) in terms
        push!(slot, s); push!(w, wk(c, k)); push!(ps, p); push!(qs, q)
      end
    end
  end
  for (m, terms) in Tbs
    s = get(slot_bs, bump ? m + 1 : m, 0)
    s == 0 && continue
    for (c, p, q, k) in terms
      push!(slot, s); push!(w, wk(c, k)); push!(ps, p); push!(qs, q)
    end
  end
  return _CompTerms(slot, w, ps, qs)
end

#---------------------------------------------------------------------------------------------------

"""
    _build_eval_plan(fit::GGCoefs) -> GGEvalPlan

Compile `fit` into a `GGEvalPlan`: assign a dense `gvals` slot to every GG value,
build the interpolation towers, and flatten the nine output components' monomial
tables into term lists. Called once per `fit` and cached by `_get_eval_plan`.
"""
function _build_eval_plan(fit::GGCoefs)
  g_ref = fit.g_ref
  z = copy(fit.z_base)
  slot_a  = Dict{Tuple{Int,Int},Int}()
  slot_b  = Dict{Tuple{Int,Int},Int}()
  slot_bs = Dict{Int,Int}()
  towers = _Tower[]
  next = 2                                   # slot 1 is the always-zero sentinel
  next = _build_towers_nm!(towers, z, fit.a,  slot_a,  next)
  next = _build_towers_nm!(towers, z, fit.b,  slot_b,  next)
  next = _build_towers_m!( towers, z, fit.bs, slot_bs, next)
  ngvals = next - 1

  comps = (
    _build_comp(Bx_a, Bx_b, Bx_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(By_a, By_b, By_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(Bs_a, Bs_b, Bs_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(Ax_a, Ax_b, Ax_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(Ay_a, Ay_b, Ay_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(As_a, As_b, As_bs, false, g_ref, slot_a, slot_b, slot_bs),
    _build_comp(Ax_a, Ax_b, Ax_bs, true,  g_ref, slot_a, slot_b, slot_bs),
    _build_comp(Ay_a, Ay_b, Ay_bs, true,  g_ref, slot_a, slot_b, slot_bs),
    _build_comp(As_a, As_b, As_bs, true,  g_ref, slot_a, slot_b, slot_bs),
  )
  pmax = 0; qmax = 0
  for ct in comps, t in 1:length(ct)
    pmax = max(pmax, ct.p[t]); qmax = max(qmax, ct.q[t])
  end
  maxdeg = isempty(towers) ? 0 : maximum(tw.deg for tw in towers)
  return GGEvalPlan((float(fit.origin[1]), float(fit.origin[2])),
                    z, towers, maxdeg, ngvals, comps, pmax, qmax)
end

#---------------------------------------------------------------------------------------------------

"""
    _get_eval_plan(fit::GGCoefs) -> GGEvalPlan

Return `fit`'s compiled `GGEvalPlan`, building it and storing it in
`fit.eval_plan` on first use. Assumes `fit` is not mutated after the first
evaluation (a stale plan is not detected).
"""
function _get_eval_plan(fit::GGCoefs)
  plan = fit.eval_plan
  plan === nothing || return plan
  plan = _build_eval_plan(fit)
  fit.eval_plan = plan
  return plan
end

#---------------------------------------------------------------------------------------------------

"""
    _fill_gvals!(gvals, upow, plan, s) -> gvals

Interpolate every tower onto `s`, scattering the results into the dense value
vector `gvals`. `upow` is a caller-provided scratch buffer of length
`plan.maxdeg + 1` for the powers of `u = s - zref`.
"""
function _fill_gvals!(gvals, upow, plan::GGEvalPlan, s::Real)
  z = plan.z; P = length(z); sq = float(s)
  if P == 1
    pair = 1
  else
    i0 = searchsortedlast(z, sq)
    pair = clamp(i0, 1, P - 1)
  end
  fill!(gvals, 0.0)
  @inbounds for tw in plan.towers
    u = sq - tw.zref[pair]
    deg = tw.deg
    upow[1] = 1.0
    for d in 1:deg
      upow[d+1] = upow[d] * u
    end
    for m in 0:tw.N
      acc = 0.0
      for d in m:deg
        ff = 1.0                     # falling factorial d·(d-1)···(d-m+1)
        for r in 0:m-1
          ff *= (d - r)
        end
        acc += tw.poly[d+1, pair] * ff * upow[d-m+1]
      end
      gvals[tw.slots[m+1]] = acc
    end
    for e in eachindex(tw.extra_slots)
      gvals[tw.extra_slots[e]] = tw.extra_planevals[e][pair]
    end
  end
  return gvals
end

"""
    _comp_value(ct, gvals, xp, yq) -> val

Evaluate a component's value `Σ w * gvals[slot] * x^p * y^q`, where `xp[i+1] = x^i`
and `yq[j+1] = y^j` are precomputed power tables.
"""
@inline function _comp_value(ct::_CompTerms, gvals, xp, yq)
  val = 0.0
  @inbounds for t in eachindex(ct.slot)
    val += ct.w[t] * gvals[ct.slot[t]] * xp[ct.p[t]+1] * yq[ct.q[t]+1]
  end
  return val
end

"""
    _comp_full(ct, gvals, xp, yq) -> (val, dvx, dvy)

Like `_comp_value` but also returns the `x` and `y` partial derivatives of the
component.
"""
@inline function _comp_full(ct::_CompTerms, gvals, xp, yq)
  val = 0.0; dvx = 0.0; dvy = 0.0
  @inbounds for t in eachindex(ct.slot)
    base = ct.w[t] * gvals[ct.slot[t]]
    p = ct.p[t]; q = ct.q[t]
    val += base * xp[p+1] * yq[q+1]
    p > 0 && (dvx += base * p * xp[p] * yq[q+1])
    q > 0 && (dvy += base * q * xp[p+1] * yq[q])
  end
  return val, dvx, dvy
end

