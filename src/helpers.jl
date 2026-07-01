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
