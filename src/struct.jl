@enumx GridAnchorPt Beginning Center End
@enumx GridGeometry XYZ 

"""
    mutable struct FieldGridTable{T}

Holds an electric and/or magnetic field sampled on a 3D grid.

`magnetic` and `electric` are 3D `OffsetArray`s whose elements are field
3-vectors: `magnetic[ix,iy,iz] == [Bx, By, Bz]` (and likewise `[Ex, Ey, Ez]`).
The grid indices `(ix, iy, iz)` need not start at 0 or 1; a grid point is at
position `r0 + dr .* (ix, iy, iz)` relative to the anchor.

Fields:
- `magnetic` — magnetic field 3-vectors `[Bx, By, Bz]` [T].
- `electric` — electric field 3-vectors `[Ex, Ey, Ez]` [V/m].
- `r0` — grid origin offset `(x0, y0, z0)` [m].
- `dr` — grid spacing `(dx, dy, dz)` [m].
- `g_ref` — curvilinear-coordinate bending strength `1/bending_radius` [1/m]
  (`0` for a straight reference curve).
- `scale` — overall field scale factor.
- `RF_frequency` — RF frequency [Hz] (`0` for a static field).
- `RF_phase` — RF phase [rad].
- `anchor_pt` — grid anchor point, a `GridAnchorPt.T` (`Beginning`, `Center`, or `End`).
- `geometry` — grid geometry, a `GridGeometry.T` (`XYZ`).

`FieldGridTable()` builds an empty table with `T = Float64`; read one from a
file with `read_field_grid`.
"""
@kwdef mutable struct FieldGridTable{T}
  # `magnetic`/`electric` are 3D grids whose elements are field 3-vectors:
  # `magnetic[ix,iy,iz] == [Bx, By, Bz]`.  Defaults use concrete Float64 (not `T`)
  # so that the parameterless `FieldGridTable()` works: it infers T = Float64.
  magnetic::OffsetArray{Vector{T}} = OffsetArray(Array{Vector{Float64}}(undef, 0, 0, 0), 1:0, 1:0, 1:0)
  electric::OffsetArray{Vector{T}} = OffsetArray(Array{Vector{Float64}}(undef, 0, 0, 0), 1:0, 1:0, 1:0)
  r0::Vector{T} = [0.0, 0.0, 0.0]
  dr::Vector{T} = [0.0, 0.0, 0.0]
  g_ref::T = 0.0
  scale::T = 1.0
  RF_frequency::T = 0.0
  RF_phase::T = 0.0
  anchor_pt::GridAnchorPt.T = GridAnchorPt.Center
  geometry::GridGeometry.T = GridGeometry.XYZ
end

# For a, b that are structs, the default `a == b` will only be true if `a` and `b` are equivalent (a === b).
# Here redefine `==` for `FieldGridTable` to return true if all fields are equal.

Base.:(==)(a::FieldGridTable, b::FieldGridTable) = all(getfield(a,f) == getfield(b,f) for f in fieldnames(FieldGridTable))

#---------------------------------------------------------------------------------------------------

@kwdef mutable struct GGTerm{T}
  coef::T = 0.0
  expn::Vector{Int} = [0,0]
end


@kwdef mutable struct GGTaylor{T}
  term::Vector{GGTerm{T}} = [GGTerm{T}()]
end

# GGTaylor() = GGTaylor{Float64}()

#---------------------------------------------------------------------------------------------------
"""
    origin = [x0, y0]

Defines the line [x0, y0, z] about which the generalized gradient coefficients are computed.
If h is non-zero, origin must be [0, 0].

    n_planes_add = Int

This parameter sets the number of z-planes added to either side of the base z-plane to
be used in the analysis of the derivatives at any given base z-plane (see "How the GG Calculation
Works" section). For example, for n_planes_add = 2, two planes would be added to either side of the
base plane making the total number of planes used in the analysis equal to five.

  core_weight = Float

Merit function weight for "core" points (field table points whose transverse (x,y)
position is near (0,0)). Default is 1.0 which gives an equal weight for all points of a given
z-plane. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  outer_plane_weight = Float

Merit function weight for z-planes away from the base z-plane when n_planes_add
is non-zero. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  output_file

Name of the output file.

"""

@kwdef mutable struct GGFitParams
  origin::Vector{Float64} = [0.0, 0.0]   # (x, y) origin about which the generalized gradients coefs are computed
  n_planes_add::Int = 1                  # Number of z-planes added.
  core_weight::Int = 1                   # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
  outer_plane_weight::Int = 1            # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
  output_file::String = "gg_fit_results.h5"
end