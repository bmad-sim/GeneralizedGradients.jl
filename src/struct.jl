@enumx GridAnchorPt Beginning Center End
@enumx GridGeometry XYZ 

"""
    mutable struct FieldGridTable

Struct to hold electric and magnetic field grid.
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