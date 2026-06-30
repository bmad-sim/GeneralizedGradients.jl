module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX
  using LinearAlgebra, Printf

  # Package-wide constants (referenced by the helpers in helpers.jl and the
  # evaluation / field-grid code).

  # Working size for the truncated (x,y) coefficient arrays.  The gg_coef table
  # is built to total monomial degree MAXTOT (12), so 20 leaves ample headroom.
  const _NMAX = 20

  # openPMD SI base-unit exponents (L, M, T, I, Theta, N, J) for Tesla and V/m.
  const _DIM_TESLA = [0.0, 1, -2, -1, 0, 0, 0]
  const _DIM_VPERM = [1.0, 1, -3, -1, 0, 0, 0]

  # GG coefficient tables: Bx_a … Bs_bs (field) and Ax_a … As_bs (vector potential).
  const _TABLE_FILE = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
  include(_TABLE_FILE)

  include("struct.jl")
  include("helpers.jl")
  include("gg_eval.jl")
  include("gg_fit.jl")
  include("write_bmad_gg_fit.jl")
  include("field_grid.jl")

  export FieldGridTable,
       GridAnchorPt,
       GridGeometry,
       GGFitInputParams,
       GGCoefs,
       gg_fit,
       gg_fit_show_results,
       write_gg_fit,
       read_gg_fit,
       write_bmad_field_grid,
       write_bmad_gg_fit,
       write_field_grid,
       read_field_grid_hdf5,
       write_field_grid_hdf5,
       field_and_potential_evaluate,
       field_and_potential_evaluate_at,
       field_coefficients_at_plane,
       field_coefficients_at_s,
       gg_coefficients_at_plane,
       gg_coefficients_at_s

end
