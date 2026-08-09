module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX
  using LinearAlgebra, Printf
  using StaticArrays: SVector, SMatrix
  using Adapt

  # GG coefficient tables: Bx_a … Bs_bs (field) and Ax_a … As_bs (vector
  # potential), plus the MAXTOT and MMAX the table was built with.
  const _TABLE_FILE = joinpath(@__DIR__, "..", "tables", "gg_coef_table.jl")
  include(_TABLE_FILE)

  # Package-wide constants (referenced by the helpers in low_level.jl and the
  # evaluation / field-grid code).

  # Working size for the truncated (x,y) coefficient arrays: K[p+1,q+1] holds
  # the coefficient of x^p y^q, and the table tabulates p + q <= MAXTOT, so the
  # largest index either p or q can reach is MAXTOT.  Derived from the table
  # rather than fixed, so a table built at a different MAXTOT still fits.
  const _NMAX = MAXTOT + 1

  # openPMD SI base-unit exponents (L, M, T, I, Theta, N, J) for Tesla and V/m.
  const _DIM_TESLA = [0.0, 1, -2, -1, 0, 0, 0]
  const _DIM_VPERM = [1.0, 1, -3, -1, 0, 0, 0]

  include("struct.jl")
  include("low_level.jl")
  include("gg_eval.jl")
  include("gg_fit.jl")
  include("gg_diagnostics.jl")
  include("gg_utils.jl")
  include("field_grid_utils.jl")

  export MAXTOT,          # defined, with their docstrings, in tables/gg_coef_table.jl
       MMAX,
       FieldGridTable,
       GridAnchorPt,
       GridGeometry,
       GGFitInputParams,
       GGFit,
       GGFitScanPoint,
       gg_calc_fit,
       gg_show_fit_results,
       gg_make_fit_residual_table,
       gg_show_fit_residuals,
       write_gg_fit,
       read_gg_fit,
       write_bmad_field_grid_element,
       write_bmad_gg_fit,
       write_field_grid,
       read_field_grid_hdf5,
       write_field_grid_hdf5,
       field_and_potential_evaluate,
       field_and_potential_evaluate_at,
       potential_evaluate_at,
       field_evaluate_at,
       eval_plan,
       field_coefficients_at_plane,
       field_coefficients_at_s,
       gg_coefficients_at_plane,
       gg_coefficients_at_s

end
