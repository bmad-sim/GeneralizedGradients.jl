module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX
  using LinearAlgebra, Printf

  include("struct.jl")
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
