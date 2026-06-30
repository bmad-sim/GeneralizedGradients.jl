module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX
  using LinearAlgebra, Printf

  include("struct.jl")
  include("gg_eval.jl")
  include("gg_fit.jl")
  include("gg_to_bmad.jl")
  include("field_grid_to_bmad.jl")
  include("field_grid_io.jl")
  include("field_grid_hdf5.jl")

  export FieldGridTable,
       GridAnchorPt,
       GridGeometry,
       GGFitInputParams,
       GGCoefs,
       gg_fit,
       gg_fit_show_results,
       write_gg_fit,
       read_gg_fit,
       field_grid_to_bmad,
       gg_to_bmad,
       write_bmad_field_grid,
       write_bmad_gen_grad_map,
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
