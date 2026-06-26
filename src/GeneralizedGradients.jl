module GeneralizedGradients

  using HDF5, OffsetArrays, Dates

  include("gg_eval.jl")
  include("hdf5_grid_field.jl")

  export gg_load_fit,
       field_and_potential_evaluate,
       field_and_potential_evaluate_at,
       field_coefficients_at_plane,
       field_coefficients_at_s,
       gg_coefficients_at_plane,
       gg_coefficients_at_s

end
