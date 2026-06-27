module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX

  include("struct.jl")
  include("gg_eval.jl")
  include("field_io.jl")
  include("hdf5_grid_field.jl")

  export FieldGridTable,
       GridAnchorPt,
       GridGeometry,
       gg_load_fit,
       gg_save_fit,
       read_field_grid,
       write_field_grid,
       read_grid_field_hdf5,
       write_grid_field_hdf5,
       field_and_potential_evaluate,
       field_and_potential_evaluate_at,
       field_coefficients_at_plane,
       field_coefficients_at_s,
       gg_coefficients_at_plane,
       gg_coefficients_at_s

end
