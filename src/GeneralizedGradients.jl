module GeneralizedGradients

  using HDF5, OffsetArrays, Dates, EnumX
  using LinearAlgebra, Printf

  include("struct.jl")
  include("gg_eval.jl")
  include("field_io.jl")
  include("hdf5_field_grid.jl")
  include("gg_fit.jl")
  include("grid_to_bmad.jl")
  include("gg_to_bmad.jl")

  export FieldGridTable,
       GridAnchorPt,
       GridGeometry,
       gg_fit,
       gg_load_fit,
       gg_save_fit,
       grid_to_bmad,
       gg_to_bmad,
       write_bmad_field_grid,
       write_bmad_gen_grad_map,
       read_field_grid,
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
