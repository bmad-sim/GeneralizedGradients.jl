"""
    run_gg_fit

Example of a command file to run gg_fit which fits generalized gradient (GG) coefficients to
a 3D magnetic field table plane by plane.

## Usage

Run with command:

```
julia run_gg_fit.jl
```

See the documentation for the `gg_fit` function for more details.

## Field grid

`read_field_grid_hdf5` returns a `FieldGridTable` with:

```
field.magnetic[ix, iy, iz]   Field 3-vector [Bx, By, Bz] at the grid point,
                             an OffsetArray (indices need not start at 0/1).
field.r0                     Grid origin (3-vector)
field.dr                     Grid spacing 3-vector
field.g_ref                  Curvilinear coordinate system bending strength = 1 / bending_radius.
```

A grid point `(ix, iy, iz)` is at `(x, y, z) = r0 + dr .* (ix, iy, iz)`.
"""

using GeneralizedGradients

grid_file = "wsnk_fieldmap_reduced.h5"
field = read_field_grid_hdf5(grid_file)

p = GGFitInputParams()
p.origin = [0.0, 0.0]      # (x, y) origin about which the generalized gradients coefs are computed
p.n_planes_add = 1            # Number of z-planes added.
p.core_weight = 1             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
p.outer_plane_weight = 1      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
p.output_file = "gg_fit_result.h5"

results = gg_fit(field, p)
gg_fit_show_results(results, field, p)
write_gg_fit(results, field, p)