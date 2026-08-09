"""
    run_gg_fit

Example of a command file to run gg_calc_fit which fits generalized gradient (GG) coefficients to
a 3D magnetic field table plane by plane.

## Usage

Run with command:

```
julia run_gg_fit.jl
```

See the documentation for the `gg_calc_fit` function for more details.

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
# Size of the fitted model. A single number pins the cutoff; a range makes gg_calc_fit try
# every combination and keep the best by p.fit_criterion (:bic or :aic), with
# the coefficient count and RMS of each combination reported by gg_show_fit_results.
p.m_max  = 2:6                 # Max multipole order m.  A single Int pins it. E.g. 1:10 to scan.
p.nd_max = 2:6                 # Max derivative order nd. A single Int pins it. E.g. 0:6 to scan.
p.fit_criterion = :bic        # How a scan picks its winner. Ignored if neither cutoff is a range.
p.core_weight = 1             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
p.outer_plane_weight = 1      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
p.output_file = "gg_fit_result.h5"

gg_fit = gg_calc_fit(field, p)
gg_show_fit_results(gg_fit, field, p)
write_gg_fit(gg_fit, field, p)
