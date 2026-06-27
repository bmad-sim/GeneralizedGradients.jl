using GeneralizedGradients

#---------------------------------------------------------------------------------------------------
# Field grid

# read_field_grid returns a FieldGridTable with:
#   field.magnetic[ix, iy, iz]     Field 3-vector [Bx, By, Bz] at the grid point,
#                                  an OffsetArray (indices need not start at 0/1).
#   field.r0                       Grid origin (3-vector)
#   field.dr                       Grid spacing 3-vector
#   field.g_ref                    Curvilinear coordinate system bending strength = 1 / bending_radius.
# A grid point (ix, iy, iz) is at (x, y, z) = r0 + dr .* (ix, iy, iz).

grid_file = joinpath(@__DIR__, "wsnk_fieldmap_reduced.h5")
field = read_field_grid(grid_file)

#---------------------------------------------------------------------------------------------------
# Other parameters

origin = [-0.0, 0.0]      # (x, y) origin about which the generalized gradients coefs are computed
n_planes_add = 1            # Number of z-planes added.
core_weight = 1             # Merit function weight on "core" (points with (x,y) near (0,0)) field table points.
outer_plane_weight = 1      # Merit function weight for the "outer" z-planes. Default is 1 (uniform weighting).
output_file = "gg_fit_result.h5"
# Parameter Documentation

"""
    origin = [x0, y0]

Defines the line [x0, y0, z] about which the generalized gradient coefficients are computed.
If h is non-zero, origin must be [0, 0].

    n_planes_add = Int

This parameter sets the number of z-planes added to either side of the base z-plane to
be used in the analysis of the derivatives at any given base z-plane (see "How the GG Calculation
Works" section). For example, for n_planes_add = 2, two planes would be added to either side of the
base plane making the total number of planes used in the analysis equal to five.

  core_weight = Float

Merit function weight for "core" points (field table points whose transverse (x,y)
position is near (0,0)). Default is 1.0 which gives an equal weight for all points of a given
z-plane. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  outer_plane_weight = Float

Merit function weight for z-planes away from the base z-plane when n_planes_add
is non-zero. See the "How the GG Calculation Works" section below for documentation on the optimizer
merit function.

  output_file

Name of the output file.

"""

#---------------------------------------------------------------------------------------------------
"""
    How the GG Calculation Works:

The GG coefficients are calculated at equally spaced z-positions corresponding to z-planes of the
field table. GG coefficients are calculated by varying the coefficients to achieve the minimum of a
"merit function". This fitting is done z-plane by z-plane. That is, the coefficients of a given
z-plane are calculated independently of the coefficients of any other z-plane and all coefficients
of a given z-plane are calculated simultaneously. For a given z-plane, The merit function is
  Merit = Sum: weight * (field_from_table - field_calculated_from_GG_coefs)^2

The z-plane at which the GG coefficients are being calculated is called the "base plane". The merit
function is calculated by a sum over those field points that lie in a z-plane that is within
n_planes_add of the base plane. For example, for n_planes_add = 2, two planes would be added to
either side of the base plane making the total number of planes used in the analysis equal to
five. Near the ends of the table, the number of z-planes used will be reduced. Thus, for
n_planes_add = 2, a base z-plane at the end of the table will only use three planes in the analysis.

To calculate the field at a non-base plane, the GG coefficients of the base plane are extrapolated
to the non-base plane by
  a(n,m)(z) = Sum_{j = m}^{N_n} (z - z_fit)^(j-m) / (j-m)! * a(n,j)(z_fit)
with similar expressions for b and bs. Adding extra planes to the analysis smooths the calculated values. 
The approximation that the GG
curve is well fit by a polynomial with coefficients given by the derivatives at the base
plane becomes less accurate as more planes are used in the analysis. Therefore, past some limit,
using more planes will make the calculation less accurate.

The weight for a given field point is determined by the setting of the core_weight and
outer_plane_weight parameters. The weight is a product of two factors:
  weight(x,y,dz) = w_core(x,y) * w_plane(dz)
where (x,y,dz) are the coordinates of the field point with respect to the base plane.

The w_core(x,y) factor is computed from:
  w_core(x,y) = core_weight * rmax^2 / (rmax^2 + r^2 * (core_weight - 1))
where r^2 = x^2 + y^2 and rmax is the maximum r over all points. A setting of core_weight of 1 (the
default) means that w_core is a constant. A setting of w_core greater than 1 means that the core
points (points with low r) will have higher weight at the expense of points farther away. A better
core fit is generally what is wanted since particles of a beam will spend most of their time near
the core.

The w_plane(dz) factor is computed from:
  w_plane(dz) = 1 + (outer_plane_weight - 1) * |dz| / dz_max
where dz_max is the maximum dz at the ends of the fit region. If n_planes_add = 0 (in which case
dz_max is zero and the above equation is singular), the above equation is ignored and a value of 1
is used for w_plane. A setting of outer_plane_weight of 1 (the default) means that w_plane is a
constant. To weight the planes nearer to the base plane more than the outer planes, choose a value
less than 1 but still non-negative.

Note: In theory, the merit function fitting does not require that the field table be a rectangular
grid of equally spaced points. In fact, a set of randomly spaced field points would work.
"""

;