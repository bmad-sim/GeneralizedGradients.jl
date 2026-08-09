"""
    plot_gg_residuals

Plot the difference between a field grid and its GG fit, as a surface over the
transverse plane. This is the picture behind one plane's RMS residual: whether
the fit is missing a multipole, whether the error is concentrated in the corners
of the grid, and whether it is smooth or noisy are all obvious at a glance and
none of them are visible in a single RMS number.

The data comes from `gg_fit_residual_map`, which is part of the package. Only the
drawing needs a plotting library, so it lives here in `programs/` rather than
becoming a dependency of `GeneralizedGradients`.

## Usage

```
] add GLMakie          # interactive, rotate the surface with the mouse
] add CairoMakie       # or this one, for writing files on a headless machine

julia plot_gg_residuals.jl
```

Then `plot_residual_surfaces(results, field, plane)` for the three components of
one plane, or `plot_residual_component(results, field, component)` to sweep a
component across planes.
"""

using GeneralizedGradients
using GLMakie          # swap for `using CairoMakie` to write files instead

const COMPONENT_NAME = ("Bx", "By", "Bs")

#---------------------------------------------------------------------------------------------------

"""
    plot_residual_surfaces(results, field, plane; dplane = 0, component = 1:3) -> Figure

Surface plot of `field table - GG fit` over the transverse grid of one base
plane, one panel per field component.

The colour scale of each panel is symmetric about zero, so the sign of the error
reads directly and a residual with `m`-fold azimuthal symmetry — the signature of
a missing multipole of order `m` — shows up as `2m` alternating lobes around the
axis. `dplane` selects a plane offset from the base plane, as in
`gg_fit_residual_map`.
"""
function plot_residual_surfaces(results, field, plane::Integer;
                                dplane::Integer = 0, component = 1:3)
  r  = gg_fit_residual_map(results, field, plane; dplane)
  ks = collect(component)
  # Axis3 draws its z label outside the panel, so leave room for it.
  fig = Figure(size = (520 * length(ks), 520), figure_padding = 34)
  for (col, k) in enumerate(ks)
    d   = r.dB[:, :, k]
    lim = maximum(abs, d)
    lim = lim == 0 ? 1.0 : lim
    ax = Axis3(fig[1, col], xlabel = "x [m]", ylabel = "y [m]", zlabel = "ΔB [T]",
               title = "$(COMPONENT_NAME[k]):  table - fit,  plane $plane, z = $(round(r.z, digits = 6)) m")
    surface!(ax, r.x, r.y, d; colormap = :balance, colorrange = (-lim, lim))
    Colorbar(fig[2, col], limits = (-lim, lim), colormap = :balance,
             vertical = false, label = "ΔB [T]")
  end
  return fig
end

#---------------------------------------------------------------------------------------------------

"""
    plot_residual_component(results, field, component; planes = ...) -> Figure

Heatmap of one field component's residual for several base planes at once, on a
colour scale shared by every panel so the planes are directly comparable.

Use this to find where along `z` a fit goes wrong, then
`plot_residual_surfaces` on the plane that stands out.
"""
function plot_residual_component(results, field, component::Integer;
                                 planes = eachindex(results.z_base), ncol::Integer = 4)
  maps = [gg_fit_residual_map(results, field, p) for p in planes]
  lim  = maximum(maximum(abs, m.dB[:, :, component]) for m in maps)
  lim  = lim == 0 ? 1.0 : lim
  nrow = cld(length(maps), ncol)
  fig  = Figure(size = (320 * ncol, 300 * nrow + 60))
  for (i, (p, m)) in enumerate(zip(planes, maps))
    row, col = fldmod1(i, ncol)
    ax = Axis(fig[row, col], aspect = 1, xlabel = "x [m]", ylabel = "y [m]",
              title = "plane $p,  z = $(round(m.z, digits = 6)) m")
    heatmap!(ax, m.x, m.y, m.dB[:, :, component];
             colormap = :balance, colorrange = (-lim, lim))
  end
  Colorbar(fig[nrow+1, 1:ncol], limits = (-lim, lim), colormap = :balance,
           vertical = false, label = "Δ$(COMPONENT_NAME[component]) [T]")
  return fig
end

#---------------------------------------------------------------------------------------------------
# Run as a script: fit the example grid and plot the worst plane.

if abspath(PROGRAM_FILE) == @__FILE__
  field = read_field_grid_hdf5(joinpath(@__DIR__, "..", "examples",
                                        "wsnk_fieldmap_reduced.h5"))

  params = GGFitInputParams()
  params.n_planes_add = 1
  results = gg_fit(field, params)

  # The numbers behind the pictures. Read this first: it says which plane to look
  # at and what to expect to see when you do.
  gg_fit_show_residuals(results, field)

  worst = argmax(results.rms_weighted_plane)
  println("Worst plane is $worst (z = $(results.z_base[worst]) m).")

  fig = plot_residual_surfaces(results, field, worst)
  save(joinpath(@__DIR__, "gg_residual_plane_$worst.png"), fig)
  fig2 = plot_residual_component(results, field, 1)
  save(joinpath(@__DIR__, "gg_residual_Bx_all_planes.png"), fig2)
  println("Wrote gg_residual_plane_$worst.png and gg_residual_Bx_all_planes.png")
end
