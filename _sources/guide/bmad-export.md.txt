# Exporting to Bmad

The package can write two kinds of [Bmad](https://www.classe.cornell.edu/bmad/)
lattice elements: a `grid_field` element (the raw field grid) and a
`gen_grad_map` element (the fitted generalized gradients).

In both cases a non-zero reference bending strength `g_ref = 1/bend_radius`
[1/m] makes the element an `sbend` with `curved_ref_frame = T`; otherwise it is
an `em_field`. Each element is anchored at its entrance
(`ele_anchor_pt = beginning`).

## Field grid → `grid_field`

`write_bmad_field_grid` writes a field grid as a Bmad `grid_field`. Its `field`
argument is either a `FieldGridTable` or the path to a `field_grid` HDF5 file:

```julia
using GeneralizedGradients
write_bmad_field_grid("field_grid.h5")                  # openPMD HDF5 grid (default)
write_bmad_field_grid("field_grid.h5"; hdf5 = false)    # plain-text grid block
write_bmad_field_grid(field)                             # from a FieldGridTable
```

The reference frame is determined by the grid's own `g_ref`: non-zero gives an
`sbend`, zero an `em_field`.

Two files are written: `<output_base>.bmad` (the lattice element) and the grid,
either `<output_base>_grid.h5` (HDF5, the default) or `<output_base>_grid.bmad`
(plain text).

From the shell:

```
julia programs/run_write_bmad_field_grid.jl <field_grid.h5> [output_base] [--text]
```

## GG fit → `gen_grad_map`

`write_bmad_gg_fit` converts a GG fit into a Bmad `gen_grad_map`. Its input is
either a GG fit file (the output of `write_gg_fit`) or a loaded fit (`fit, meta`
as returned by `read_gg_fit`):

```julia
using GeneralizedGradients
write_bmad_gg_fit("gg_fit_result.h5")
write_bmad_gg_fit("gg_fit_result.h5"; cutoff = 1e-6)   # prune negligible multipoles
write_bmad_gg_fit(fit, meta)                            # from a loaded fit
```

Two files are written: `<output_base>.bmad` (the lattice element) and
`<output_base>_gg.bmad` (the attached `gen_grad_map`). `cutoff` is a relative
magnitude threshold: a multipole curve is dropped if its peak `|GG|` is below
`cutoff × (largest peak |GG| of any curve)`. The default `0` keeps every
non-zero curve.

From the shell:

```
julia programs/run_gg_to_bmad.jl <gg_fit_result.h5> [output_base] [cutoff]
```

```{note}
Bmad's `gen_grad_map` uses azimuthal-harmonic gradients `C_{m,sin/cos}`, a
different convention from this project's midplane-derivative GGs
(`a_n`, `b_n`, `b_s`). `write_bmad_gg_fit` performs the exact conversion between
the two; the derivation is given in the `write_bmad_gg_fit` docstring (see the
**API Reference**) and summarized under [Theory](theory.md).
```
