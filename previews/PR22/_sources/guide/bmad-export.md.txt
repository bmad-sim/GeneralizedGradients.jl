# Exporting to Bmad

The package can write two kinds of [Bmad](https://www.classe.cornell.edu/bmad/)
lattice elements: a `grid_field` element (the raw field grid) and a
`gen_grad_map` element (the fitted generalized gradients).

In both cases a non-zero reference bending strength `g_ref = 1/bend_radius`
[1/m] makes the element an `sbend` with `curved_ref_frame = T`; otherwise it is
an `em_field`. Each element is anchored at its entrance
(`ele_anchor_pt = beginning`).

## Field grid → `grid_field`

`field_grid_to_bmad` writes a field grid as a Bmad `grid_field`. Its `input` is
either a `FieldGridTable` or the path to a `field_grid` HDF5 file:

```julia
using GeneralizedGradients
field_grid_to_bmad("field_grid.h5")                  # openPMD HDF5 grid (default)
field_grid_to_bmad("field_grid.h5"; hdf5 = false)    # plain-text grid block
field_grid_to_bmad(field)                             # from a FieldGridTable
```

The reference frame is determined by the grid's own `g_ref`: non-zero gives an
`sbend`, zero an `em_field`.

Two files are written: `<output_base>.bmad` (the lattice element) and the grid,
either `<output_base>_grid.h5` (HDF5, the default) or `<output_base>_grid.bmad`
(plain text). The core writer `write_bmad_field_grid(field; ...)` is also exported
and can be called directly on a `FieldGridTable`.

From the shell:

```
julia programs/run_field_grid_to_bmad.jl <field_grid.h5> [output_base] [--text]
```

## GG fit → `gen_grad_map`

`gg_to_bmad` converts a GG fit file (the output of `write_gg_fit`) into a
Bmad `gen_grad_map`:

```julia
using GeneralizedGradients
gg_to_bmad("gg_fit_result.h5")
gg_to_bmad("gg_fit_result.h5"; cutoff = 1e-6)   # prune negligible multipoles
```

Two files are written: `<output_base>.bmad` (the lattice element) and
`<output_base>_gg.bmad` (the attached `gen_grad_map`). `cutoff` is a relative
magnitude threshold: a multipole curve is dropped if its peak `|GG|` is below
`cutoff × (largest peak |GG| of any curve)`. The default `0` keeps every
non-zero curve. The core writer `write_bmad_gen_grad_map(fit, meta; ...)` is
exported too (`fit`, `meta` as returned by `read_gg_fit`).

From the shell:

```
julia programs/run_gg_to_bmad.jl <gg_fit_result.h5> [output_base] [cutoff]
```

```{note}
Bmad's `gen_grad_map` uses azimuthal-harmonic gradients `C_{m,sin/cos}`, a
different convention from this project's midplane-derivative GGs
(`a_n`, `b_n`, `b_s`). `gg_to_bmad` performs the exact conversion between the two;
the derivation is given in the `gg_to_bmad` docstring (see the **API Reference**)
and summarized under [Theory](theory.md).
```
