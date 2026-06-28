---
title: Exporting to Bmad
---

# Exporting to Bmad

The package can write two kinds of [Bmad](https://www.classe.cornell.edu/bmad/)
lattice elements: a `grid_field` element (the raw field grid) and a
`gen_grad_map` element (the fitted generalized gradients).

In both cases a non-zero reference bending strength `g_ref = 1/bend_radius`
[1/m] makes the element an `sbend` with `curved_ref_frame = T`; otherwise it is
an `em_field`. Each element is anchored at its entrance
(`ele_anchor_pt = beginning`).

## Field grid → `grid_field`

`grid_to_bmad` reads a field grid and writes it as a Bmad `grid_field`:

```julia
using GeneralizedGradients
grid_to_bmad("field_grid.h5")                      # text grid block
grid_to_bmad("field_grid.h5"; hdf5 = true)         # openPMD HDF5 grid (faster)
grid_to_bmad("field_grid.h5"; g_ref = 0.01)        # force an sbend
```

Two files are written: `<output_base>.bmad` (the lattice element) and the grid,
either `<output_base>_grid.bmad` (plain text) or `<output_base>_grid.h5`
(HDF5). The core writer `write_bmad_field_grid(field; ...)` is also exported and
can be called directly on a `FieldGridTable`.

From the shell:

```
julia programs/run_grid_to_bmad.jl <field_grid.h5> [output_base] [g_ref] [--hdf5]
```

## GG fit → `gen_grad_map`

`gg_to_bmad` converts a GG fit file (the output of `gg_fit_write_results`) into a
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
non-zero curve. The core writer `write_bmad_gen_grad_map(fit; ...)` is exported
too.

From the shell:

```
julia programs/run_gg_to_bmad.jl <gg_fit_result.h5> [output_base] [cutoff]
```

```{note}
Bmad's `gen_grad_map` uses azimuthal-harmonic gradients `C_{m,sin/cos}`, a
different convention from this project's midplane-derivative GGs
(`a_n`, `b_n`, `b_s`). `gg_to_bmad` performs the exact conversion between the two;
the derivation is given in the `gg_to_bmad` docstring in the
[API Reference](https://bmad-sim.github.io/GeneralizedGradients.jl/api/) and
summarized under [Theory](theory.md).
```
