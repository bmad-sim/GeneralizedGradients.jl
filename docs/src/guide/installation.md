# Installation

GeneralizedGradients.jl targets Julia **1.10 or newer** (see the `[compat]`
section of `Project.toml`).

## From the Julia registry / GitHub

The package is hosted at
[bmad-sim/GeneralizedGradients.jl](https://github.com/bmad-sim/GeneralizedGradients.jl).
Until it is in the General registry, add it by URL:

```julia
using Pkg
Pkg.add(url = "https://github.com/bmad-sim/GeneralizedGradients.jl")
```

Then load it:

```julia
using GeneralizedGradients
```

## For development

Clone the repository and `dev` it into your environment:

```julia
using Pkg
Pkg.develop(path = "/path/to/GeneralizedGradients")
```

Run the test suite with:

```julia
Pkg.test("GeneralizedGradients")
```

## Dependencies

The package relies on `HDF5` (field-grid and fit-result I/O), `OffsetArrays`
(grids whose indices need not start at 0/1), `LinearAlgebra` (the least-squares
solve), plus `Printf`, `Dates`, `EnumX`, and `Symbolics`. These are installed
automatically by the package manager.

## Example data

The `examples/` directory contains a runnable workflow (`examples/run_gg_fit.jl`)
and a reduced field map (`examples/wsnk_fieldmap_reduced.h5`) used throughout
this guide.
