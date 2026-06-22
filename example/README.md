## Overview

This example starts with a field grid from which gg function values are created. 
From the gg function values, field and vector potential values are calculated.

## Preliminary: Create a field data file for use in this example

File created: `example/wsnk_fieldmap_reduced.jld2`
This is a reduced field map file containing the first 12 planes of the full field map.
This speeds things up.

Note: jld2 data files are actually larger than the Julia ASCII data files they come from. 
It is unclear why, however the speed at which data can be read in is much faster.

Created via:
```
julia> using JLD2, OffsetArrays
julia> include("ags-snakes/wsnk_fieldmap.jl")       # Full table 
julia> pt = OffsetArray(pt[:,:,0:11], 0, 0, -1);    # Truncate for test
julia> save("wsnk_fieldmap_reduced.jld2", Dict("r0_grid" => r0_grid, "dr_grid" =>  dr_grid, "pt" => pt))
```

To read back in use:
```
julia> field = load("wsnk_fieldmap_reduced.jld2")
```

## Create a GG fit file.


