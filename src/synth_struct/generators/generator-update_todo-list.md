# Generator Update List

## Goals

- Refactor what Microstructure and Generator classes own
  - Move num_grains to Microstructure class
- Make a new set of Leguerre generators (in addition to Voronoi)
  - Make it so custom grain size distributions are possible
- Improve seed generation for columnar and equiaxed (only need in plane)

## Specific tasks

- Microstructure
  - Move _allocate_grain_arrays from gen_base to microstructure
  - Add measure_grain_volumes to Microstructure class

- Examples and tests
  - Add num_grains as a part of Microstructure class initiation
