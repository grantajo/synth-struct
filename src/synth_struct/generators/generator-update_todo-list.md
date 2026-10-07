# Generator Update List

## Finished

- Added in metadata for generators
- Moved num_grains back out of Microstructure
- Added in target_num_grains to tests
  - Failing some tests now for some reason

## Goals

- Refactor what Microstructure and Generator classes own
  - Change examples to include num_grains in microstructure
  - Change tests to have num_grains in microstructure
- Make a new set of Leguerre generators (in addition to Voronoi)
  - Make it so custom grain size distributions are possible
- Improve seed generation for columnar and equiaxed (only need in plane)

## Specific tasks

- Microstructure
  - Add measure_grain_volumes to Microstructure class

- Examples and tests
  - Add num_grains as a part of Microstructure class initiation
  - Add generator metadata to tests and examples
