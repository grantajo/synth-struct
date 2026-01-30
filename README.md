This is a microstructure generator that outputs an HDF5 file with the grain structure and the orientations.

Things to be added:
- Add grain structures with grain size better distribution
- Fix HDF5 writer so that a software can read it
- Integrate with orix for better orientation information and plotting
  - Redo IPF maps
  - Add in IPFs and ODFs
  - Add in ability to have the colorkey shown in an IPF map
- Add in ability to have porosity
- Make sure there is the ability to have a second phase with a different stiffness tensor
    - Add in ability to have precipitates
 
Long term additions:
- Add in ability to have orientation gradients within grains, subgrains and GNDs
- Add in ability to create a mesh with the microstructure
 
Dependencies
- Numpy
- Scipy
- h5py
- matplotlib
- matplotlib-scalebar
- orix


To do for branch 'change-to-nparrays'
- Update texture capabilities
- Update examples
  - plotting
  - get grains and change texture
- Update plotting
- Update tests
- Update get_grain_ids in region function
- Remove unnecessary stuff in microstructure.py
- Update lath generator to spatially put colonies together


Currently doing on 'change-to-nparrays'
- Adding in texture capabilities
  - See if Generator scheme would be useful


Done:
- Added in some plotting capabilities (gen_plot)
- Finished shapes example


File structure:\n
project_root/ (synth_struct)\n
├── README.md\n
├── src/\n
│   ├── __init__.py
│   ├── micro_utils.py # Utility functions such as getting the grains IDs from a specific region\n
│   ├── microstructure.py # Houses the Microstructure base class\n
│   ├── generators/\n
│   │   ├── __init__.py\n
│   │   ├── gen_base.py # Houses the MicrostructureGenerator base class\n
│   │   ├── gen_utils.py # Utility functions\n
│   │   ├── voronoi.py\n
│   │   ├── ellipsoidal.py\n
│   │   ├── columnar.py\n
│   │   └── mixed.py\n
│   ├── orientation/\n
│   │   ├── __init__.py\n
│   │   ├── rotation_converter.py # Conversions between orientation standards (eu, quat, rotmats)
│   │   └── texture.py # Creates a set of textures for a given microstructure or set of grains (may switch to separate files?)
│   ├── stiffness/
│   │   ├── __init__.py
│   │   ├── stif_utils.py # Helper functions for stiffness rotations
│   │   └── stiffness.py # Rotates the stiffness matrix for each grain based on the orientation of the grain
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── plot_utils.py # Helper functions for plotting
│   │   ├── gen_plot.py # General plotting functions (Grain IDs, etc.)
│   │   ├── orix_plot.py # Plotting using orix for IPFs, Pole figures, ODFs, etc.
│   │   └── ipfcolorkeys.py # Functions to plot the series of IPF color keys from orix
│   └── io/
│       ├── __init__.py
│       ├── write_utils.py # Helper functions for outputting files
│       ├── vtk_writer.py # Write a mesh to VTK file
│       └── hdf5_writer.py # Write to HDF5 file
├── examples/
│   ├── __init__.py
│   ├── basic_example_2d.py # Basic examples
│   ├── basic_example_3d.py # Basic examples
│   ├── shapes.py # Examples for each of the microstructure generator types
│   └── vis_example.py # Examples for plotting
├── output/
└── tests/
    ├── __init__.py
    ├── test_microstructure.py
    ├── test_rotations.py
    ├── test_stiffness.py
    └── test_texture.py




