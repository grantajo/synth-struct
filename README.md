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

Currently doing on 'change-to-nparrays'
- Finish updating orientation converter
- Finish adding ellipsoidal

To do for branch 'change-to-nparrays'
- Add in other microstructure generator types
  - Ellipsoidal
  - Columnar
  - Mixed
- Update texture capabilities
- Update examples
- Update plotting
- Update tests
- Update get_grain_ids in region function
- Remove unnecessary stuff in microstructure.py


File structure:
project_root/ (synth_struct)
├── README.md
├── src/
│   ├── __init__.py
│   ├── grain_utils.py # Utility functions such as getting the grains IDs from a specific region
│   ├── microstructure.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py # Houses the microstructure generator base class
│   │   ├── gen_utils.py # Utility functions
│   │   ├── voronoi.py
│   │   ├── ellipsoidal.py
│   │   ├── columnar.py
│   │   └── mixed.py
│   ├── orientation/
│   │   ├── __init__.py
│   │   ├── rotation_converter.py # Conversions between orientation standards (eu, quat, rotmats)
│   │   └── texture.py # Creates a set of textures for a given microstructure or set of grains (may switch to separate files?)
│   ├── stiffness/
│   │   ├── __init__.py
│   │   └── stiffness.py # Rotates the stiffness matrix for each grain based on the orientation of the grain
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── plotting.py # Functions to plot IPF maps, pole figures, etc.
│   │   └── ipfcolorkeys.py # Functions to plot the series of IPF color keys from orix
│   └── io/
│       ├── __init__.py
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




