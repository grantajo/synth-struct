This is a microstructure generator that outputs an HDF5 file with the grain structure and the orientations.

install:
When in project root run:  
pip install -e .

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
- See what adding in ODF textures would be like and what would be gained from it
 
Dependencies
- Numpy
- Scipy
- h5py
- matplotlib
- matplotlib-scalebar
- orix

Optional:
- g++
- Eigen (libeigen3-dev)


To do for branch 'change-to-nparrays'
- Update examples
  - plotting
  - get grains and change texture
- Update plotting
- Update tests
- Update lath generator to spatially put colonies together


Currently doing on 'change-to-nparrays'
- Fix plotting save locations (ipfcolorkeys)
- Creating test files
  - finish texture generators
- Add in crytallography plotting
- Add in IPF, pole figure, and ODF examples
  - Figure out how to handle hexagonal directions for pole figures and ODFs
  - Add in ability to do ODF contour plots
  - Update ipfcolorkeys.py
- Add in texture examples


Done:
- Used linters and formatters to fix some errors.


File structure: 
synth_struct/  
├── README.md  
├── examples  
│   ├── __init__.py  
│   ├── basic_example_2d.py  
│   ├── basic_example_3d.py  
│   ├── mask_box.py  
│   ├── mask_custom.py  
│   ├── mask_cylinder.py  
│   ├── mask_layer.py  
│   ├── mask_sphere.py  
│   ├── shapes.py  
│   └── vis_example.py  
├── output  
├── pyproject.toml  
├── src  
│   ├── synth_struct  
│   │   ├── __init__.py  
│   │   ├── generators  
│   │   │   ├── __init__.py  
│   │   │   ├── columnar.py  
│   │   │   ├── ellipsoidal.py  
│   │   │   ├── gen_base.py  
│   │   │   ├── gen_utils.py  
│   │   │   ├── lath.py  
│   │   │   ├── lath_updated.py  
│   │   │   ├── mixed.py  
│   │   │   └── voronoi.py  
│   │   ├── io  
│   │   │   └── hdf5_writer.py  
│   │   ├── micro_utils.py  
│   │   ├── microstructure.py  
│   │   ├── orientation  
│   │   │   ├── __init__.py  
│   │   │   ├── rotation_converter.py  
│   │   │   └── texture  
│   │   ├── plotting  
│   │   │   ├── __init__.py  
│   │   │   ├── gen_plot.py  
│   │   │   ├── ipf_maps.py  
│   │   │   ├── ipfcolorkeys.py  
│   │   │   ├── odf_plot.py  
│   │   │   ├── orix_utils.py  
│   │   │   ├── plot_utils.py  
│   │   │   └── pole_figures.py  
│   │   └── stiffness  
│   │       ├── __init__.py  
│   │       └── stiffness.py  
└── tests  
    ├── __init__.py  
    ├── test_columnar.py  
    ├── test_ellipsoidal.py  
    ├── test_generator_base.py  
    ├── test_micro_utils.py  
    ├── test_microstructure.py  
    ├── test_mixed.py  
    ├── test_texture.py  
    ├── test_texture_base.py  
    ├── test_texture_cubic.py  
    ├── test_texture_hexagonal.py  
    └── test_voronoi.py  


