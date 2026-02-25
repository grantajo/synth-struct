## Todo Lists:
### Just finished/Currently working on
- Finished adding in hexagonal texture plotting and phases
- Working on making imports easier for examples (__init__.py)
    - Need to change out the imports in the examples to test
    - Maybe make a power user subspace such as rotation conversion
    or symmetry operations, or something of the sort.

### Short-term todo:
- Continue orix integration
    - Add in IPFs and ODFs
    - Add in ability to have the colorkey shown in an IPF map
        - Add in ability to plot a single given colorkey with ipfcolorkeys.py
    - Add in ability to do ODF contour plots
    - Add in plotting examples
        - Maybe don't need to add since there will be plotting examples in 
        other scripts?

### Things to be added:
- Add grain structures with grain size better distribution
- Add in file output generators
    - HDF5
    - VTK meshing
 
### Long term additions:
- Figure out an example for texture
- Add in second phases
    - Add in ability to have precipitates
    - Add in porosity
- Add grain structures with grain size better distribution
- New lath generator (CA see generators/lath_updated.py)
- Add HDF5 and VTK writers
- Add in ability to create a mesh with the microstructure
- Add in ability to have orientation gradients within grains, subgrains and GNDs
- See what adding in ODF textures would be like and what would be gained from it


## File structure: 
synth_struct/<br>
├── README.md<br>
├── examples/<br>
│   ├── __init__.py<br>
│   ├── basic_example_2d.py<br>
│   ├── basic_example_3d.py<br>
│   ├── masks.py<br>
│   ├── middle_mask.py<br>
│   ├── plotIPFcolorkeys.py<br>
│   ├── texture_cubic.py<br>
│   ├── texture_custom.py<br>
│   └── shapes.py<br>
├── output<br>
├── pyproject.toml<br>
├── setup.py<br>
├── src/<br>
│   ├── synth_struct/<br>
│   │   ├── __init__.py<br>
│   │   ├── generators/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── columnar.py<br>
│   │   │   ├── ellipsoidal.py<br>
│   │   │   ├── gen_base.py<br>
│   │   │   ├── gen_utils.py<br>
│   │   │   ├── lath.py<br>
│   │   │   ├── lath_updated.py<br>
│   │   │   ├── mixed.py<br>
│   │   │   └── voronoi.py<br>
│   │   ├── _cpp_exensions/<br>
│   │   │   ├── __init__.py<br>
│   │   │   └── aniso_voronoi_eigen.cpp<br>
│   │   ├── io/<br>
│   │   │   └── hdf5_writer.py<br>
│   │   ├── micro_utils.py<br>
│   │   ├── microstructure.py<br>
│   │   ├── orientation/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── rotation_converter.py<br>
│   │   │   └── texture/<br>
│   │   │      ├── __init__.py<br>
│   │   │      ├── cubic.py<br>
│   │   │      ├── hexagonal.py<br>
│   │   │      ├── random.py<br>
│   │   │      ├── texture.py<br>
│   │   │      ├── texture_base.py<br>
│   │   │      └── custom.py<br>
│   │   ├── plotting/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── gen_plot.py<br>
│   │   │   ├── ipf_maps.py<br>
│   │   │   ├── ipfcolorkeys.py<br>
│   │   │   ├── odf_plot.py<br>
│   │   │   ├── orix_utils.py<br>
│   │   │   ├── plot_utils.py<br>
│   │   │   └── pole_figures.py<br>
│   │   └── stiffness/<br>
│   │       ├── __init__.py<br>
│   │       └── stiffness.py<br>
└── tests/<br>
    ├── __init__.py<br>
    ├── test_columnar.py<br>
    ├── test_ellipsoidal.py<br>
    ├── test_generator_base.py<br>
    ├── test_micro_utils.py<br>
    ├── test_microstructure.py<br>
    ├── test_mixed.py<br>
    ├── test_texture.py<br>
    ├── test_texture_base.py<br>
    ├── test_texture_cubic.py<br>
    ├── test_texture_hexagonal.py<br>
    └── test_voronoi.py<br>
