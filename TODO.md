
# Todo Lists

## Just finished

- File generator
  - Couldn't get conforming mesh to work. Maybe I'll try again later.

## Currently working on

- Maybe make an example that has orientations to be able to use ParaView with.

## Short-term todo

- Make tests
  - Structured_mesh
  - Phases
- Put out a version 0.1

## Things to be added

- Add in GPU CUDA implementations for anisotropic voronoi assignment and new lath generator when done
  - Won't be able to use Eigen implementation
- Continue orix integration
  - Add in ability to have the colorkey shown in an IPF map
    - Add in ability to plot a single given colorkey with ipfcolorkeys.py
- Add in second phase generation
  - Add in ability to have precipitates
  - Add in porosity
- Add grain structures with better grain size distribution abilities
- Maybe make a power user import subspace such as rotation conversion or symmetry operations?

## Long term additions

- Add grain structures with grain size better distribution
- New lath generator (CA see generators/lath_updated.py)
- Add in ability to create a mesh with the microstructure
- Add in ability to have orientation gradients within grains, subgrains and GNDs
- See what adding in ODF textures would be like and what would be gained from it

## File structure

synth_struct/  
├── README.md  
├── examples/  
│   ├── __init__.py  
│   ├── basic_example_2d.py  
│   ├── basic_example_3d.py  
│   ├── masks.py  
│   ├── middle_mask.py  
│   ├── plotIPFcolorkeys.py  
│   ├── texture_cubic.py  
│   ├── texture_custom.py  
│   └── shapes.py  
├── output  
├── pyproject.toml  
├── setup.py  
├── src/  
│   ├── synth_struct/  
│   │   ├── __init__.py  
│   │   ├── generators/  
│   │   │   ├── __init__.py  
│   │   │   ├── columnar.py  
│   │   │   ├── ellipsoidal.py  
│   │   │   ├── gen_base.py  
│   │   │   ├── gen_utils.py  
│   │   │   ├── lath.py  
│   │   │   ├── lath_updated.py  
│   │   │   ├── mixed.py  
│   │   │   └── voronoi.py  
│   │   ├── _cpp_exensions/  
│   │   │   ├── __init__.py  
│   │   │   └── aniso_voronoi_eigen.cpp  
│   │   ├── io/  
│   │   │   └── hdf5_writer.py  
│   │   ├── micro_utils.py  
│   │   ├── microstructure.py  
│   │   ├── orientation/  
│   │   │   ├── __init__.py  
│   │   │   ├── rotation_converter.py  
│   │   │   └── texture/  
│   │   │      ├── __init__.py  
│   │   │      ├── cubic.py  
│   │   │      ├── hexagonal.py  
│   │   │      ├── random.py  
│   │   │      ├── texture.py  
│   │   │      ├── texture_base.py  
│   │   │      └── custom.py  
│   │   ├── plotting/  
│   │   │   ├── __init__.py  
│   │   │   ├── gen_plot.py  
│   │   │   ├── ipf_maps.py  
│   │   │   ├── ipfcolorkeys.py  
│   │   │   ├── odf_plot.py  
│   │   │   ├── orix_utils.py  
│   │   │   ├── plot_utils.py  
│   │   │   └── pole_figures.py  
│   │   └── stiffness/  
│   │       ├── __init__.py  
│   │       └── stiffness.py  
└── tests/  
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

## Output files

Microstructure + Generator
        │
        ├── Conforming path  ──────────────────────────────────────────┐
        │   Analytic surfaces (.vtp)                                   │
        │       └── Gmsh API                                           │
        │               ├── Tet10 .msh  → FEniCS, deal.ii              │
        │               ├── Tet10 .inp  → Abaqus Standard (elasticity) │
        │               └── Tet10 .inp  → Abaqus CPFEM                 │
        │                                                              │
        └── Regular grid path ─────────────────────────────────────────┘
            Voxel grain_ids
                ├── Hex8 .vti        → DAMASK spectral
                ├── Hex8 .hdf5       → DAMASK native
                ├── Hex8 .inp        → Abaqus Explicit (wave)
                └── Hex8 custom .h5  → your own CPFE solver
