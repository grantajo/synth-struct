# synth_struct
This is a code base that generates microstrcutures that are intended to be used in
mechanical simulations, such as ultrasonic, crystal plasticity, etc.

## Installation
### Create a virtual environment
Run the following commands to create a virtual environment<br>
`conda create --name synth`<br>
`conda activate synth`<br>

### Install dependencies in virtual environment
|Package              |Installation                       |Requirement             |
|:-------------------:|:---------------------------------:|:----------------------:|
|numpy                |`conda install numpy`              |Required                |
|scipy                |`conda install scipy`              |Required                |
|matplotlib           |`conda install matplotlib`         |Required                |
|matplotlib-scalebar  |`conda install matplotlib-scalebar`|Required for plotting   |
|orix                 |`conda install orix=0.14.0`        |Required for plotting   |
|pybind11             |`conda install pybind11`           |Required for C++ speedup|
|g++                  |`apt install build-essential gcc`  |Required for C++ speedup|
|Eigen                |`apt install libeigen3-dev`        |Required for C++ speedup|
|pytest               |`conda install pytest`             |Required for tests      |

### Clone repository
Clone the repository from Github

### Make output directory
`cd synth-struct`<br>
`mkdir output`

### Install the project
When in project root run:<br>
`pip install -e .`

### Testing Installation
Once installed, run the following command to run the tests<br>
`pytest tests/`<br>

### Installation notes
The first time running an example, the code may take slightly longer 
than normal due to the program compiling on first run

## Todo Lists:
### Just finished
- Started mask shapes examples
    - There is a bug in the cylindrical mask
    - There is a bug in the layers (the orientation of the layers is incorrect)

### Short-term todo:
- More examples
    - Textures
    - Stiffness (what do I even do for this?)

### Things to be added:
- Continue orix integration
    - Add in IPFs and ODFs
    - Add in ability to have the colorkey shown in an IPF map
        - Add in ability to plot a single given colorkey with ipfcolorkeys.py
    - Figure out how to handle hexagonal directions for pole figures and ODFs
    - Add in ability to do ODF contour plots
    - Add in plotting examples
- Add grain structures with grain size better distribution
- Add in file output generators
    - HDF5
    - VTK meshing
 
### Long term additions:
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
│   ├── __init__.py<br>
│   ├── basic_example_2d.py<br>
│   ├── basic_example_3d.py<br>
│   ├── mask_box.py<br>
│   ├── mask_custom.py<br>
│   ├── mask_cylinder.py<br>
│   ├── mask_layer.py<br>
│   ├── mask_sphere.py<br>
│   ├── shapes.py<br>
│   └── vis_example.py<br>
├── output<br>
├── pyproject.toml<br>
├── setup.py<br>
├── src/<br>
│   ├── synth_struct/<br>
│   │   ├── __init__.py<br>
│   │   ├── generators/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── columnar.py<br>
│   │   │   ├── ellipsoidal.py<br>
│   │   │   ├── gen_base.py<br>
│   │   │   ├── gen_utils.py<br>
│   │   │   ├── lath.py<br>
│   │   │   ├── lath_updated.py<br>
│   │   │   ├── mixed.py<br>
│   │   │   └── voronoi.py<br>
│   │   ├── _cpp_exensions/<br>
│   │   │   ├── __init__.py<br>
│   │   │   └── aniso_voronoi_eigen.cpp<br>
│   │   ├── io/<br>
│   │   │   └── hdf5_writer.py<br>
│   │   ├── micro_utils.py<br>
│   │   ├── microstructure.py<br>
│   │   ├── orientation/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── rotation_converter.py<br>
│   │   │   └── texture/<br>
│   │   │      ├── __init__.py<br>
│   │   │      ├── cubic.py<br>
│   │   │      ├── hexagonal.py<br>
│   │   │      ├── random.py<br>
│   │   │      ├── texture.py<br>
│   │   │      ├── texture_base.py<br>
│   │   │      └── custom.py<br>
│   │   ├── plotting/<br>
│   │   │   ├── __init__.py<br>
│   │   │   ├── gen_plot.py<br>
│   │   │   ├── ipf_maps.py<br>
│   │   │   ├── ipfcolorkeys.py<br>
│   │   │   ├── odf_plot.py<br>
│   │   │   ├── orix_utils.py<br>
│   │   │   ├── plot_utils.py<br>
│   │   │   └── pole_figures.py<br>
│   │   └── stiffness/<br>
│   │       ├── __init__.py<br>
│   │       └── stiffness.py<br>
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


