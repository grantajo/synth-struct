
# synth-struct (synthetic microstructures)

This is a code base that generates microstrcutures that are intended to be used in
mechanical simulations, such as ultrasonic, crystal plasticity, etc.

This project is licensed under the terms of the MIT license.

## Installation

### Create a virtual environment

Run the following commands to create a virtual environment  
`conda create --name synth`  
`conda activate synth`  

### Install dependencies in virtual environment

|Package              |Installation                       |Requirement                  |
|:-------------------:|:---------------------------------:|:---------------------------:|
|numpy                |`conda install numpy`              |Required                     |
|scipy                |`conda install scipy`              |Required                     |
|matplotlib           |`conda install matplotlib`         |Required                     |
|matplotlib-scalebar  |`conda install matplotlib-scalebar`|Required for plotting        |
|h5py                 |`conda install h5py`               |Required for HDF5 output     |
|pyvista              |`conda install pyvista`            |Required for meshing outputs |
|gmsh                 |`pip install h5py`                 |Required for meshing outputs |
|orix                 |`conda install orix=0.14.0`        |Required for plotting        |
|pybind11             |`conda install pybind11`           |Required for C++ speedup     |
|g++                  |`apt install build-essential gcc`  |Required for C++ speedup     |
|Eigen                |`apt install libeigen3-dev`        |Required for C++ speedup     |
|pytest               |`conda install pytest`             |Required for tests           |

### Clone repository

Clone the repository from Github

### Make output directory

`cd synth-struct`  
`mkdir output`

### Install the project

When in project root run:  
`pip install -e .`

### Testing Installation

Once installed, run the following command to run the tests  
`pytest tests/`  

### Installation notes

The first time running an example, the code may take slightly longer than normal due to the program compiling on first run
