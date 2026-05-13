
# synth-struct (synthetic microstructures)

This is a code base that generates microstrcutures that are intended to be used in
mechanical simulations, such as ultrasonic, crystal plasticity, etc.

This project is licensed under the terms of the MIT license.

## Installation

It is recommended that you install via Linux (WSL works well)

### Create a virtual environment

Run the following commands to create a virtual environment using conda
`conda create --name synth`  
`conda activate synth`  

### Install dependencies in virtual environment

|Package              |Installation                          |Requirement                  |
|:-------------------:|:------------------------------------:|:---------------------------:|
|numpy                |`conda install numpy`                 |Required                     |
|scipy                |`conda install scipy`                 |Required                     |
|matplotlib           |`conda install matplotlib`            |Required for plotting        |
|matplotlib-scalebar  |`conda install matplotlib-scalebar`   |Required for plotting        |
|orix                 |`conda install orix=0.14.0`           |Required for plotting        |
|h5py                 |`conda install h5py`                  |Required for HDF5 output     |
|pyvista              |`conda install pyvista`               |Required for meshing outputs |
|pytest               |`conda install pytest`                |Required for tests           |
|pip                  |`conda install pip`                   |Required for editable install|
|pybind11             |`conda install pybind11`              |Required for C++ speedup     |
|g++                  |`sudo apt install build-essential gcc`|Required for C++ speedup*    |
|Eigen                |`sudo apt install libeigen3-dev`      |Required for C++ speedup*    |

*For g++ and Eigen, it is preferable to install them outside of the conda virtual environment

### Clone repository

Clone the repository from Github

### Make output directory

`cd synth-struct`  
`mkdir output`

### Install the project

When in project root run to install as editable package:  
`pip install -e .`

### Testing Installation

Once installed, run the following command to run the tests  
`pytest tests/`  

### Installation notes

The first time running an example, the code may take slightly longer than normal due to the program compiling on first run
