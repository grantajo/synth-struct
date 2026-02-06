# /synth_struct/setup.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os

# Find Eigen
eigen_include = None
possible_eigen_paths = [
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
    '/opt/homebrew/include/eigen3',
    'C:/eigen3',
]

for path in possible_eigen_paths:
    if os.path.exists(path):
        eigen_include = path
        break

if eigen_include is None:
    print("Warning: Eigen not found. Trying default path.")
    eigen_include = '/usr/include/eigen3'

ext_modules = [
    Pybind11Extension(
        "synth_struct._cpp_extensions.aniso_voronoi_eigen",  # Full module path
        ["src/synth_struct/_cpp_extensions/aniso_voronoi_eigen.cpp"],
        include_dirs=[eigen_include],
        extra_compile_args=[
            '-O3',
            '-march=native',
            '-ffast-math',
            '-DEIGEN_NO_DEBUG',
            '-DEIGEN_DONT_PARALLELIZE',
            '-fopenmp' if sys.platform != 'win32' else '/openmp',
        ],
        extra_link_args=[
            '-fopenmp' if sys.platform != 'win32' else '',
        ],
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
