# synth-struct/src/synth_struct/__init__.py

"""
synth-struct: A package for generating synthetic microstructures
with crystallographic textures.
"""

from .generators.columnar import ColumnarGenerator
from .generators.ellipsoidal import EllipsoidalGenerator
from .generators.lath import LathGenerator
from .generators.mixed import MixedGenerator
from .generators.voronoi import VoronoiGenerator
from .io.structured_mesh import (
    ElementType,
    SolverFormat,
    export_microstructure,
)
from .micro_utils import get_grains_in_region
from .microstructure import Microstructure
from .orientation.phase import Phase
from .orientation.texture.cubic import CubicTexture
from .orientation.texture.custom import CustomTexture
from .orientation.texture.hexagonal import HexagonalTexture
from .orientation.texture.random import RandomTexture
from .plotting import (
    inverse_pole_figures,
    ipf_maps,
    pole_figures,
)
from .plotting.gen_plot import Plotter
from .plotting.ipfcolorkeys import plot_all_colorkeys
from .stiffness.cubic_stiffness import CubicStiffnessGenerator
from .stiffness.hexagonal_stiffness import HexagonalStiffnessGenerator
from .stiffness.isotropic_stiffness import IsotropicStiffnessGenerator

__all__ = [
    # Core
    "Microstructure",
    "get_grains_in_region",
    # Microstructure Generators
    "VoronoiGenerator",
    "EllipsoidalGenerator",
    "MixedGenerator",
    "ColumnarGenerator",
    "LathGenerator",
    # Phase
    "Phase",
    # Texture Generators
    "RandomTexture",
    "CubicTexture",
    "HexagonalTexture",
    "CustomTexture",
    # Stiffness Generators
    "IsotropicStiffnessGenerator",
    "CubicStiffnessGenerator",
    "HexagonalStiffnessGenerator",
    # Plotting
    "Plotter",
    "ipf_maps",
    "pole_figures",
    "inverse_pole_figures",
    "plot_all_colorkeys",
    # IO
    "export_microstructure",
    "SolverFormat",
    "ElementType",
]
