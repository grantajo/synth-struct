# synth-struct/src/synth_struct/__init__.py

"""
synth-struct: A package for generating synthetic microstructures
with crystallographic textures.
"""

from .microstructure import Microstructure
from .micro_utils import get_grains_in_region

from .generators.voronoi import VoronoiGenerator
from .generators.ellipsoidal import EllipsoidalGenerator
from .generators.mixed import MixedGenerator
from .generators.columnar import ColumnarGenerator
from .generators.lath import LathGenerator

from .orientation.phase import Phase
from .orientation.texture.random import RandomTexture
from .orientation.texture.cubic import CubicTexture
from .orientation.texture.hexagonal import HexagonalTexture
from .orientation.texture.custom import CustomTexture

from .stiffness.isotropic_stiffness import IsotropicStiffnessGenerator
from .stiffness.cubic_stiffness import CubicStiffnessGenerator
from .stiffness.hexagonal_stiffness import HexagonalStiffnessGenerator

from .plotting.gen_plot import Plotter
from .plotting.ipf_maps import (
    plot_ipf_map,
    plot_multiple_ipf_maps,
    plot_multiple_ipf_slices,
)
from .plotting.ipfcolorkeys import plot_all_colorkeys

# from .plotting.pole_figures import ...
# from .plotting.odf_plot import ...


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
    "plot_ipf_map",
    "plot_multiple_ipf_maps",
    "plot_multiple_ipf_slices",
    "plot_all_colorkeys",
]
