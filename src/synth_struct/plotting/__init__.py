# synth-struct/src/synth_struct/plotting/__init__.py

"""
Plotting functions
"""

from .gen_plot import Plotter
from .inverse_pole_figures import plot_ipf, plot_multiple_ipfs

# from .ipfcolorkeys import (
#
# )
from .ipf_maps import (
    create_ipf_map_figure,
    get_ipf_rgb,
    plot_ipf_map,
    plot_multiple_ipf_maps,
    plot_multiple_ipf_slices,
)
from .orix_utils import (
    create_crystal_map,
    filter_crystal_map_by_grains,
    get_crystal_map_slice,
    get_grain_average_orientations,
    subsample_orientations,
)
from .plot_utils import (
    create_grain_boundary_overlay,
    get_colony_colormap,
    get_grain_size_colormap,
    shuffle_display_grain_ids,
)
from .pole_figures import (
    create_pole_figure_axes,
    create_standard_pole_figures,
    plot_multiple_pole_figures,
    plot_pole_figure,
)

__all__ = [
    "Plotter",
    "create_crystal_map",
    "create_grain_boundary_overlay",
    "create_ipf_map_figure",
    "create_pole_figure_axes",
    "create_standard_pole_figures",
    "filter_crystal_map_by_grains",
    "get_colony_colormap",
    "get_crystal_map_slice",
    "get_grain_average_orientations",
    "get_grain_size_colormap",
    "get_ipf_rgb",
    "plot_ipf",
    "plot_ipf_map",
    "plot_multiple_ipf_maps",
    "plot_multiple_ipf_slices",
    "plot_multiple_ipfs",
    "plot_multiple_pole_figures",
    "plot_pole_figure",
    "shuffle_display_grain_ids",
    "subsample_orientations",
]
