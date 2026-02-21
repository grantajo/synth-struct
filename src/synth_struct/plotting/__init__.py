# synth-struct/src/synth_struct/plotting/__init__.py

"""
Plotting functions
"""

from .plot_utils import (
    shuffle_display_grain_ids,
    get_colony_colormap,
    create_grain_boundary_overlay,
    get_grain_size_colormap,
)
from .gen_plot import Plotter
from .orix_utils import (
    create_crystal_map,
    get_crystal_map_slice,
    subsample_orientations,
    get_grain_average_orientations,
    filter_crystal_map_by_grains,
)

# from .ipfcolorkeys import (
#
# )
from .ipf_maps import (
    get_ipf_rgb,
    plot_ipf_map,
    plot_multiple_ipf_maps,
    create_ipf_map_figure,
    plot_multiple_ipf_slices,
)
from .pole_figures import (
    plot_pole_figure,
    plot_multiple_pole_figures,
    create_pole_figure_axes,
    create_standard_pole_figures,
)
from .odf_plot import plot_odf, create_odf_figure

__all__ = [
    "shuffle_display_grain_ids",
    "get_colony_colormap",
    "create_grain_boundary_overlay",
    "get_grain_size_colormap",
    "Plotter",
    "create_crystal_map",
    "get_crystal_map_slice",
    "subsample_orientations",
    "get_grain_average_orientations",
    "filter_crystal_map_by_grains",
    "get_ipf_rgb",
    "plot_ipf_map",
    "plot_multiple_ipf_maps",
    "create_ipf_map_figure",
    "plot_pole_figure",
    "plot_multiple_pole_figures",
    "create_pole_figure_axes",
    "create_standard_pole_figures",
    "plot_odf",
    "create_odf_figure",
    "plot_multiple_ipf_slices",
]
