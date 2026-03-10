# synth-struct/src/synth_struct/plotting/pole_figures.py

import numpy as np
import matplotlib.pyplot as plt
from orix.vector import Vector3d
from orix.plot import register_projections

from .orix_utils import create_crystal_map

"""
Inverse Pole Figure visualizations

Pole figures show the distribution of crystallographic orientations
relative to reference directions.
"""

register_projections()


def plot_ipf(
    ax,
    micro,
    phase_id=None,
    grain_subset=None,
    crystal_map=None,
    direction=None,
    show_labels=True,
    sample_fraction=None,
    plot_type="scatter",
    marker_size=15,
    sigma=5,
    **kwargs,
):
    """
    Plot an inverse pole figure on provided axes.

    Creates an IPF projection showing the distribution
    of orientations in a specified direction.

    Args:
    - ax: matplotlib Axes object
    - micro: Microstructure class object
    - phase_id: ID for Phase being analyzed
    - crystal_map: Orix CrystalMap holder if already created
    - direction: tuple - Sample direction vectors
        (0, 0, 1) for normal direction (ND)
        (1, 0, 0) for rolling direction (RD)
        (0, 1, 0) for transverse direction (TD)
    - grain_subset: np.ndarray or None - Grain mask. If None, uses all grains
    - show_labels: bool - Whether to show axis labels
    - sample_fraction: float or None - Fraction of orientations to plot (0-1)
    - plot_type: str - 'scatter' or 'density'
    - marker_size: float - Size of scatter points
    - sigma: float - Angular resolution of broadening in degrees
    **kwargs: Additional elements passed to orix/matplotlib plotting functions

    Returns:
    - matplotlib plot artist
    """

    if crystal_map is None:
        crystal_map = create_crystal_map(
            micro,
            grain_subset=grain_subset,
        )

    if phase_id is None:
        phase_id = next(i for i in crystal_map.phases.ids if i >= 0)

    phase = crystal_map.phases[phase_id]

    phase_mask = crystal_map.phase_id == phase_id
    orientations = crystal_map.orientations[phase_mask]

    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError(f"sample_fraction must be in (0,1], got {sample_fraction}")
        n = orientations.size
        n_sample = int(sample_fraction * n)
        idx = np.random.choice(n, size=n_sample, replace=False)
        orientations = orientations[idx]

    sample_dir = Vector3d(np.array(direction, dtype=float))
    crystal_dirs = orientations.outer(sample_dir)

    crystal_dirs = crystal_dirs.in_fundamental_sector(phase.point_group)

    xyz = crystal_dirs.data.reshape(-1, 3).copy()
    xyz[xyz[:, 2] < 0] *= -1
    crystal_dirs = Vector3d(xyz)

    if plot_type == "scatter":
        scatter_defaults = {"s": marker_size, "c": "C0", "alpha": 0.5}
        scatter_defaults.update(kwargs)
        ipf = ax.scatter(crystal_dirs, **scatter_defaults)
    elif plot_type == "density":
        density_defaults = {"sigma": sigma, "cmap": "jet"}
        density_defaults.update(kwargs)
        ipf = ax.pole_density_function(crystal_dirs, **density_defaults)
    else:
        raise ValueError(f"plot_type must be 'scatter' or 'density', got '{plot_type}'")

    # TODO:
    # Find out how to add labels to IPFs

    x, y, z = direction
    ax.set_title(rf"IPF $[{x}, {y}, {z}]$")

    return ipf


def plot_multiple_ipfs(
    axes,
    micro,
    directions,
    phase_id=None,
    grain_subset=None,
    show_labels=True,
    sample_fraction=None,
    plot_type="scatter",
    **plot_kwargs,
):
    """
    Plot multiple IPFs on provided axes.

    Creates a series of IPFs for different sample directions,
    typically ND, RD, and TD for a complete texture description.

    Args:
    - axes: list of matplotlib Axes objects with stereographic projection
    - micro: Microstructure object
    - sample_directions: list of tuples - Sample directions to plot, e.g.
        [(0,0,1), (1,0,0), (0,1,0)] for ND, RD, TD
    - phase_id: int or None - Phase ID to plot. If None, uses first valid phase
    - grain_subset: np.ndarray or None - Grain IDs to include. If None, uses all
    - show_labels: bool - Whether to show axis labels
    - sample_fraction: float or None - Fraction of orientations to plot (0-1)
    - plot_type: str - 'scatter' or 'density'
    **plot_kwargs: Additional arguments passed to plot_inverse_pole_figure

    Returns:
    - list: List of matplotlib artists
    """

    if len(axes) != len(directions):
        raise ValueError(
            f"Number of axes ({len(axes)}) must match "
            f"number of Miller indices ({len(directions)})"
        )

    crystal_map = create_crystal_map(
        micro,
        grain_subset=grain_subset,
    )

    artists = []
    for ax, direction in zip(axes, directions):
        artist = plot_ipf(
            ax,
            micro,
            direction,
            phase_id=phase_id,
            grain_subset=grain_subset,
            crystal_map=crystal_map,
            show_labels=show_labels,
            sample_fraction=sample_fraction,
            plot_type=plot_type,
            **plot_kwargs,
        )
        artists.append(artist)

    return artists


def create_ipf_axes(
    fig,
    n_figures,
    projection="ipf",
    layout="row",
):
    """
    Creates multiple IPF projection axes for inverse pole figures.

    Args:
    - fig: matplotlib Figure object
    - n_figures: int - Number of inverse pole figures to create
    - projection: str - Projection type (default: 'ipf')
    - layout: str - 'row', 'column', or 'grid'

    Returns:
    - list: List of axes objects with ipf projection
    """

    if layout == "row":
        nrows, ncols = 1, n_figures
    elif layout == "column":
        nrows, ncols = n_figures, 1
    elif layout == "grid":
        ncols = int(np.ceil(np.sqrt(n_figures)))
        nrows = int(np.ceil(n_figures / ncols))
    else:
        raise ValueError(f"Layout must be 'row', 'column' or 'grid', got '{layout}'")

    axes = []
    for i in range(n_figures):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=projection)
        axes.append(ax)

    return axes
