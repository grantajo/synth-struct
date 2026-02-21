# synth-struct/src/synth_struct/plotting/pole_figures.py

import numpy as np
import matplotlib.pyplot as plt
from orix.vector import Miller
from .orix_utils import create_crystal_map

"""
Pole figure visualization.

Pole figures show the distribution of crystallographic directions in stereographic projection
to analyze crystallographic texture.
"""


def plot_pole_figure(
    ax,
    micro,
    miller_index,
    crystal_structure="cubic",
    grain_subset=None,
    show_labels=True,
    sample_fraction=None,
    marker_size=1,
    **scatter_kwargs,
):
    """
    Plot a pole figure on provided axes.

    Creates a stereographic projection showing the distribution of a specific crystallographic
    direction across all grain orientations.

    Args:
    - ax: matplotlib Axes object
    - micro: Microstructure object
    - miller_index: tuple - Miller indices (h, k, l) for the pole to plot
    - crystal_structure: str - 'cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp'
    - grain_subset: np.ndarray or None - Grain mask. If None, uses all grains
    - show_labels: bool - Whether to show axis labels (X, Y, Z)
    - sample_fraction: float or None - Fraction of orientations to plot (0-1). Useful for large datasets
    - marker_size: float - Size of scatter points
    **scatter_kwargs: Additional elements passed to ax.scatter (e.g., c='red', alpha=0.5, cmap='viridis')

    Returns:
    - matplotlib scatter plot artist

    Example:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='stereographic')
        plot_pole_figure(ax, micro, (1,0,0), show_labels=True)
        plt.savefig('pf_100.png')
    """

    crystal_map = create_crystal_map(
        micro, crystal_structure, grain_subset=grain_subset
    )

    phase = crystal_map.phases[0]
    miller = Miller(uvw=miller_index, phase=phase)

    orientations = crystal_map.orientations

    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError(f"sample_fraction must be in (0,1], got {sample_fraction}")
        n = orientations.size
        n_sample = int(sample_fraction * n)
        idx = np.random.choice(n, size=n_sample, replace=False)
        orientations = orientations[idx]

    scatter_defaults = {"s": marker_size, "c": "C0", "alpha": 0.5}
    scatter_defaults.update(scatter_kwargs)

    pf = ax.scatter(orientations.outer(miller), **scatter_defaults)

    if show_labels:
        ax.set_labels("X", "Y", "Z")

    h, k, l = miller_index
    ax.set_title(rf"$\{{{h}\,{k}\,{l}\}}$ Pole Figure")

    return pf


def plot_multiple_pole_figures(
    axes,
    micro,
    miller_indices,
    crystal_structure="cubic",
    grain_subset=None,
    show_labels=True,
    sample_fraction=None,
    **plot_kwargs,
):
    """
    Plot multiple pole figures on provided axes.

    Creates a series of pole figures for different crystallographic directions,
    useful for comprehensive texture analysis.

    Args:
    - axes: lis of matplotlib Axes object
    - micro: Microstructure object
    - miller_index: list of tuples - List of Miller indices to plot (e.g., [(1,0,0), (1,1,0), (1,1,1)]
    - crystal_structure: str - 'cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp'
    - grain_subset: np.ndarray or None - Grain mask. If None, uses all grains
    - show_labels: bool - Whether to show axis labels (X, Y, Z)
    - sample_fraction: float or None - Fraction of orientations to plot (0-1). Useful for large datasets
    - marker_size: float - Size of scatter points
    **plot_kwargs: Additional elements passed to plot_pole_figure for each subplot

    Returns:
    - list: List of scatter plot artists

    Example:
        fig = plt.figure(figsize=(15,5))
        axes = []
        for i, hkl in enumerate([(1,0,0), (1,1,0), (1,1,1)]):
            ax = fig.add_subplot(1, 3, i+1, projection='stereographic')
            axes.append(ax)
        plot_multiple_pole_figures(axes, [(1,0,0), (1,1,0), (1,1,1)], micro)
    """

    if len(axes) != len(miller_indices):
        raise ValueError(
            f"Number of axes ({len(axes)}) must match "
            f"number of Miller indices ({len(miller_indices)})"
        )

    artists = []
    for ax, hkl in zip(axes, miller_indices):
        artist = plot_pole_figure(
            ax,
            hkl,
            micro,
            crystal_structure=crystal_structure,
            grain_subset=grain_subset,
            show_labels=show_labels,
            sample_fraction=sample_fraction,
            **plot_kwargs,
        )
        artists.append(artist)

    return artists


def create_pole_figure_axes(fig, n_figures, projection="stereographic", layout="row"):
    """
    Creates multiple stereographic projection axes.

    Helper function to create properly configured axes for pole figures.

    Args:
    - fig: matplotlib Figure object
    - n_figures: int - Number of pole figures to create
    - projection: str - Projection type (default: 'stereographic')
    - layout: str - Subplot layout
        - 'row': Single row of subplots
        - 'column': Single column of subplots
        - 'grid': Roughly square grid of subplots

    Returns:
    - list: List of axes objects with stereographic projection

    Example:
        fig = plt.figure(figsize=(12,4))
        axes = create_pole_figure_axes(fig, 3, layout='row')
        for ax, hkl in zip(axes, [(1,0,0), (1,1,0), (1,1,1)]):
            plot_pole_figure(ax, micro, hkl)
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


def create_standard_pole_figures(
    micro, crystal_structure="cubic", filename=None, **kwargs
):
    """
    Create standard pole figures set for a given crystal structure.

    Automaticaly selects the most commonly analyzed poles for the specified crystal structure.

    Args:
    - micro: Microstructure object
    - crystal_structure: str - Crystal structure type ('cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp')
    - filename: str or None - Save filename
    - **kwargs: Additional arguments passed to plot_multiple_pole_figures

    Example:
        fig, axes = create_standard_pole_figures(
            micro,
            crystal_structure='cubic',
            filename='standard_pf.png'
        )
    """

    standard_poles = {
        "cubic": [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        "fcc": [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        "bcc": [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
        "hexagonal": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
        "hcp": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],  # Need to fix
    }

    crystal_structure = crystal_structure.lower()
    if crystal_structure not in standard_poles:
        raise ValueError(
            f"Unknown crystal structure: {crystal_structure}. "
            f"Supported: {list(standard_poles.keys())}"
        )

    miller_indices = standard_poles[crystal_structure]

    fig = plt.figure(figsize=(15, 5))
    axes = create_pole_figure_axes(fig, 3, layout="row")
    plot_multiple_pole_figures(
        axes, micro, miller_indices, crystal_structure=crystal_structure, **kwargs
    )

    plt.tight_layout()

    if filename:
        plt.savefig(f"../output/{filename}", dpi=150, bbox_inches="tight")
        print(f"Saved pole figures to ../output/{filename}")

    return fig, axes
