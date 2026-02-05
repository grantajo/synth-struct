# synth_struct/src/synth_struct/plotting/odf_plot.py

import matplotlib.pyplot as plt

from .orix_utils import create_crystal_map

"""
Orientation Distribution Function (ODF) visualization.

ODFs show the distribution of all grain orientations in orientation space,
providing a complete representation of crystallographic texture.
"""


def plot_odf(
    ax,
    micro,
    crystal_structure="cubic",
    grain_subset=None,
    projection="axangle",
    **scatter_kwargs,
):
    """
    Plot Orientation Distribution Function on provided axes.

    Displays all grain orientations in axis-angle sapce (or other provided representation),
    where clustering indicates preferred orientations (texture).

    Args:
    - ax: matplotlib Axes object
    - micro: Microstructure object
    - crystal_structure: str - Crystal structure type ('cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp')
    - grain_subset: np.ndarray or None - Specific grain IDs to include. If None, uses all grains.
    - projection: str - Projection type for ODF (default: 'axangle')
    - **scatter_kwargs: Additional arguments passed to scatter plot (e.g., s=1, c='blue')

    Returns:
    - scatter artist

    Example:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='axangle')
        plot_odf(ax, micro, crystal_structure='cubic', s=2, alpha=0.6)
        plt.savefig('odf.png')
    """

    crystal_map = create_crystal_map(
        micro, crystal_structure, grain_subset=grain_subset
    )

    scatter_defaults = {"s": 1, "alpha": 0.5, "c": "C0"}
    scatter_defaults.update(scatter_kwargs)

    artist = crystal_map.rotations.scatter(ax=ax, **scatter_defaults)

    return artist


def create_odf_figure(
    micro,
    crystal_structure="cubic",
    grain_subset=None,
    projection="axangle",
    filename=None,
    figsize=(5, 5),
    dpi=150,
    title=None,
    **kwargs,
):
    """
    Create standalone ODF plot.

    Convenience function that creates a figure with an ODF and optionally saves it.

    Args:
    - micro: Microstructure object
    - crystal_structure: str - Crystal structure type ('cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp')
    - projection: str - Projection type for ODF (default: 'axangle')
    - grain_subset: np.ndarray or None - Specific grain IDs to include. If None, uses all grains.
    - filename: str or None - Save filename (relative to ../output/). If None, doesn't save.
    - figsize: tuple - Figure size in inches
    - dpi: int - Resolution for saving figures
    - title: str or None - Custom title. If None, uses default title.
    - **kwargs: Additional arguments passed to plot_odf

    Returns:
    - fig: matplotlib Figure object
    - ax: matplotlib Axes object

    Example:
        fig, ax = create_odf_figure(micro, filename='odf.png')
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)

    plot_odf(
        ax,
        micro,
        crystal_structure=crystal_structure,
        grain_subset=grain_subset,
        **kwargs,
    )

    if title is None:
        title = "Orientation Distribution Function"
        if grain_subset is not None:
            title += f" ({len(grain_subset)} grains)"

    ax.set_title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(f"../output/{filename}", dpi=dpi, bbox_inches="tight")
        print(f"Saved ODF to ../output/{filename}")

    return fig, ax
