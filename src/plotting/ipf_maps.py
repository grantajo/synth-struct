# synth_struct/src/plotting/ipf_maps.py

import numpy as np
import maplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

from .orix_utils import create_crystal_map, get_crystal_map_slice

"""
IPF (Inverse Pole Figure) colore map visualization.
"""

def get_ipf_rgb(crystal_map, direction='z'):
    """
    Get RGB colors for IPF map.
    
    Converts crystallographic orientations to RGB colors based on the 
    IPF color key for the specified direction.
    
    Args:
    - crystal_map: orix CrystalMap object containing orientations
    - direction: str - IPF direction ('x', 'y', or 'z')
    
    Returns:
    np.ndarray: RGB color array with shape (N, 3) where N is the number of points
    
    Example:
        from plotting.orix_utils import create_crystal_map
        crystal_map = create_crystal_map(micro, 'cubic')
        rgb = get_ipf_rgb(crystal_map, direction='z')
    """
    
    direction_map = {
        'x': Vector3d.xvector(),
        'y': Vector3d.yvector(),
        'z': Vector3d.zvector()
    }
    
    direction = direction.lower()
    if direction not in direction_map:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'x', 'y', or 'z'"
        )
        
    ipf_direction = direction_map[direction]
    
    symmetry = crystal_map.phases[0].point_group
    
    ipf_key = IPFColorKeyTSL(symmetry, direction=ipf_direction)
    rgb_pixels = ipf_key.orientation2color(crystal_map.rotations)
    
    return rgb_pixels
    
    
def plot_ipf_map(ax, micro, direction='z', crystal_structure='cubic',
                 slice_idx=None, slice_direction='z', grain_subset=None,
                 show_scalebar=True, show_title=False, title=None, scalebar_kwargs=None):
    """
    Plot IPF map on provided axes.
    
    Creates in Inverse Pole Figure map showing grain orientations colred according to the
    IPF color scheme. For 3D microstructures, displays a 2D slice.
    
    Args:
    - ax: matplotlib.axes.Axes to plot on
    - micro: Microstructure object with grain_ids and orientations
    - direction: str - IPF direction ('x', 'y', or 'z')
    - crystal_strcuture: str - Crystal structure: 'cubic', 'fcc', 'bcc', 'hexagonal', 'hcp'
    - slice_idx: int or None - Slice index for 3D microstructures. If None, uses middle slice
    - slice_direction: str - Direction to slice for 3D ('x', 'y', or 'z')
    - grain_subset: np.ndarray or None - Specific grain IDs to visualize. 
                                         Grains not in this list will appear with default orientation
    - show_scalebar: bool - Whether to show scale bar
    - show_title: bool - Whether to show automatic title
    - title: str or None - Custom title. Overrides automatic title if provided
    - scalebar_kwargs: dict or None - Additional kwargs for Scalebar (dx, units, location, etc.)
    
    Returns:
    - matplotlib.image.AxesImage: Image artist
    
    Example:
        fig, ax = plot.subplots()
        plot_ipf_map(ax, micro, direction='z', show_title=True)
        plt.savefig('ipf_z.png')
    """
    
    crystal_map = create_crystal_map(
        micro, crystal_structure, grain_subset=grain_subset
    )
    
    # Get slice if 3D, returns itself if 2D
    crystal_map_slice, shape = get_crystal_map_slice(
        crystal_map, micro.dimensions, slice_idx, slice_direction
    )
        
    rgb_pixels = get_ipf_rgb(crystal_map_slice, direction
    
    ipf_image = rgb_pixels.reshape(*shape, 3)
    
    im = ax.imshow(ipf_image, origin='lower')
    
    # Title
    if title is not None:
        ax.set_title(title)
    elif show_title:
        auto_title = f'IPF-{direction.upper()} Map'
        if len(micro.dimensions) == 3:
            if slice_idx is None:
                slice_idx = micro.dimensions[{'x':0, 'y':1, 'z':2}[slice_direction.lower()]] // 2
            auto_title += f' ({slice_direction}={slice_idx})'
        ax.set_title(auto_title)
        
    ax.axis('off')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        
    if show_scalebar:
        sb_kwargs = {
            'dx': micro.resolution,
            'units': micro.units,
            'location': 'lower right',
            'box_color': 'white',
            'box_alpha': 0.75,
            'color': 'black'
        }
        if scalebar_kwargs:
            sb_kwargs.update(scalebar_kwargs)
            
        scalebar = ScaleBar(**sb_kwargs)
        ax.add_artist(scalebar)
        
    return im
    
    
def plot_multiple_ipf_maps(axes, micro, directions=['x', 'y', 'z'],
                           crystal_structure='cubic', **kwargs):
    """
    Plot multiple IPF maps (different directions on provided axes.
    
    Useful for comparing how grain orientations appear when viewed 
    along different sample directions.
    
    Args:
    - axes: list of maplotlib axes
    - micro: Microstructure object
    - directions: list of str - IPF directions to plot (e.g., ['x', 'y', 'z'])
    - crystal_structure: str - Crystal structure type
    - **kwargs - Additional arguments passed to plot_ipf_maps
    
    Returns:
    list: List of AxesImage objects
    
    Example:
        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        plot_multiple_ipf_maps(axes, micro, directions=['x', 'y', 'z'])
        plt.tight_layout()
    """
    
    if len(axes) != len(directions):
        raise ValueError(
            f"Number of axes ({len(axes)}) must match "
            f"number of directions ({len(directions)})"
        )
        
    images = []
    
    for ax, direction in zip(axes, directions):
        im = plot_ipf_map(ax, micro, direction=direction, crystal_structure=crystal_structure,
                          show_title=True, **kwargs)
        images.append(im)
        
    return images
    

def create_ipf_map_figure(micro, direction='z', crystal_structure='cubic', slice_idx=None,
                          slice_direction='z', grain_subset=None, filename=None, 
                          figsize=(6,6), dpi=150, **kwargs):
    """
    Create standalone IPF map figure.
    
    Convenience function that creates a new figure, plots IPF map, and optionally saves it.
    
    Args:
    - micro: Microstructure object with grain_ids and orientations
    - direction: str - IPF direction ('x', 'y', or 'z')
    - crystal_strcuture: str - Crystal structure: 'cubic', 'fcc', 'bcc', 'hexagonal', 'hcp'
    - slice_idx: int or None - Slice index for 3D microstructures. If None, uses middle slice
    - slice_direction: str - Direction to slice for 3D ('x', 'y', or 'z')
    - grain_subset: np.ndarray or None - Specific grain IDs to visualize. 
                                         Grains not in this list will appear with default orientation
    - filename: str or None: Save filename (relative to ../output/). If None, doesn't save.
    - figsize: tuple - Figure size in inches
    - dpi: int - Resolution for saving in dots per inch
    - **kwargs: Additional arguments passed to plot_ipf_map
    
    Returns:
    - fig: matplotlib Figure object
    - ax: matplotlib Axes object
    
    Example:
        # Simple
        fig, ax = create_ipf_map_figure(micro, direction='z', filename='ipf_z.png')
        
        # With grain subset
        from micro_utils improt get_grains_in_region
        surface_grains = get_grains_in_region(micro, 'box', z_min=90)
        fig, ax = create_ipf_map_figure(micro, grain_subset=surface_grains, filename='surface_ipf.png')
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_ipf_map(
        ax, micro, direction=direction, crystal_structure=crystal_structure, slice_idx=slice_idx,
        slice_direction=slice_direction, grain_subset=grain_subset, **kwargs
    )
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'../output/{filename}', dpi=dpi, bbox_inches='tight')
        print(f"Saved IPF map to ../output/{filename}")
        
    return fig, ax
    
    
