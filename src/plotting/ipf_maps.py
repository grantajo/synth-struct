# synth_struct/src/plotting/ipf_maps.py

import numpy as np
import maplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

from .orix_utils import create_crystal_map, get_crystal_map_slice

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
    
    
    Example:
    """
