# synth_struct/src/synth_struct/plotting/plot_utils.py

import numpy as np

"""
Utility functions for preparing microstructure data for visualization.
"""

def shuffle_display_grain_ids(grain_ids, num_grains, seed=None):
    """
    Get shuffled grain IDs for visualization without modifying the actual IDs.
    
    This creates a random color mapping for better visualizaiton while 
    preserving the actual grain ID structure for analysis.
    
    Args:
    - grain_ids: np.ndarray - Original grain ID array from microstructure
    - num_grains: int - Number of grains (excluding background)
    - seed: int or None - Random seed for reproducibility
    
    Returns:
    - np.ndarray - Grain IDs remapped for display (same shape as input)
    
    Example:
        micro = Microstructure(dimensions=(100,100), resolution=1.0)
        gen.generate(micro)
        display_ids = shuffle_display_grain_ids(micro.grain_ids, micro.num_grains, seed=42)
        plt.imshow(display_ids, cmap='nipy_spectral') 
    """
    
    if seed:
        np.random.seed(seed)
    
    max_grain_id = int(np.max(grain_ids))
    map_size = max(num_grains + 1, max_grain_id + 1)
    # Create random permutation
    old_ids = np.arange(1, num_grains + 1)
    new_ids = np.random.permutation(old_ids) + 1
    
    # Create display mapping
    id_map = np.zeros(map_size, dtype=np.int32)
    id_map[0] = 0 # Background stays 0
    id_map[old_ids] = new_ids
    
    if max_grain_id > num_grains:
        extra_ids = np.arange(num_grains + 1, max_grain_id + 1)
        id_map[extra_ids] = extra_ids
    
    # Return mapped grain_ids
    return id_map[grain_ids]
    
def get_colony_colormap(grain_ids, colony_ids, num_colonies):
    """
    Get a colormap where grains in the same colony have similar colors.
    
    Useful for visualizing colony structure in lath or other hierarchical
    microstructures where grains are grouped into colonies with similar
    crystallographic orientations.
    
    Args:
        grain_ids: np.ndarray - Grain ID array from microstructure
        colony_ids: np.ndarray - Array mapping grain index to colony ID
        num_colonies: int - Total number of colonies
        
    Returns:
        np.ndarray - Color values for visualization (same shape as grain_ids)
        
    Example:
        >>> gen = LathGenerator(num_grains=300, num_colonies=20)
        >>> gen.generate(micro)
        >>> colony_info = gen.get_colony_info()
        >>> colors = get_colony_colormap(micro.grain_ids, 
        ...                              colony_info['colony_ids'],
        ...                              colony_info['num_colonies'])
        >>> plt.imshow(colors, cmap='hsv')
    """
    num_grains = len(colony_ids)
    
    # Create a colormap where each colony gets a base hue
    grain_colors = np.zeros(num_grains + 1)
    grain_colors[0] = 0  # Background
    
    for grain_idx in range(num_grains):
        grain_id = grain_idx + 1
        colony_id = colony_ids[grain_idx]
        
        # Base color from colony, slight variation per grain
        base_color = colony_id * (256.0 / num_colonies)
        variation = (grain_id % 10) * 2  # Small variation within colony
        grain_colors[grain_id] = base_color + variation
    
    # Map to grain_ids array
    return grain_colors[grain_ids.astype(int)]


def create_grain_boundary_overlay(grain_ids):
    """
    Create a boolean array marking grain boundaries.
    
    Useful for overlaying grain boundaries on other visualizations.
    
    Args:
        grain_ids: np.ndarray - Grain ID array (2D or 3D)
        
    Returns:
        np.ndarray - Boolean array, True at grain boundaries
        
    Example:
        >>> boundaries = create_grain_boundary_overlay(micro.grain_ids[:, :, 50])
        >>> plt.imshow(micro.grain_ids[:, :, 50], cmap='nipy_spectral')
        >>> plt.contour(boundaries, colors='black', linewidths=0.5)
    """
    if grain_ids.ndim == 2:
        # 2D case
        boundaries = np.zeros_like(grain_ids, dtype=bool)
        boundaries[:-1, :] |= grain_ids[:-1, :] != grain_ids[1:, :]
        boundaries[:, :-1] |= grain_ids[:, :-1] != grain_ids[:, 1:]
        return boundaries
    
    elif grain_ids.ndim == 3:
        # 3D case - boundaries in any direction
        boundaries = np.zeros_like(grain_ids, dtype=bool)
        boundaries[:-1, :, :] |= grain_ids[:-1, :, :] != grain_ids[1:, :, :]
        boundaries[:, :-1, :] |= grain_ids[:, :-1, :] != grain_ids[:, 1:, :]
        boundaries[:, :, :-1] |= grain_ids[:, :, :-1] != grain_ids[:, :, 1:]
        return boundaries
    
    else:
        raise ValueError(f"grain_ids must be 2D or 3D, got {grain_ids.ndim}D")


def get_grain_size_colormap(grain_ids, num_grains):
    """
    Color grains by their size (number of voxels).
    
    Args:
        grain_ids: np.ndarray - Grain ID array
        num_grains: int - Number of grains
        
    Returns:
        np.ndarray - Size values for each voxel (same shape as grain_ids)
        
    Example:
        >>> sizes = get_grain_size_colormap(micro.grain_ids, micro.num_grains)
        >>> plt.imshow(sizes, cmap='viridis')
        >>> plt.colorbar(label='Grain Size (voxels)')
    """
    # Count voxels per grain
    grain_sizes = np.zeros(num_grains + 1)
    unique_ids, counts = np.unique(grain_ids, return_counts=True)
    grain_sizes[unique_ids] = counts
    
    # Map to grain_ids array
    return grain_sizes[grain_ids.astype(int)]
