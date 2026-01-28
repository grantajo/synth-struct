# synth_struct/src/generators/utils.py

"""
Useful utility functions for microstructure generators
"""

import numpy as np

def get_seed_coordinates(num_grains, dimensions, seed=None):
    """
    Generate uniformly random seed coordinates for grains in a microstructure.
    
    Args:
    - num_grains (int): Number of grains to generate
    - dimensions (tuple): (nx, ny) or (nx, ny, nz) for 2D/3D microstructures.
    - seed (int, optional): Random seed for reproducibility

    Returns:
    - np.ndarray: Array of shape (num_grains, ndim) with coordinates
    """
    
    if seed:
        np.random.seed(seed)
        
    ndim = len(dimensions)
    seeds = np.random.rand(num_grains, ndim) * np.array(dimensions)
    
    return seeds
