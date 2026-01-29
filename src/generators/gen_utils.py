# synth_struct/src/generators/gen_utils.py

"""
Useful utility functions for microstructure generators
"""

import time
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
    
    
def aniso_voronoi_assignment(micro, seeds, scale_factors, rotations, chunk_size=500_000):
    """
    Assign voxels using anisotropic distance metric.
    
    For each point p and seed s with scaling S and rotation R:
    d_aniso = ||S^-1 * R^T * (p - s)||^2
    
    Args:
        micro: Microstructure - Instance to populate grain_ids
        seeds: np.ndarray of shape (num_grains, ndim) - Seed coordinates
        scale_factors: np.ndarray of shape (num_grains, ndim) - Scaling per axis
        rotations: list of rotation matrices - One per grain
        chunk_size: int - Number of voxels to process per chunk
    """
    ndim = len(micro.dimensions)
    total_voxels = int(np.prod(micro.dimensions))
    grain_ids_flat = np.zeros(total_voxels, dtype=np.int32)
    
    print(f"Performing anisotropic Voronoi tessellation...")
    start_time = time.time()
    
    for start in range(0, total_voxels, chunk_size):
        end = min(start + chunk_size, total_voxels)
        
        chunk_indices = np.arange(start, end)
        chunk_coords = np.column_stack(np.unravel_index(chunk_indices, micro.dimensions))
        chunk_coords = chunk_coords.astype(float)
        
        distances = np.zeros((len(chunk_coords), len(seeds)))
        
        for i, (seed, scale, R) in enumerate(zip(seeds, scale_factors, rotations)):
            diff = chunk_coords - seed
            diff_rotated = diff @ R.T  # R^T * (p-s)
            diff_scaled = diff_rotated / scale[np.newaxis, :]
            
            distances[:, i] = np.sum(diff_scaled**2, axis=1)
            
        grain_assignment = np.argmin(distances, axis=1) + 1
        grain_ids_flat[start:end] = grain_assignment
        
        if (start // chunk_size) % 10 == 0:
            progress = 100 * end / total_voxels
            print(f"  Progress: {progress:.1f}%")
            
    micro.grain_ids = grain_ids_flat.reshape(micro.dimensions)
    print("Done!")
    
    
