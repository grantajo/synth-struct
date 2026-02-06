# synth_struct/src/synth_struct/generators/gen_utils.py

"""
Useful utility functions for microstructure generators
"""

import numpy as np
from .._cpp_extensions import aniso_voronoi_assignment as cpp_aniso_voronoi
from .._cpp_extensions import EIGEN_AVAILABLE


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


def aniso_voronoi_assignment(
    micro, seeds, scale_factors, rotations, chunk_size=500_000, use_cpp=True
):
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
        use_cpp: bool - Use C++ implementation if available
    """
    if use_cpp and EIGEN_AVAILABLE and cpp_aniso_voronoi is not None:
        # Use C++ Eigen implementation
        dimensions = np.array(micro.dimensions, dtype=np.int32)
        seeds_float = np.ascontiguousarray(seeds.astype(np.float32))
        scales_float = np.ascontiguousarray(scale_factors.astype(np.float32))
        rotations_float = [np.ascontiguousarray(R.astype(np.float32)) for R in rotations]
        
        grain_ids_flat = cpp_aniso_voronoi(
            dimensions, seeds_float, scales_float, rotations_float, chunk_size
        )
        
        micro.grain_ids = grain_ids_flat.reshape(micro.dimensions)
    
    else:
        total_voxels = int(np.prod(micro.dimensions))
        grain_ids_flat = np.zeros(total_voxels, dtype=np.int32)

        print("  Performing anisotropic Voronoi tessellation (Python)...")

        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)

            chunk_indices = np.arange(start, end)
            chunk_coords = np.column_stack(
                np.unravel_index(chunk_indices, micro.dimensions)
            )
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
