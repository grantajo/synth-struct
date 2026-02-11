# synth_struct/src/synth_struct/generators/voronoi.py

"""
This class holds the VoronoiGenerator class that generates an
isotropic Voronoi microstructure.
"""

import numpy as np
from scipy.spatial import cKDTree

from .gen_base import MicrostructureGenerator
from .gen_utils import get_seed_coordinates


class VoronoiGenerator(MicrostructureGenerator):
    """
    Voronoi tessellation generator for a Microstructure object.

    Generates grains by assigning the nearest seed point to each voxel.
    """

    def __init__(self, num_grains, seed=None, chunk_size=500_000):
        """
        Initiates generator information

        Args:
        - num_grains: int
            Number of grains
        - seed: int or None
            Random seed for reproducibility
        - chunk_size: int
            Number of voxels to process per chunk for memory efficiency
        """

        self.num_grains = num_grains
        self.seed = seed
        self.chunk_size = chunk_size
        self.seeds = None  # Will store coordinates for Voronoi seeds

    def _generate_internal(self, micro):
        """
        Generate grains with a standard Voronoi tesselation

        Args:
        - micro: Microstructure
            Instance of a Microstructure class with 'dimensions' and 'grain_ids'
        """
        if self.seed:
            np.random.seed(self.seed)

        # Generate random seed points
        self.seeds = get_seed_coordinates(self.num_grains, micro.dimensions, self.seed)
        tree = cKDTree(self.seeds)

        # Total number of voxels
        total_voxels = int(np.prod(micro.dimensions))
        grain_ids_flat = np.empty(total_voxels, dtype=np.int32)

        # Process in chunks
        for start in range(0, total_voxels, self.chunk_size):
            end = min(start + self.chunk_size, total_voxels)

            # Convert flat indices to coordinates
            flat_indices = np.arange(start, end)
            coords = np.column_stack(np.unravel_index(flat_indices, micro.dimensions))
            _, indices = tree.query(coords)
            grain_ids_flat[start:end] = indices + 1

        # Reshape to Microstructure voxel grid
        micro.grain_ids = grain_ids_flat.reshape(micro.dimensions)
