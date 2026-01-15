import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class Microstructure:
    def __init__(self, dimensions, resolution, units='micron'):
        """
        dimensions: tuple (nx, ny) or (nx, ny, nz) for 2D or 3D
        resolution: physical size per voxel
        """
        
        self.dimensions = dimensions
        self.resolution = resolution
        self.units = units
        self.grain_ids = np.zeros(dimensions, dtype=np.int32)
        self.orientations = {} # grain_id: orientation (Euler angles)
        
    def get_num_grains(self):
        return len(np.unique(self.grain_ids)) - 1 # exclude background (0)
            
    def gen_voronoi(self, num_grains, seed=None, chunk_size=1_000_000):
        """
        More memory-efficient Voronoi using KDTree
        """
        if seed:
            np.random.seed(seed)
        
        # Number of dimensions
        ndim = len(self.dimensions)
        
        # Generate random seed points
        seeds = np.random.rand(num_grains, ndim) * np.array(self.dimensions)
        tree = cKDTree(seeds)
        
        # Total number of voxels
        total_voxels = np.prod(self.dimensions)
        grain_ids_flat = np.zeros(total_voxels, dtype=np.int32)
        
        # Process in chunks
        chunk_size = 1_000_000
        for start in range(0, total_voxels, chunk_size):
            end = min(start+chunk_size, total_voxels)
            
            # Convert flat indices to coordinates
            chunk_indices = np.arange(start, end)
            chunk_coords = np.unravel_index(chunk_indices, self.dimensions)
            chunk_coords = np.column_stack(chunk_coords)
            
            # Find nearest seed for each coordinate
            distances, indices = tree.query(chunk_coords)
            grain_ids_flat[start:end] = indices + 1
            
        # Reshape back to original dimensions
        self.grain_ids = grain_ids_flat.reshape(self.dimensions)
                
        
        
