import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class Microstructure:
    def __init__(self, dimensions, resolution):
        """
        dimensions: tuple (nx, ny) or (nx, ny, nz) for 2D or 3D
        resolution: physical size per voxel
        """
        
        self.dimensions = dimensions
        self.resolution = resolution
        self.grain_ids = np.zeros(dimensions, dtype=np.int32)
        self.orientations = {} # grain_id: orientation (Euler angles)
        
    def get_num_grains(self):
        return len(np.unique(self.grain_ids)) - 1 # exclude background (0)
    
    # Old voronoi generator    
    def generate_voronoi_grains(self, num_grains, seed=None):
        """
        Generate grains using Voronoi tessellation
        """
        if seed:
            np.random.seed(seed)
        
        # 2D Case    
        if len(self.dimensions) == 2:
            # Generate random seed points
            seeds = np.random.rand(num_grains, 2) * self.dimensions
            
            # Create grid of all voxel centers
            y, x = np.mgrid[0:self.dimensions[0], 0:self.dimensions[1]]
            points = np.column_stack([x.ravel(), y.ravel()])
            
            # Assign each point to nearest seed
            distances = cdist(points, seeds)
            grain_assignment = np.argmin(distances, axis=1) + 1 # +1 so grains start at 1
            
            self.grain_ids = grain_assignment.reshape(self.dimensions)
        
        # For 3D Case
        if len(self.dimensions) == 3:
            # Generate random seed points
            seeds = np.random.rand(num_grains, 3) * self.dimensions
            
            # Create grid of all voxel centers
            y, x, z = np.mgrid[0:self.dimensions[0], 0:self.dimensions[1], 0:self.dimensions[2]]
            points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            # Assign each point to nearest seed
            distances = cdist(points, seeds)
            grain_assignment = np.argmin(distances, axis=1) + 1 # +1 so grains start at 1
            
            self.grain_ids = grain_assignment.reshape(self.dimensions)
            
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
                
        
        
