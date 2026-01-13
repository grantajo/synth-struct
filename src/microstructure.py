import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

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
        
        # For 2D Case
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
        
