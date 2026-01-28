import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class VoronoiGenerator:
    """
    Voronoi tessellation generator for a Microstructure object.
    
    Generates grains by assigning the nearest seed point to each voxel.
    """
    
    
    def __init__(self, num_grains, seed=None, chunk_size=1_000_000):
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
        self.seeds = None # Will store coordinates for Voronoi seeds

    def generate(self, micro):
        """
        Generate grains with a standard Voronoi tesselation
        
        Args:
        - micro: Microstructure
            Instance of a Microstructure class with 'dimensions' and 'grain_ids'
        """
        if self.seed:
            np.random.seed(self.seed)
        
        # Number of dimensions
        ndim = len(micro.dimensions)
        micro.num_grains = self.num_grains
        
        # Generate random seed points
        self.seeds = np.random.rand(self.num_grains, ndim) * np.array(micro.dimensions)
        tree = cKDTree(self.seeds)
        
        # Total number of voxels
        total_voxels = int(np.prod(micro.dimensions))
        grain_ids_flat = np.empty(total_voxels, dtype=np.int32)
        
        # Process in chunks
        for start in range(0, total_voxels, self.chunk_size):
            end = min(start+self.chunk_size, total_voxels)
            
            # Convert flat indices to coordinates
            flat_indices = np.arange(start, end)
            coords = np.column_stack(np.unravel_index(flat_indices, micro.dimensions))
            _, indices = tree.query(coords)
            grain_ids_flat[start:end] = indices + 1
            
        # Reshape to Microstructure voxel grid
        micro.grain_ids = grain_ids_flat.reshape(micro.dimensions)
        
        # Allocate default per-grain arrays if needed
        if not hasattr(micro, 'orientations') or micro.orientations is None:
            self._allocate_grain_arrays(micro)
        
    
    def _allocate_grain_arrays(self, micro):
        """
        Allocate per-grain arrays for orientatations, stiffnes, and phase.
        """
        
        n = micro.num_grains + 1 # index 0 reserved for background
        
        micro.orientations = np.zeros((n, 3), dtype=np.float64)
        micro.stiffness = np.zeros((n, 6, 6), dtype=np.float32)
        micro.phase = np.zeros(n, dtype=np.int8)
        
    def get_seed_coordinates(self):
        """
        Return the coordinates of the Voronoi seed points
        """
        
        return self.seeds



