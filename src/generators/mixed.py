# synth_struct/src/generators/mixed.py

from .gen_base import MicrostructureGenerator
from .gen_utils import get_seed_coordinates, aniso_voronoi_assignment
from ..orientation import (
    euler_to_rotation_matrix,
    create_rotation_matrix_2d
)
import numpy as np

class MixedGenerator(MicrostructureGenerator):
    """
    Weighted Voronoi tessellation generation with mixed grain morphologies.
    
    Generates a mixture of ellipsiodal (elongated) and equiaxed (spherical) grains.
    """
    
    def __init__(self, num_grains, ellipsoid_fraction=0.5, aspect_ratio_mean=5.0,
                 aspect_ratio_std=0.5, base_size=10.0, seed=None, chunk_size=500_000):
        """
        Initialize mixed generator.
        
        Args:
        - num_grains: int - Number of grains
        - ellipsoid_fraction: float - Fraction of ellipsoidal grains (0.0-1.0)
        - aspect_ratio_mean: float - Mean aspect ratio for ellipsoidal grains
        - aspect_ratio_std: float - Standard deviation of aspect ratio
        - base_size: float - Base size for all grains
        - seed: int or None - Random seed for reproducibility
        - chunk_size: int - Number of voxels to process per chunk for memory efficiency
        """
        
        self.num_grains = num_grains
        self.ellipsoid_fraction = np.clip(ellipsoid_fraction, 0.0, 1.0)
        self.aspect_ratio_mean = aspect_ratio_mean
        self.aspect_ratio_std = aspect_ratio_std
        self.base_size = base_size
        self.seed = seed
        self.chunk_size = chunk_size
        
        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None
        
    def _generate_internal(self, micro):
        """
        Generate mixed grain morphologies.
        
        Args:
        - micro: Microstructure - Instance with 'dimensions' and 'grain_ids'
        """
        
        if self.seed:
            np.random.seed(self.seed)
            
        ndim = len(micro.dimensions)
        micro.num_grains = self.num_grains
        
        # Generate random seed points
        self.seeds = get_seed_coordinates(self.num_grains, micro.dimensions, self.seed)
        
        # Generate mixed grain parameters
        self.scale_factors, self.rotations = self._generate_mixed_params(
            self.num_grains, ndim
        )
        
        # Perform weighted Voronoi tessellation
        aniso_voronoi_assignment(micro, self.seeds, self.scale_factors, 
                                 self.rotations, self.chunk_size)
                                 
        num_ellipsoidal = int(self.num_grains * self.ellipsoid_fraction)
        num_equiaxed = self.num_grains - num_ellipsoidal
        
        print(f"Generated {self.num_grains} grains: "
              f"{num_ellipsoidal} ellipsoidal, {num_equiaxed} equiaxed")
              
        
    def _generate_mixed_params(self, num_grains, ndim):
        """
        Generate scale factors and rotation matrices for mixed grain morphologies.
        
        Args:
        - num_grains: int - Number of grains
        - ndim: int - Number of dimensions (2 or 3)
        
        Returns:
        - scale_factors: np.ndarray of shape (num_grains, ndim)
        - rotations: list of rotation matrices (each is ndim x ndim)
        """
        
        num_ellipsoidal = int(num_grains * self.ellipsoid_fraction)
        
        scale_factors = np.zeros((num_grains, ndim))
        rotations = []
        
        for i in range(num_grains):
            if i < num_ellipsoidal: # Ellipsoidal grains
                aspect_ratio = np.random.normal(self.aspect_ratio_mean, self.aspect_ratio_std)
                aspect_ration = np.clip(aspect_ratio, 1.5, 15.0)
                
                if ndim == 3:
                    scale_factors[i] = [self.base_size, self.base_size, self.base_size * aspect_ratio]
                    angles = np.random.uniform(0, 2*np.pi, 3)
                    R = euler_to_rotation_matrix(angles)
                else: # 2D
                    scale_factors[i] = [self.base_size, self.base_size * aspect_ratio]
                    angle = np.random.uniform(0, 2*np.pi, 3)
                    R = create_rotation_matrix_2d(angle)
                    
            else: # Equiaxed grains
                if ndim == 3:
                    scale_factors[i] = [self.base_size, self.base_size, self.base_size]
                else: # 2D
                    scale_factors[i] = [self.base_size, self.base_size]
                R = np.eye(ndim)
                
            rotations.append(R)
            
        return scale_factors, rotations
    
        
