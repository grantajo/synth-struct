# synth_struct/src/generators/lath.py

from .gen_base import MicrostructureGenerator
from .gen_utils import get_seed_coordinates, aniso_voronoi_assignment
from ..orientation import (
    euler_to_rotation_matrix,
    create_rotation_matrix_2d
)
import numpy as np

"""
Possible improvements:
- Second phases between laths
- Spatial clustering of colonies
"""

class LathGenerator(MicrostructureGenerator):
    """
    Weighted Voronoi tessellation generation for lath microstructures.
    
    Generates thin, elongated lath-liek grains typical of martensitic, bainitic, or titanium
    microstructures. Laths are organized into colonies with similar orientations.
    """
    
    def __init__(self, num_grains, num_colonies=None, aspect_ratio_mean=10.0,
                 aspect_ratio_std=2.0, width_mean=2.0, width_std=0.5,
                 thickness_mean=0.5, thickness_std=0.5, colony_misorientation=15.0, 
                 seed=None, chunk_size=500_000):
        """
        Initialize lath generator.
        
        Args:
        - num_grains: int - Number of laths
        - num_colonies: int or None - Number of colonies (groups of laths with similar orientation).
                                      If None, defaults to num_grains // 8
        - aspect_ratio_mean: float - Mean aspect ratio (length/width)
        - aspect_ratio_std: float - Standard deviation of aspect ratio
        - width_mean: float - Mean lath width
        - width_std: float - Standard deviation of lath width
        - colony_misorientation: float - Max misorientation (degrees) within a colony
        - seed: int or None - Random seed for reproducibility
        - chunk_size: int - Number of voxels to process per chunk for memory efficiency
        """ 
        
        self.num_grains = num_grains
        self.num_colonies = num_colonies if num_colonies is not None else max(1, num_grains // 8)
        self.aspect_ratio_mean = aspect_ratio_mean
        self.aspect_ratio_std = aspect_ratio_std
        self.width_mean = width_mean
        self.width_std = width_std
        self.colony_misorientation = np.radians(colony_misorientation)
        self.seed = seed
        self.chunk_size = chunk_size
        
        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None
        self.colony_ids = None # Which colony each lath belongs to
        
    def _generate_internal(self, micro):
        """
        Generate lath microstruture.
        
        Args:
        - micro: Microstructure - Instance with 'dimension' and 'grain_ids'
        """
        
        if self.seed:
            np.random.seed(self.seed)
        
        ndim = len(micro.dimensions)
        
        if ndim != 3:
            raise ValueError("Lath microstructures only supported for 3D")
            
        micro.num_grains = self.num_grains
        
        # Generate random seed points
        self.seeds = get_seed_coordinates(self.num_grains, micro.dimensions, self.seed)
        
        # Assign laths to colonies
        self.colony_ids = np.random.randint(0, self.num_colonies, self.num_grains)
        
        # Generate lath parameters
        self.scale_factors, self.rotations = self._generate_lath_params(
            self.num_grains, ndim
        )
        
        # Perform weighted Voronoi tessellation
        aniso_voronoi_assignment(micro, self.seeds, self.scale_factors,
                                 self.rotations, self.chunk_size)
        
        print(f"Generated {self.num_grains} laths in {self.num_colonies} colonies")
        
    def _generate_lath_params(self, num_grains, ndim):
        """
        Generate scale factors and rotation matrices for lath grains.
        
        Laths within the same colony have similar orientations.
        
        Args:
        - num_grains: int - Number of laths
        - ndim: int - Number fo dimensions (must be 3)
        
        Returns:
        - scale_factors: np.ndarray of shape (num_grains, 3)
        - rotations: list of rotation matrices (each is 3x3)
        """
        
        # Generate base orientation for each colony
        colony_orientations = np.random.uniform(0, 2*np.pi, (self.num_colonies, 3))
        
        scale_factors = np.zeros((num_grains, 3))
        rotations = []
        
        for i in range(num_grains):
            # Generate lath dimensions
            aspect_ratio = np.random.normal(self.aspect_ratio_mean, self.aspect_ratio_std)
            aspect_ratio = np.clip(aspect_ratio, 3.0, 40.0)
            
            width = np.random.normal(self.width_mean, self.width_std)
            width = np.clip(width, 0.5, 5.0)
            
            thickness = width * np.random.uniform(0.3, 0.7)
            length = width * aspect_ratio
            
            # Laths are elongated along z, thin along y
            scale_factors[i] = [width, thickness, length]
            
            # Get colony base orientation
            colony_id = self.colony_ids[i]
            base_angles = colony_orientations[colony_id]
            
            # Add small misorientation within packet
            misori = np.random.uniform(-self.colony_misorientation, self.colony_misorientation, 3)
            angles = base_angles + misori
            
            R = euler_to_rotation_matrix(angles)
            rotations.append(R)
            
        return scale_factors, rotations
        
    def get_colony_info(self):
        """
        Get information about colony assignments.
        
        Returns:
        dict with keys:
        - 'colony_ids': array mapping grain_id to colony_id
        - 'num_colonies': total number of colonies
        - 'grains_per_colony': dict mapping colony_id to list of grain_ids
        """
        
        if self.colony_ids is None:
            raise ValueError("Must call generate() before getting colony info")
            
        grains_per_colony = {}
        for grain_id in range(1, self.num_grains + 1):
            colony_id = self.colony_ids[grain_id - 1]
            if colony_id not in grains_per_colony:
                grains_per_colony[colony_id] = []
            grains_per_colony[colony_id].append(grain_id)
            
        return {
            'colony_ids': self.colony_ids,
            'num_colonies': self.num_colonies,
            'grains_per_colony': grains_per_colony
        }







