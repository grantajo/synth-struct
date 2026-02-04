# synth_struct/src/synth_struct/generators/lath.py

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
                 colony_misorientation=5.0, basketweave_fraction=0.7, 
                 bw_variants=24, seed=None, chunk_size=500_000):
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
        - basketweave_fraction: float - Fraction of microstructure that has basketweave (0.0)
                                        0.0 = pure colony, 1.0 = pure basketweave
        - bw_variants: int - Number of orthogonal variants for basketweave (2, 3, 4, 6, 12, or 24)
                             24 or None = nearly realistic K-S or N-W transformation
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
        self.basketweave_fraction = np.clip(basketweave_fraction, 0.0, 1.0)
        self.bw_variants = np.clip(bw_variants, 2, 8)
        self.seed = seed
        self.chunk_size = chunk_size
        
        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None
        self.colony_ids = None # Which colony each lath belongs to
        self.colony_centers = None
        self.is_basketweave = None # Boolean array: which grains use basketweave
        
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
        
        # Determine which grains are basketweave vs colony
        self.is_basketweave = np.random.rand(self.num_grains) < self.basketweave_fraction
        
        # Generate colony centers and assign laths to colonies
        self._generate_colony_structure(micro.dimensions)
        
        # Generate lath parameters
        self.scale_factors, self.rotations = self._generate_lath_params(ndim)
        
        # Perform weighted Voronoi tessellation
        aniso_voronoi_assignment(micro, self.seeds, self.scale_factors,
                                 self.rotations, self.chunk_size)
        
        num_basketweave = np.sum(self.is_basketweave)
        num_colony = self.num_grains - num_basketweave
        
        print(f"Generated {self.num_grains} laths in {self.num_colonies} colonies "
              f"({num_basketweave} basketweave, {num_colony} colony)")
              
        
    def _generate_colony_structure(self, dimensions):
        """
        Generate colony centers and assign laths to colonies spatially.
        
        Args:
        - dimensions: tuple - Microstructure dimensions
        """
        # Generate colony centers
        self.colony_centers = np.random.rand(self.num_colonies, 3) * np.array(dimensions)
        
        # Generate seeds with spatial clustering
        self.seeds = np.zeros((self.num_grains, 3))
        self.colony_ids = np.zeros(self.num_grains, dtype=np.int32)
        
        # Calculate how many laths per colony
        laths_per_colony = self.num_grains // self.num_colonies
        extra_laths = self.num_grains % self.num_colonies
        
        # Cluster size relative to domain size
        cluster_radius = np.mean(dimensions) / (2 * self.num_colonies**0.5)
        
        grain_idx = 0
        for colony_id in range(self.num_colonies):
            # Number of laths in this colony
            n_laths = laths_per_colony + (1 if colony_id < extra_laths else 0)
            
            center = self.colony_centers[colony_id]
            
            for _ in range(n_laths):
                # Clustering strength depends on basketweave fraction
                # Pure colony: tight clustering
                # Pure basketweave: looser clustering
                clustering_factor = 1.0 - 0.7*self.basketweave_fraction
                
                # Gaussian distribution around colony center
                offset = np.random.randn(3) * cluster_radius * clustering_factor
                seed = center + offset
                
                # Clip to domain boundaries
                seed = np.clip(seed, [0, 0, 0], dimensions)
                
                self.seeds[grain_idx] = seed
                self.colony_ids[grain_idx] = colony_id
                grain_idx += 1
        
        
    def _generate_lath_params(self, ndim):
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
        
        variant_base_angles = self._get_variant_orientations()
        colony_to_variant = np.arange(self.num_colonies) % self.bw_variants
        
        scale_factors = np.zeros((self.num_grains, 3))
        rotations = []
        
        for i in range(self.num_grains):
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
            
            if self.is_basketweave[i]:
                # Use variant orientation
                variant_id = colony_to_variant[colony_id]
                base_angles = variant_base_angles[variant_id]
            else:
                # Use colony orientation
                base_angles = colony_orientations[colony_id]
            
            # Add small misorientation within packet
            misori = np.random.uniform(-self.colony_misorientation, self.colony_misorientation, 3)
            angles = base_angles + misori
            
            R = euler_to_rotation_matrix(angles)
            rotations.append(R)
            
        return scale_factors, rotations
        
    def _get_variant_orientations(self):
        """
        Get base orientations for basketweave variants.
        
        Returns:
        - np.ndarray of shape (bw_variants, 3) - Euler angles for each variant
        """
        if self.bw_variants == 2:
            # Two variants
            return np.array([
                [0, np.pi/4, 0],
                [np.pi/2, np.pi/4, 0]
            ])
        
        if self.bw_variants == 3:
            # Three variants 120 degrees apart
            return np.array([
                [0, np.pi/4, 0],
                [2*np.pi/3, np.pi/4, np.pi/6],
                [4*np.pi/3, np.pi/4, np.pi/3]
            ])
            
        if self.bw_variants == 4:
            # Four variants (tetrahedral distribution)
            return np.array([
                [0, np.pi/4, 0],
                [np.pi/2, np.pi/4, np.pi/6],
                [np.pi, np.pi/4, np.pi/3],
                [3*np.pi/2, np.pi/4, np.pi/2]
            ])
            
        if self.bw_variants == 6:
            # Six variants
            variants = []
            phi2_vals = [0, np.pi/6, np.pi/3]
            for i, phi1 in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
                phi2 = phi2_vals[i % 3]
                variants.append([phi1, np.pi/4, phi2])
            return np.array(variants)
            
        if self.bw_variants == 12:
            variants = []
            for theta in [np.pi/6, np.pi/3]:
                for phi1 in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    phi2 = np.random.uniform(0, 2*np.pi) # Randomize phi2
                    variants.append([phi1, theta, phi2])
            return np.array(variants)
        
        else:
            return self._ks_variants()
            
    def _ks_variants(self):
        """
        Generate 24 variants based on the Kurdjumov-Sachs orientation relationship.
        
        K-S: austenite  ||  martensite
               {111}    ||    {011}
               <101>    ||    <111>
        
        This creates 24 crystallographically equivalent variants.
        """
        
        variants = []
        sq2 = np.sqrt(2)
        
        # Four {111} planes
        habit_planes = [
            [np.arctan(sq2), 0, 0],
            [np.arctan(sq2), 0, 2*np.pi/3],
            [np.arctan(sq2), 0, 4*np.pi/3],
            [np.pi - np.arctan(sq2), 0, 0]
        ]
        
        # Six <111> directions per plane
        for base_phi1, Phi, base_phi2 in habit_planes:
            for i in range(6):
                phi1 = base_phi1 + i * np.pi/3
                phi2 = base_phi2 + i * np.pi/6
                variants.append([phi1 % (2*np.pi), Phi, phi2 % (2*np.pi)])
                
        return np.array(variants)
    
    def get_colony_info(self):
        """
        Get information about colony assignments.
        
        Returns:
        dict with keys:
        - 'colony_ids': array mapping grain_id to colony_id
        - 'num_colonies': total number of colonies
        - 'grains_per_colony': dict mapping colony_id to list of grain_ids
        - 'colony_centers': array of colony center coordinates
        - 'basketweave_fraction': fraction of basketweave grains
        - 'is_basketweave': boolean array indicating basketweave grains
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
            'grains_per_colony': grains_per_colony,
            'colony_centers': self.colony_centers,
            'basketewave_fraction': self.basketweave_fraction,
            'is_basketweave': self.is_basketweave
        }







