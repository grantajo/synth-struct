# synth_struct/src/orientation/texture/hexagonal.py

import numpy as np
from .texture_base import TextureGenerator
from ..rotation_converter import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat
)

HEXAGONAL_ORIENTATIONS = {
    'basal': np.array([0.0, 0.0, 0.0]),
    'prismatic': np.array([0.0, np.radians(90.0), 0.0])
}

class HexagonalTexture(TextureGenerator):
    """
    Generates a hexagonal texture for a given microstructure.
    
    Args:
    - type: str
        One of 'basal' or 'prismatic'.
    - degspread: float - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """
    
    def __init__(self, type=None, degspread=5.0, seed=None):
        if type not in HEXAGONAL_TEXTURES:
            raise ValueError(f"Unkown hexagonal texture type {type}")
        self.type = type
        self.degspread = degspread
        self.seed = seed
        
    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        n = micro.num_grains
        base_orienation = HEXAGONAL_TEXTURES[self.type]
        
        if self.spread == 0:
            orientations = np.tile(base_orientation, (n, 1))
        else:
            orientations = np.random.normal(
                loc=base_orientation,
                scale=np.radians(self.degspread),
                size=(n,3)
            )
        
        orientations = orientations % (2*np.pi)
        
        return orientations
