# synth_struct/src/orientation/texture/cubic.py

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

CUBIC_ORIENTATIONS = {
    'cube': np.array([0., 0., 0.]),
    'goss': np.array([0.0, np.radians(45.0), np.radians(45.0)]),
    'brass': np.array([np.radians(35.26), np.radians(45.0), 0.0]),
    'copper': np.array([np.radians(90.0), np.radians(35.26), np.radians(45.0)]),
    's': np.array([np.radians(58.98), np.radians(36.70), np.radians(63.43)]),
    'p': np.array([np.radians(70.53), np.radians(45.0), 0.0]),
    'rotated_cube': np.array([np.radians(45.0), 0.0, 0.0]),
    'rotated_goss': np.array([0.0, np.radians(45.0), np.radians(45.0)])
}

class CubicTexture(TextureGenerator):
    """
    Generates a cubic texture for a given microstructure.
    
    Args:
    - type: str
        One of 'cube', 'goss', 'brass', 'copper', 's', 'p', 'rotated_cube', 'rotated_goss'.
    - degspread: float - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """
    
    def __init__(self, type=None, degspread=5.0, seed=None):
        if type not in CUBIC_ORIENTATIONS:
            raise ValueError(f"Unkown cubic texture type {type}")
        self.type = type
        self.degspread = degspread
        self.seed = seed
        
    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        n = microstructure.num_grains
        base_orienation = CUBIC_ORIENTATIONS[self.type]
        
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
        
