# synth-struct/src/orientation/texture/hexagonal.py

"""
This holds the HexagonalTexture Generator.

There are two basic textures for hexagonal:
- basal
- prismatic
"""

import numpy as np

from .texture import Texture
from .texture_base import TextureGenerator

HEXAGONAL_ORIENTATIONS = {
    "basal": np.array([0.0, 0.0, 0.0]),
    "prismatic": np.array([0.0, np.radians(90.0), 0.0]),
    "pyramidal": np.array([0.0, np.radians(45.0), 0.0]),
}


class HexagonalTexture(TextureGenerator):
    """
    Generates a hexagonal texture for a given microstructure.

    Args:
    - texture_type: str
        One of 'basal', 'prismatic', or 'pyramidal' .
    - degspread: float - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(self, texture_type=None, degspread=5.0, seed=None):
        if texture_type not in HEXAGONAL_ORIENTATIONS:
            raise ValueError(f"Unknown hexagonal texture type {texture_type}")
        if degspread is not None and degspread < 0:
            raise ValueError("scale < 0")
        
        self.texture_type = texture_type
        self.degspread = degspread
        self.seed = seed

    def generate(self, micro):
        """Generate a Texture for the given microstructure."""
        orientations = self._generate_orientations(micro)
        
        texture = Texture(
            orientations=orientations, 
            representation="euler", 
            symmetry="hexagonal"
        )
        if self.degspread is not None and self.degspread > 0:
            texture = texture.apply_scatter(self.degspread, seed=self.seed)
            
        return texture

    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        if isinstance(micro, np.ndarray):
            n = len(micro)
            include_background = False
        else:
            n = micro.num_grains
            include_background = True

        base_orientation = HEXAGONAL_ORIENTATIONS[self.texture_type]

        if include_background:
            orientations = np.zeros((n + 1, 3))
            start_idx = 1
        else:
            orientations = np.zeros((n, 3))
            start_idx = 0

        orientations[start_idx:] = np.tile(base_orientation, (n, 1))

        return orientations
