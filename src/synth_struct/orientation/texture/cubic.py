# synth_struct/src/orientation/texture/cubic.py

"""
This holds the CubicTexture Generator.

There are several basic textures for cubic:
- cube
- goss
- brass
- copper
- s
- p
- rotated_cube
- rotated_goss
"""

import numpy as np
from .texture_base import TextureGenerator

CUBIC_TEXTURES = {
    "cube": np.array([0.0, 0.0, 0.0]),
    "goss": np.array([0.0, np.radians(45.0), np.radians(45.0)]),
    "brass": np.array([np.radians(35.26), np.radians(45.0), 0.0]),
    "copper": np.array([np.radians(90.0), np.radians(35.26), np.radians(45.0)]),
    "s": np.array([np.radians(58.98), np.radians(36.70), np.radians(63.43)]),
    "p": np.array([np.radians(70.53), np.radians(45.0), 0.0]),
    "rotated_cube": np.array([np.radians(45.0), 0.0, 0.0]),
    "rotated_goss": np.array([0.0, np.radians(45.0), np.radians(45.0)]),
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
        if type not in CUBIC_TEXTURES:
            raise ValueError(f"Unknown cubic texture type {type}")
        self.type = type
        self.degspread = degspread
        self.seed = seed

    def generate(self, micro):
        """Generate a Texture for the given microstructure."""
        if self.seed:
            np.random.seed(self.seed)

        from .texture import Texture

        orientations = self._generate_orientations(micro)

        return Texture(
            orientations=orientations, representation="euler", symmetry="cubic"
        )

    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        if isinstance(micro, np.ndarray):
            n = len(micro)
        else:
            n = micro.num_grains
        
        base_orientation = CUBIC_TEXTURES[self.type]
        
        orientations = np.zeros((n + 1, 3))

        if self.degspread == 0:
            orientations[1:] = np.tile(base_orientation, (n, 1))
        else:
            orientations[1:] = np.random.normal(
                loc=base_orientation, scale=np.radians(self.degspread), size=(n, 3)
            )

        orientations[1:] = orientations[1:] % (2 * np.pi)

        return orientations
        
        
