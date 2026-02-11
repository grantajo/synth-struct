# synth_struct/src/orientation/texture/hexagonal.py

"""
This holds the HexagonalTexture Generator.

There are two basic textures for hexagonal:
- basal
- prismatic
"""

import numpy as np
from .texture_base import TextureGenerator

HEXAGONAL_ORIENTATIONS = {
    "basal": np.array([0.0, 0.0, 0.0]),
    "prismatic": np.array([0.0, np.radians(90.0), 0.0]),
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
        if type not in HEXAGONAL_ORIENTATIONS:
            raise ValueError(f"Unknown hexagonal texture type {type}")
        self.type = type
        self.degspread = degspread
        self.seed = seed

    def generate(self, micro):
        """Generate a Texture for the given microstructure."""
        if self.seed is not None:
            np.random.seed(self.seed)

        from .texture import Texture

        orientations = self._generate_orientations(micro)

        return Texture(
            orientations=orientations, representation="euler", symmetry="hexagonal"
        )

    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        n = micro.num_grains
        base_orientation = HEXAGONAL_ORIENTATIONS[self.type]

        if self.degspread == 0:
            orientations = np.tile(base_orientation, (n, 1))
        else:
            orientations = np.random.normal(
                loc=base_orientation, scale=np.radians(self.degspread), size=(n, 3)
            )

        orientations = orientations % (2 * np.pi)

        return orientations
