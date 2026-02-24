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
from ..phase import Phase

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
    - phase: Phase object
    - degspread: float - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(self, texture_type=None, phase=None, degspread=5.0, seed=None):
        if texture_type not in HEXAGONAL_ORIENTATIONS:
            raise ValueError(f"Unknown hexagonal texture type {texture_type}")
        if degspread is not None and degspread < 0:
            raise ValueError("scale < 0")
        if phase is None:
            phase = Phase(
                name="hexagonal_default",
                crystal_system="hexagonal",
                lattice_params=(1, 1, 1.633),
            )
        if phase.crystal_system != "hexagonal":
            raise ValueError(
                f"HexagonalTexture requires a hexagonal phase, "
                f"got '{phase.crystal_system}'"
            )

        self.texture_type = texture_type
        self.phase = phase
        self.degspread = degspread
        self.seed = seed

    def generate(self, micro):
        """Generate a Texture for the given microstructure."""
        orientations = self._generate_orientations(micro)

        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=self.phase,
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
