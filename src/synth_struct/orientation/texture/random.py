# synth_struct/src/orientation/texture/random.py

"""
This holds the CustomTexture Generator.

This creates a random orientation for each grain in a Microstructure
"""

import numpy as np
from .texture_base import TextureGenerator


class RandomTexture(TextureGenerator):
    """
    Generate a random (uniform) texture.

    Orientations are sampled uniformly in SO(3) and reduced
    according to crystal symmetry.

    Args:
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(self, seed=None):
        self.seed = seed

    def generate(self, micro):
        """
        Generate random orientations for all grains in the microstructure

        Args:
        - micro: Microstructure to assign orientations to
        """
        if self.seed:
            np.random.seed(self.seed)

        from .texture import Texture

        orientations = self._generate_orientations(micro)
        micro.orientations = orientations

        return Texture(
            orientations=orientations, representation="euler", symmetry="cubic"
        )

    def _generate_orientations(self, micro):
        """
        Returns a Numpy ndarray of shape (num_grains, 3) of random Euler angles
        """
        if isinstance(micro, np.ndarray):
            n = len(micro)
            include_background = False
        else:
            n = micro.num_grains
            include_background = True

        if include_background:
            orientations = np.zeros((n + 1, 3))
            start_idx = 1
        else:
            orientations = np.zeros((n, 3))
            start_idx = 0

        orientations[start_idx:, 0] = np.random.uniform(0.0, 2 * np.pi, n)
        orientations[start_idx:, 1] = np.arccos(np.random.uniform(-1.0, 1.0, n))
        orientations[start_idx:, 2] = np.random.uniform(0.0, 2 * np.pi, n)

        return orientations
