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

    def _generate_orientations(self, micro):
        """
        Returns a Numpy ndarray of shape (num_grains, 3) of random Euler angles
        """
        if self.seed:
            np.random.seed(self.seed)

        n = micro.num_grains

        orientations = np.ndarray((n, 3))
        orientations[:, 0] = np.random.uniform(0.0, 2 * np.pi, n)
        orientations[:, 1] = np.arccos(np.random.uniform(-1.0, 1.0, n))
        orientations[:, 2] = np.random.uniform(0.0, 2 * np.pi, n)

        return orientations
