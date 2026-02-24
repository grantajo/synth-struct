# synth-struct/src/orientation/texture/random.py

"""
This holds the CustomTexture Generator.

This creates a random orientation for each grain in a Microstructure
"""

import numpy as np

from .texture import Texture
from .texture_base import TextureGenerator
from ..phase import Phase


class RandomTexture(TextureGenerator):
    """
    Generate a random (uniform) texture.

    Orientations are sampled uniformly in SO(3) and reduced
    according to crystal symmetry.

    Args:
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(self, phase=None, seed=None):
        if phase is None:
            phase = Phase(
                name="default", crystal_system="cubic", lattice_params=(1, 1, 1)
            )
        if not isinstance(phase, Phase):
            raise TypeError(f"Expected Phase, got {type(phase)}")
        self.phase = phase
        self.seed = seed

    def generate(self, micro):
        """
        Generate random orientations for all grains in the microstructure

        Args:
        - micro: Microstructure to assign orientations to
        """
        orientations = self._generate_orientations(micro)
        micro.orientations = orientations

        return Texture(
            orientations=orientations,
            representation="euler",
            phase=self.phase,
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

        rng = np.random.default_rng(self.seed)

        orientations[start_idx:, 0] = rng.uniform(0.0, 2 * np.pi, n)
        orientations[start_idx:, 1] = np.arccos(rng.uniform(-1.0, 1.0, n))
        orientations[start_idx:, 2] = rng.uniform(0.0, 2 * np.pi, n)

        return orientations
