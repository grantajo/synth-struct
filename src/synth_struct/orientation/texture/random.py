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
        else:
            n = micro.num_grains

        orientations = np.ndarray((n + 1, 3))
        orientations[1:, 0] = np.random.uniform(0.0, 2 * np.pi, n)
        orientations[1:, 1] = np.arccos(np.random.uniform(-1.0, 1.0, n))
        orientations[1:, 2] = np.random.uniform(0.0, 2 * np.pi, n)

        return orientations
