# synth_struct/src/orientation/texture/texture_base.py

"""
Base TextureGenerator class to be able to call each individual generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextureGenerator(ABC):
    """
    Abstract base class for all texture generators

    TextureGenerators generate Texture objects from a Microstructure,
    but do not modify the Microstructure objects themselves.
    """

    @abstractmethod
    def generate(self, micro):
        """
        Generate a Texture for the given microstructure.

        Args:
        - micro: Microstructure object

        Returns:
        - Texture: Texture object
        """

        raise NotImplementedError
