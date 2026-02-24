# synth-struct/src/orientation/texture/texture_base.py

"""
Base TextureGenerator class to be able to call each individual generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..phase import Phase


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

    @property
    def phase(self) -> Optional[Phase]:
        """Returns the phase associated with this generator, if any."""
        return getattr(self, "_phase", None)

    @phase.setter
    def phase(self, value: Phase):
        if not isinstance(value, Phase):
            raise TypeError(f"Expected Phase, got {type(value)}")
        self._phase = value
