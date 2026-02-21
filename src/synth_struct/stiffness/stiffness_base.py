# synth-struct/src/synth_struct/stiffness/stiffness_base.py

"""
StiffnessGenerator class

Instantiates stiffness rotation generators (not really a generator, but oh well)
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class StiffnessGenerator(ABC):
    """
    Abstract base class for all stiffness generators.
    StiffnessGenerators generate stiffness objects from a
    Microstructure and Texture, but do not modify the Microstructure
    """

    @abstractmethod
    def generate(self, micro, texture):
        """
        Generate a Stiffness object for the given Microstructure and Texture

        Args:
        - micro: Microstructure object
        - texture: Texture object containing orientations

        Returns:
        - Stiffness: Stiffness object with rotated stiffness tensors
        """
        raise NotImplementedError
