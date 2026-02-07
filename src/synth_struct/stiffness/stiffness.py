# synth_struct/src/synth_struct/stiffness/stiffness.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class Stiffness:
    """
    Data container for stiffness tensors.

    A Stiffness object represents a mapping from grains (or voxels)
    to rotated stiffness tensors (4th-order tensors in Voigt notation).

    Attributes:
    - stiffness_tensors: np.ndarray of shape (n, 6, 6) where n is number of grains/voxels
    - crystal_structure: str indicating crystal type (e.g., 'cubic', 'hexagonal', 'isotropic')
    - metadata: Dict for additional information
    """

    stiffness_tensors: np.ndarray  # Shape: (n, 6, 6) in Voigt notation
    crystal_structure: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.stiffness_tensors, np.ndarray):
            raise TypeError("Stiffness tensors must be a NumPy array")

        if self.stiffness_tensors.ndim != 3:
            raise ValueError(
                "Stiffness tensors must have shape (n, 6, 6) where n is number of grains"
            )

        if self.stiffness_tensors.shape[1:] != (6, 6):
            raise ValueError(
                f"Each stiffness tensor must be 6x6 (Voigt notation). "
                f"Got shape {self.stiffness_tensors.shape[1:]}"
            )

        if not isinstance(self.crystal_structure, str):
            raise TypeError(
                "crystal_structure must be a string (e.g. 'cubic', 'hexagonal', 'isotropic')"
            )

    @property
    def n_tensors(self) -> int:
        """Number of stiffness tensors."""
        return self.stiffness_tensors.shape[0]

    def copy(self) -> "Stiffness":
        """Create a deep copy of this Stiffness object."""
        return Stiffness(
            stiffness_tensors=self.stiffness_tensors.copy(),
            crystal_structure=self.crystal_structure,
            metadata=self.metadata.copy(),
        )

    def subset(self, indices: np.ndarray) -> "Stiffness":
        """
        Return a Stiffness object corresponding to a subset of grains.

        Args:
        - indices: NumPy array of grain indices to extract

        Returns:
        - Stiffness: New Stiffness object with selected tensors
        """
        return Stiffness(
            stiffness_tensors=self.stiffness_tensors[indices],
            crystal_structure=self.crystal_structure,
            metadata=self.metadata.copy(),
        )
