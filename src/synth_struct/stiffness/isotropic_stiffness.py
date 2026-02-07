# synth_struct/src/synth_struct/stiffness/isotropic_stiffness.py

from __future__ import annotations
import numpy as np
from .stiffness_base import StiffnessGenerator
from .stiffness import Stiffness
from .stiffness_utils import rotate_stiffness_tensors_batch

"""
Isotropic Stiffness Tensor
[  c, lam, lam,  0,  0,  0],
[lam,   c, lam,  0,  0,  0],
[lam, lam,   c,  0,  0,  0],
[  0,   0,   0, mu,  0,  0],
[  0,   0,   0,  0, mu,  0],
[  0,   0,   0,  0,  0, mu]

lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
c = lam + 2 * mu
"""


class IsotropicStiffnessGenerator(StiffnessGenerator):
    """
    Generates stiffness tensors for Isotropic materials.

    Isotropic materials have 2 independent elastic constants: E, nu.
    """

    def __init__(self, E: float, nu: float):
        """
        Initialize cubic stiffness generator.

        Args:
        - E: Elastic modulus (GPa)
        - nu: Poisson ratio
        """
        self.E = E
        self.nu = nu
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.c = self.lam + 2 * self.mu
        self._base_tensor = self._create_base_tensor()

    def _create_base_tensor(self) -> np.ndarray:
        """Create the base stiffness tensor for Isotropic symmetry."""

        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = self.c
        C[0, 1] = C[0, 2] = C[1, 2] = self.lam
        C[1, 0] = C[2, 0] = C[2, 1] = self.lam
        C[3, 3] = C[4, 4] = C[5, 5] = self.mu
        return C

    def generate(self, micro, texture):
        """
        Generate rotated stiffness tensors for each grain/voxel.

        Args:
        - micro: Microstructure object
        - texture: Texture object with orientations

        Returns:
        - Stiffness object with rotated tensors
        """
        # Convert texture to rotation matrices if needed
        if texture.representation != "rotmat":
            texture_rotmat = texture.to_representation("rotmat")
        else:
            texture_rotmat = texture

        # Rotate stiffness tensors
        rotated_tensors = rotate_stiffness_tensors_batch(
            self._base_tensor, texture_rotmat.orientations
        )

        return Stiffness(
            stiffness_tensors=rotated_tensors,
            crystal_structure="cubic",
            metadata={
                "E": self.E,
                "nu": self.nu,
                "lam": self.lam,
                "mu": self.mu,
                "c": self.c,
            },
        )
