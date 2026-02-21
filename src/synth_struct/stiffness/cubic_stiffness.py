# synth-struct/src/synth_struct/stiffness/cubic_stiffness.py

"""
Rotation calculator for local stiffness tensors in a cubic system.

Base Cubic Stiffness Tensor:
[C11, C12, C12,   0,   0,   0],
[C12, C11, C12,   0,   0,   0],
[C12, C12, C11,   0,   0,   0],
[  0,   0,   0, C44,   0,   0],
[  0,   0,   0,   0, C44,   0],
[  0,   0,   0,   0,   0, C44],
"""

from __future__ import annotations

import numpy as np

from .stiffness_base import StiffnessGenerator
from .stiffness import Stiffness
from .stiffness_utils import rotate_stiffness_tensors_batch


class CubicStiffnessGenerator(StiffnessGenerator):
    """
    Generates stiffness tensors for cubic crystal structures.

    Cubic materials have 3 independent elastic constants: C11, C12, C44
    """

    def __init__(self, C11: float, C12: float, C44: float):
        """
        Initialize cubic stiffness generator.

        Args:
        - C11: Elastic constant C11 (GPa)
        - C12: Elastic constant C12
        - C44: Elastic constant C44
        """
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44
        self._base_tensor = self._create_base_tensor()

    def _create_base_tensor(self) -> np.ndarray:
        """
        Create the base stiffness tensor for cubic symmetry.
        """
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = self.C11
        C[0, 1] = C[0, 2] = C[1, 2] = self.C12
        C[1, 0] = C[2, 0] = C[2, 1] = self.C12
        C[3, 3] = C[4, 4] = C[5, 5] = self.C44
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
                "C11": self.C11,
                "C12": self.C12,
                "C44": self.C44,
            },
        )
