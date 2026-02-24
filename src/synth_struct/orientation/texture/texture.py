# synth-struct/src/orientation/texture/texture.py

"""
Texture class for instantiating a texture for the TextureGenerators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..phase import Phase
from ..rotation_converter import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat,
)


@dataclass
class Texture:
    """
    Data container for crystallographic textures.

    A Texture represents a mapping from grains (or voxels)
    to orientations, together with symmetry information.
    """

    orientations: np.ndarray
    representation: str
    phase: Phase
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.orientations, np.ndarray):
            raise TypeError("Orientations must be a NumPy array")

        if self.orientations.ndim < 2:
            raise ValueError(
                "Orientations must have shape (n, ...) where n is number of grains"
            )

        if self.representation not in {"euler", "quat", "rotmat"}:
            raise ValueError(
                f"Unknown representation '{self.representation}'. "
                f"Expected 'euler', 'quat', or 'rotmat'."
            )

        if not isinstance(self.phase, Phase):
            raise TypeError("Phase must be a Phase object, got {type(self.phase)}")

    @property
    def n_orientations(self) -> int:
        """
        Returns number of orientations
        """
        return self.orientations.shape[0]

    @property
    def symmetry(self) -> str:
        """Convenience accessor for crystal system via phase."""
        return self.phase.crystal_system

    def to_representation(self, representation: str) -> "Texture":
        """
        Convert from one texture type to another.

        Args:
        - representation: {'euler', 'quat', or 'rotmat'}
        """

        if representation == self.representation:
            return self.copy()

        if self.representation == "euler" and representation == "quat":
            new_orientations = euler_to_quat(self.orientations)

        elif self.representation == "euler" and representation == "rotmat":
            new_orientations = euler_to_rotation_matrix(self.orientations)

        elif self.representation == "quat" and representation == "euler":
            new_orientations = quat_to_euler(self.orientations)

        elif self.representation == "quat" and representation == "rotmat":
            new_orientations = quat_to_rotation_matrix(self.orientations)

        elif self.representation == "rotmat" and representation == "euler":
            new_orientations = rotation_matrix_to_euler(self.orientations)

        elif self.representation == "rotmat" and representation == "quat":
            new_orientations = rotation_matrix_to_quat(self.orientations)

        else:
            raise ValueError(
                f"Conversion from '{self.representation}' to "
                f"'{representation}' not supported."
            )

        return Texture(
            orientations=new_orientations,
            representation=representation,
            phase=self.phase,
            metadata=self.metadata.copy(),
        )

    def apply_scatter(self, degspread: float, seed: int or None) -> "Texture":
        """
        Apply scatter around each orientation in rotation space.

        Perturbations are applied as small random rotations (Rodrigues
        formula) composed with the base rotation.

        Parameters
        ----------
        degspread : float
            Standard deviation of scatter in degrees
        seed : int or None
            Random seed for reproducibility

        Returns
        -------
        "Texture"
            Texture with scattered orientations in the same
            representation
        """
        rotmat_texture = self.to_representation("rotmat")
        base_rotmats = rotmat_texture.orientations

        rng = np.random.default_rng(seed)
        n = self.n_orientations

        angles = rng.normal(0.0, np.radians(degspread), size=n)
        axes = rng.normal(0.0, 1.0, size=(n, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)

        K = np.zeros((n, 3, 3))
        K[:, 0, 1] = -axes[:, 2]
        K[:, 0, 2] = axes[:, 1]
        K[:, 1, 0] = axes[:, 2]
        K[:, 1, 2] = -axes[:, 0]
        K[:, 2, 0] = -axes[:, 1]
        K[:, 2, 1] = axes[:, 0]

        s = np.sin(angles)[:, None, None]
        c = (1 - np.cos(angles))[:, None, None]
        I = np.eye(3)[None, :, :]
        dR = I + s * K + c * np.einsum("nij,njk->nik", K, K)

        scattered = np.einsum("nij,njk->nik", dR, base_rotmats)

        scattered_texture = Texture(
            orientations=scattered,
            representation="rotmat",
            symmetry=self.symmetry,
            metadata=self.metadata.copy(),
        )

        return scattered_texture.to_representation(self.representation)

    def copy(self) -> "Texture":
        """
        Creates a copy of the texture for analysis
        """
        return Texture(
            orientations=self.orientations.copy(),
            representation=self.representation,
            symmetry=self.symmetry,
            metadata=self.metadata.copy(),
        )

    def subset(self, indices: np.ndarray) -> "Texture":
        """
        Return a texture corresponding to a subset of grains.
        """
        return Texture(
            orientations=self.orientations[indices],
            representation=self.representation,
            symmetry=self.symmetry,
            metadata=self.metadata.copy(),
        )
