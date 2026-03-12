# synth-struct/src/orientation/texture/texture.py

"""
Texture class for instantiating a texture for the TextureGenerators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

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


def _get_symmetry_operators() -> dict:
    """
    Return the rotation matrices for all symmetry operators of a point group.

    Only the proper rotations (no improper/inversion elements) are included,
    since we are acting on SO(3) orientations, not crystal vectors.

    Returns
    -------
    dict mapping point_group str -> np.ndarray of shape (s, 3, 3)
    """
    I = np.eye(3)

    # Proper rotation generators
    def Rz(deg):
        t = np.radians(deg)
        return np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

    def Rx(deg):
        t = np.radians(deg)
        return np.array(
            [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
        )

    def Ry(deg):
        t = np.radians(deg)
        return np.array(
            [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
        )

    # Cubic proper rotations (24 elements of group 432)
    cubic_ops = np.array(
        [
            I,
            Rz(90),
            Rz(180),
            Rz(270),
            Rx(90),
            Rx(180),
            Rx(270),
            Ry(90),
            Ry(180),
            Ry(270),
            # 8 body diagonals (C3)
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],  # C3^2
            [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
            # 6 face diagonal C2s
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
            [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
            [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        ],
        dtype=float,
    )

    # Hexagonal proper rotations (12 elements of group 622)
    hex_ops = np.array(
        [
            I,
            Rz(60),
            Rz(120),
            Rz(180),
            Rz(240),
            Rz(300),
            # 6 C2s perpendicular to c-axis
            Rx(180),
            Ry(180),
            np.array([[-0.5, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]),
            np.array(
                [[-0.5, -np.sqrt(3) / 2, 0], [-np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
            ),
            np.array([[0.5, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, -0.5, 0], [0, 0, -1]]),
            np.array(
                [[0.5, -np.sqrt(3) / 2, 0], [-np.sqrt(3) / 2, -0.5, 0], [0, 0, -1]]
            ),
        ],
        dtype=float,
    )

    # Map from point group to its proper rotation subgroup operators
    return {
        # Cubic
        "m-3m": cubic_ops,
        "m-3": cubic_ops,
        "432": cubic_ops,
        "-43m": cubic_ops,
        "23": cubic_ops[:12],
        # Hexagonal
        "6/mmm": hex_ops,
        "6mm": hex_ops,
        "-6m2": hex_ops,
        "622": hex_ops,
        "6/m": hex_ops[:6],
        "-6": hex_ops[:6],
        "6": hex_ops[:6],
        # Trigonal
        "3": np.array([I, Rz(120), Rz(240)]),
        "-3": np.array([I, Rz(120), Rz(240)]),
        "32": np.array(
            [
                I,
                Rz(120),
                Rz(240),
                Rx(180),
                np.array(
                    [[-0.5, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
                np.array(
                    [[-0.5, -np.sqrt(3) / 2, 0], [-np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
            ]
        ),
        "-3m": np.array(
            [
                I,
                Rz(120),
                Rz(240),
                Rx(180),
                np.array(
                    [[-0.5, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
                np.array(
                    [[-0.5, -np.sqrt(3) / 2, 0], [-np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
            ]
        ),
        "3m": np.array(
            [
                I,
                Rz(120),
                Rz(240),
                Rx(180),
                np.array(
                    [[-0.5, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
                np.array(
                    [[-0.5, -np.sqrt(3) / 2, 0], [-np.sqrt(3) / 2, 0.5, 0], [0, 0, -1]]
                ),
            ]
        ),
        # Tetragonal
        "4/mmm": np.array(
            [
                I,
                Rz(90),
                Rz(180),
                Rz(270),
                Rx(180),
                Ry(180),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            ]
        ),
        "4mm": np.array(
            [
                I,
                Rz(90),
                Rz(180),
                Rz(270),
                Rx(180),
                Ry(180),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            ]
        ),
        "-42m": np.array(
            [
                I,
                Rz(90),
                Rz(180),
                Rz(270),
                Rx(180),
                Ry(180),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            ]
        ),
        "422": np.array(
            [
                I,
                Rz(90),
                Rz(180),
                Rz(270),
                Rx(180),
                Ry(180),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            ]
        ),
        "4/m": np.array([I, Rz(90), Rz(180), Rz(270)]),
        "-4": np.array([I, Rz(90), Rz(180), Rz(270)]),
        "4": np.array([I, Rz(90), Rz(180), Rz(270)]),
        # Orthorhombic
        "mmm": np.array([I, Rx(180), Ry(180), Rz(180)]),
        "mm2": np.array([I, Rx(180), Ry(180), Rz(180)]),
        "222": np.array([I, Rx(180), Ry(180), Rz(180)]),
        # Monoclinic
        "2/m": np.array([I, Ry(180)]),
        "m": np.array([I, Ry(180)]),
        "2": np.array([I, Ry(180)]),
        # Triclinic
        "1": np.array([I]),
        "-1": np.array([I]),
    }


_SYMMETRY_OPERATORS = _get_symmetry_operators()


@dataclass
class Texture:
    """
    Data container for crystallographic textures.

    A Texture represents a mapping from grains (or voxels)
    to orientations, together with phase information.
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

    def apply_symmetry_reduction(self) -> "Texture":
        """
        Reduce orientations to the fundamental zone for the phase's point group.

        For each orientation, all symmetry-equivalent rotations are generated
        and the one closest to the identity (i.e. with th elargest real quaternion
        component) is selected as the canonical representative.

        Returns
        -------
        "Texture"
            New Texture with orientations reduced to the fundamental zone,
            in rotmat representation
        """
        rotmat_texture = self.to_representation("rotmat")
        R = rotmat_texture.orientations  # (n, 3, 3)

        sym_ops = _SYMMETRY_OPERATORS[self.phase.point_group]  # (s, 3, 3)

        # Generate all symmetry-equivalent orientations (n, s, 3, 3)
        # equivalent[i, j] = sym_ops[j] @ R[i]
        equivalent = np.einsum("nij,sjk->nsik", R, sym_ops)

        # Convert quaternions to find the one closest to identity.
        # The canonical representative has teh laregest w (real) component,
        # which minimizes the rotation angle away from identity.
        n, s = equivalent.shape[:2]
        flat = equivalent.reshape(n * s, 3, 3)
        quats = rotation_matrix_to_quat(flat)  # (n*s, 4), (w, x, y, z)
        quats = quats.reshape(n, s, 4)

        # Ensure consistent hemisphere (w >= 0) before comparison
        quats = np.where(quats[:, :, :1] < 0, -quats, quats)

        best = np.argmax(quats[:, :, 0], axis=1)  # (n,)
        idx = np.arange(n)
        reduced = equivalent[idx, best]  # (n, 3, 3)

        return Texture(
            orientations=reduced,
            representation="rotmat",
            phase=self.phase,
            metadata=self.metadata.copy(),
        )

    def apply_scatter(self, degspread: float, seed: Optional[int]) -> "Texture":
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

        angles = np.abs(rng.normal(0.0, np.radians(degspread), size=n))
        axes = rng.normal(0.0, 1.0, size=(n, 3))
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axes /= np.where(norms < 1e-12, 1.0, norms)

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

        scattered = np.einsum("nij,njk->nik", base_rotmats, dR)

        scattered_texture = Texture(
            orientations=scattered,
            representation="rotmat",
            phase=self.phase,
            metadata=self.metadata.copy(),
        )

        # This is an exercise in caution. There was no need to add this.
        # if self.phase is not None:
        #     scattered_texture = scattered_texture.apply_symmetry_reduction()

        return scattered_texture.to_representation(self.representation)

    def copy(self) -> "Texture":
        """
        Creates a copy of the texture for analysis
        """
        return Texture(
            orientations=self.orientations.copy(),
            representation=self.representation,
            phase=self.phase,
            metadata=self.metadata.copy(),
        )

    def subset(self, indices: np.ndarray) -> "Texture":
        """
        Return a texture corresponding to a subset of grains.
        """
        return Texture(
            orientations=self.orientations[indices],
            representation=self.representation,
            phase=self.phase,
            metadata=self.metadata.copy(),
        )
