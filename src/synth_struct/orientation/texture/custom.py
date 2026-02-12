# synth_struct/src/orientation/texture/custom.py

"""
This holds the CustomTexture Generator.

This is where the user can decide on a texture based on
two directions (parallel to RD and parallel to RD)
"""

import warnings

import numpy as np

from .texture_base import TextureGenerator
from ..rotation_converter import rotation_matrix_to_euler


def _normalize(v):
    """
    Normalize a vector before rotation and translation
    """
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < 1e-10:
        raise ValueError("Zero-length vector provided")

    return v / norm


class CustomTexture(TextureGenerator):
    """
    Generates a custom texture for a given microstructure

    The texture is defined by crystallographic alignment:
        (hkl) || ND
        [uvw] || RD

    Args:
    - hkl: array-like - Direction parallel to ND as [h, k, l]
    - uvw: array-like - Direction parallel to RD as [u, v, w]
    - degspread: float or None - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(self, hkl, uvw, degspread=5.0, seed=None):
        self.hkl = _normalize(hkl)
        self.uvw = _normalize(uvw)
        uvw_normalized = _normalize(uvw)

        dot_product = np.abs(np.dot(self.hkl, self.uvw))
        if dot_product > 1e-6:
            # Orthogonalize
            uvw_corrected = uvw_normalized - np.dot(uvw_normalized, self.hkl) * self.hkl
            self.uvw = _normalize(uvw_corrected)

            warnings.warn(
                f"uvw was not orthogonal to hkl (dot product = {dot_product:.4f}). "
                f"Corrected uvw to {self.uvw}"
            )
        else:
            self.uvw = uvw_normalized

        self.degspread = degspread
        self.seed = seed

    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """

        n = micro.num_grains

        # Construct orthonormal crystal basis
        # cRD = uvw (rolling direction)
        # cND = hkl (normal direction)
        # cTD = cND x cRD (transverse direction, completes the right-hand system)
        cRD = self.uvw
        cND = self.hkl
        cTD = np.cross(cND, cRD)

        # Rotation matrix: crystal -> sample
        # Each column represents where a crystal axis points in sample frame
        # Sample frame basis: [RD, TD, ND]
        # Crystal frame basis: [cRD, cTD, cND]
        R = np.column_stack([cRD, cTD, cND])

        # Convert to Euler angles
        euler = rotation_matrix_to_euler(R)
        
        orientations = np.zeros((n + 1, 3))
        
        if self.degspread:
            orientations[1:] = self._apply_scatter(euler, n)
        else:
            # Repeat base orienation for all grains
            orientations[1:] = np.tile(euler, (n, 1))

        return orientations

    def _apply_scatter(self, base_euler, n):
        """
        Apply Gaussian scatter around base orientation.

        Args:
        - base_euler: np.ndarray of shape (3,) - Base Euler angles [phi1, Phi, phi2] in degrees.
        - n: int - Number of orientations to generate (number of grains)

        Returns
        - orientations: np.ndarray of shape (n, 3) - Scatter Euler angles in degrees.
        """
        if self.seed:
            np.random.seed(self.seed)

        perturbations = np.random.normal(0.0, self.degspread, size=(n, 3))

        orientations = base_euler + perturbations

        orientations[:, 0] %= 360.0
        orientations[:, 1] = np.clip(orientations[:, 1], 0.0, 180.0)
        orientations[:, 2] %= 360.0

        return orientations
