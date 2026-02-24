# synth-struct/src/orientation/texture/custom.py

"""
This holds the CustomTexture Generator.

This is where the user can decide on a texture based on
two directions (parallel to RD and parallel to RD)
"""

import warnings

import numpy as np

from .texture import Texture
from .texture_base import TextureGenerator
from ..rotation_converter import rotation_matrix_to_euler
from ..phase import Phase


def _normalize(v):
    """
    Normalize a vector before rotation and translation
    """
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < 1e-10:
        raise ValueError("Zero-length vector provided")

    return v / norm


def _crystal_to_cartesian(indices, crystal_system, lattice_params):
    """
    Convert crystal direction indices into a Cartesian vector.

    This is for hexagonal, non-orthogonal bases.

    Parameters:
        - indices: array-like - [h, k, l] or [h, k, i, l]
        - crystal_system: str - 'cubic' or 'hexagonal'
        - lattice_params: tuple - (a, b, c) lattice parameters

    Returns:
        - np.ndarray of shape (3,) in Cartesian coordinates
    """
    indices = np.array(indices, dtype=float)
    a, b, c = lattice_params

    if crystal_system == "cubic":
        return indices

    elif crystal_system == "hexagonal":

        if len(indices) == 4:
            h, k, i, l = indices
            # Verify i = -(h+k) within tolerance
            if abs(i + h + k) > 1e-6:
                warnings.warn(
                    f"Miller-Bravais index i={i:.4f} does not equal"
                    f"-(h+k)={-(h+k):.4f}. "
                    f"Ignoring i and computing from h, k."
                )
            indices = np.array([h, k, l])

        h, k, l = indices
        # Convert hexagonal Miller to orthogonal Cartesian
        x = a * (h + k * 0.5)
        y = a * (k * np.sqrt(3) / 2)
        z = c * l

        return np.array([x, y, z])

    else:
        raise ValueError(
            f"Unknown crystal system '{crystal_system}'. "
            f"Expected 'cubic' or 'hexagonal'."
        )


class CustomTexture(TextureGenerator):
    """
    Generates a custom texture for a given microstructure

    The texture is defined by crystallographic alignment:
        (hkl) || ND
        [uvw] || RD

    For hexagonal materials, Miller-Bravais 4-index notation is accepted:
        (hkil) || ND
        [uvtw] || RD

    Args:
    - hkl: array-like - Direction parallel to ND as [h, k, l]
    - uvw: array-like - Direction parallel to RD as [u, v, w]
    - phase: Phase object
    - degspread: float or None - Gaussian spread around ideal orientation (degrees)
    - seed: int or None - Random seed for reproducibility
    """

    def __init__(
        self,
        hkl,
        uvw,
        phase=None,
        degspread=5.0,
        seed=None,
    ):
        if phase is None:
            phase = Phase(
                name="default", crystal_system="cubic", lattice_params=(1, 1, 1)
            )
        if not isinstance(phase, Phase):
            raise TypeError(f"Expected Phase, got {type(phase)}")
        if degspread is not None and degspread < 0:
            raise ValueError("scale < 0")

        self.phase = phase
        self.degspread = degspread
        self.seed = seed

        hkl_cart = _crystal_to_cartesian(
            hkl, phase.crystal_system, phase.lattice_params
        )
        uvw_cart = _crystal_to_cartesian(
            uvw, phase.crystal_system, phase.lattice_params
        )

        self.hkl = _normalize(hkl_cart)

        dot_product = np.abs(np.dot(self.hkl, _normalize(uvw_cart)))
        if dot_product > 1e-6:
            # Orthogonalize
            uvw_corrected = (
                _normalize(uvw_cart) - np.dot(_normalize(uvw_cart), self.hkl) * self.hkl
            )
            self.uvw = _normalize(uvw_corrected)

            warnings.warn(
                f"uvw was not orthogonal to hkl (dot product = {dot_product:.4f}). "
                f"Corrected uvw to {self.uvw}"
            )
        else:
            self.uvw = _normalize(uvw_cart)

    def generate(self, micro):
        """Generate a Texture for the given microstructure."""

        orientations = self._generate_orientations(micro)

        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=self.phase,
        )
        if self.degspread is not None and self.degspread > 0:
            texture = texture.apply_scatter(self.degspread, seed=self.seed)

        return texture

    def _generate_orientations(self, micro):
        """
        Returns a (num_grains, 3) array of Euler angles
        """
        if isinstance(micro, np.ndarray):
            n = len(micro)
            include_background = False
        else:
            n = micro.num_grains
            include_background = True

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
        R = np.vstack([cRD, cTD, cND])

        # Convert to Euler angles
        euler = rotation_matrix_to_euler(R)

        if include_background:
            orientations = np.zeros((n + 1, 3))
            start_idx = 1
        else:
            orientations = np.zeros((n, 3))
            start_idx = 0

        # Repeat base orienation for all grains
        orientations[start_idx:] = np.tile(euler, (n, 1))

        return orientations
