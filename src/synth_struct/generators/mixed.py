# synth-struct/src/synth_struct/generators/mixed.py

"""
This class holds the MixedGenerator class that generates an
anisotropic Voronoi generation that generates a microstructure
with both equiaxed and elongated grains, can be 2D or 3D.
"""

import numpy as np

from ..orientation import create_rotation_matrix_2d, euler_to_rotation_matrix
from .gen_base import MicrostructureGenerator
from .gen_utils import aniso_voronoi_assignment, get_seed_coordinates


class MixedGenerator(MicrostructureGenerator):
    """
    Weighted Voronoi tessellation generation with mixed grain morphologies.

    Generates a mixture of ellipsiodal (elongated) and equiaxed (spherical) grains.
    """

    def __init__(
        self,
        target_num_grains,
        ellipsoid_fraction=0.5,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=0.5,
        base_size=10.0,
        seed=None,
        chunk_size=500_000,
    ):
        """
        Initialize mixed generator.

        Args:
        - target_num_grains: int - Number of grains
        - ellipsoid_fraction: float - Fraction of ellipsoidal grains (0.0-1.0)
        - aspect_ratio_mean: float - Mean aspect ratio for ellipsoidal grains
        - aspect_ratio_std: float - Standard deviation of aspect ratio
        - base_size: float - Base size for all grains
        - seed: int or None - Random seed for reproducibility
        - chunk_size: int - Number of voxels to process per chunk for memory efficiency
        """

        self.target_num_grains = target_num_grains
        self.ellipsoid_fraction = np.clip(ellipsoid_fraction, 0.0, 1.0)
        self.aspect_ratio_mean = aspect_ratio_mean
        self.aspect_ratio_std = aspect_ratio_std
        self.base_size = base_size
        self.seed = seed
        self.chunk_size = chunk_size

        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None

    def _generate_internal(self, micro):
        """
        Generate mixed grain morphologies.

        Args:
        - micro: Microstructure - Instance with 'dimensions' and 'grain_ids'
        """

        if self.seed:
            np.random.seed(self.seed)

        ndim = len(micro.dimensions)

        # Generate random seed points
        self.seeds = get_seed_coordinates(
            self.target_num_grains, micro.dimensions, self.seed
        )

        # Generate mixed grain parameters
        self.scale_factors, self.rotations = self._generate_mixed_params(
            self.target_num_grains, ndim
        )

        # Perform weighted Voronoi tessellation
        aniso_voronoi_assignment(
            micro, self.seeds, self.scale_factors, self.rotations, self.chunk_size
        )

        num_ellipsoidal = int(self.target_num_grains * self.ellipsoid_fraction)
        num_equiaxed = self.target_num_grains - num_ellipsoidal

        print(
            f"Generated {self.target_num_grains} grains: "
            f"{num_ellipsoidal} ellipsoidal, {num_equiaxed} equiaxed"
        )

    def _generate_mixed_params(self, target_num_grains, ndim):
        """
        Generate scale factors and rotation matrices for mixed grain morphologies.

        Args:
        - target_num_grains: int - Number of grains
        - ndim: int - Number of dimensions (2 or 3)

        Returns:
        - scale_factors: np.ndarray of shape (target_num_grains, ndim)
        - rotations: list of rotation matrices (each is ndim x ndim)
        """

        num_ellipsoidal = int(target_num_grains * self.ellipsoid_fraction)

        scale_factors = np.zeros((target_num_grains, ndim))
        rotations = []

        for i in range(target_num_grains):
            if i < num_ellipsoidal:  # Ellipsoidal grains
                aspect_ratio = np.random.normal(
                    self.aspect_ratio_mean, self.aspect_ratio_std
                )
                aspect_ratio = np.clip(aspect_ratio, 1.5, 15.0)

                if ndim == 3:
                    scale_factors[i] = [
                        self.base_size,
                        self.base_size,
                        self.base_size * aspect_ratio,
                    ]
                    angles = np.random.uniform(0, 2 * np.pi, 3)
                    R = euler_to_rotation_matrix(angles)
                else:  # 2D
                    scale_factors[i] = [self.base_size, self.base_size * aspect_ratio]
                    angle = np.random.uniform(0, 2 * np.pi)
                    R = create_rotation_matrix_2d(angle)

            else:  # Equiaxed grains
                if ndim == 3:
                    scale_factors[i] = [self.base_size, self.base_size, self.base_size]
                else:  # 2D
                    scale_factors[i] = [self.base_size, self.base_size]
                R = np.eye(ndim)

            rotations.append(R)

        return scale_factors, rotations

    def _get_metadata(self):
        """
        Update mixed generator specific metadata.
        """
        metadata = super()._get_metadata()
        metadata.update(
            {
                "ellipsoid_fraction": self.ellipsoid_fraction,
                "aspect_ratio_mean": self.aspect_ratio_mean,
                "aspect_ratio_std": self.aspect_ratio_std,
                "base_size": self.base_size,
            }
        )

        return metadata
