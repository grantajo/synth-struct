# synth-struct/src/synth_struct/generators/ellipsoidal.py

"""
This class holds the EllipsoidalGenerator class that generates an
anisotropic Voronoi generation that has grains elongated in a
given direction, can be 2D or 3D.
"""

import numpy as np

from ..orientation import (
    create_rotation_matrix_2d,
    euler_to_rotation_matrix,
    rotation_z_to_x,
    rotation_z_to_y,
)
from .gen_base import MicrostructureGenerator
from .gen_utils import aniso_voronoi_assignment, get_seed_coordinates


class EllipsoidalGenerator(MicrostructureGenerator):
    """
    Weighted Voronoi tessellation generation with ellipsoidal variation

    Generates grains with an anisotropic distance etric to create elongated,
    ellipsoidal morphologies
    """

    def __init__(
        self,
        target_num_grains,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=0.5,
        orientation="z",
        base_size=10.0,
        seed=None,
        chunk_size=500_000,
    ):
        """
        Initiates ellipsoidal generator.

        Args:
        - num_grains: int - Number of grains
        - aspect_ratio_mean: float - Mean aspect ratio (length/width)
        - aspect_ratio_std: float - Standard deviation of aspect ratio
        - orientation: str - Preferred elongation direction ('x', 'y', 'z', or 'random')
        - base_size: float - Size of the short axis
        - seed: int or None - Random seed for reproducibility
        - chunk_size: int - Number of voxels to process per chunk for memory efficiency
        """

        self.target_num_grains = target_num_grains
        self.aspect_ratio_mean = aspect_ratio_mean
        self.aspect_ratio_std = aspect_ratio_std
        self.orientation = orientation
        self.base_size = base_size
        self.seed = seed
        self.chunk_size = chunk_size

        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None

    def _generate_internal(self, micro):
        """
        Generate ellipsoidal grains

        Args:
        - micro: Microstructure
            Instance of a Microstructure class with 'dimensions' and 'grain_ids'
        """

        if self.seed:
            np.random.seed(self.seed)

        ndim = len(micro.dimensions)

        # Generate random seed points
        self.seeds = get_seed_coordinates(
            self.target_num_grains, micro.dimensions, self.seed
        )

        # Generate ellipsoidal parameters
        self.scale_factors, self.rotations = self._generate_ellipsoidal_params(
            self.target_num_grains, ndim
        )

        # Perform weighted Voronoi tessellation
        aniso_voronoi_assignment(
            micro, self.seeds, self.scale_factors, self.rotations, self.chunk_size
        )

        print(f"Generated {self.target_num_grains} grains with ellipsoidal morphology")

    def _generate_ellipsoidal_params(self, target_num_grains, ndim):
        """
        Generate scale factors and rotation matrices for ellipsoidal grains.

        Returns:
        - scale_factors: np.ndarray of shape (target_num_grains, ndim)
        - rotations: list of rotation matrices
        """

        # Generate aspect ratios for each grain
        aspect_ratios = np.random.normal(
            self.aspect_ratio_mean, self.aspect_ratio_std, target_num_grains
        )
        aspect_ratios = np.clip(aspect_ratios, 1.5, 10.0)

        scale_factors = np.zeros((target_num_grains, ndim))
        rotations = []

        for i in range(target_num_grains):
            if ndim == 3:
                long_axis = self.base_size * aspect_ratios[i]
                short_axis = self.base_size
                scale_factors[i] = [short_axis, short_axis, long_axis]

                if self.orientation == "random":
                    angles = np.random.uniform(0, 2 * np.pi, 3)
                    R = euler_to_rotation_matrix(angles)
                elif self.orientation == "x":
                    R = rotation_z_to_x()
                elif self.orientation == "y":
                    R = rotation_z_to_y()
                elif self.orientation == "z":
                    R = np.eye(3)
                else:
                    R = np.eye(3)

                rotations.append(R)

            else:  # 2D
                long_axis = self.base_size * aspect_ratios[i]
                short_axis = self.base_size
                scale_factors[i] = [short_axis, long_axis]

                if self.orientation == "random":
                    angle = np.random.uniform(0, 2 * np.pi)
                    R = create_rotation_matrix_2d(angle)
                elif self.orientation == "x":
                    R = create_rotation_matrix_2d(0)
                elif self.orientation == "y":
                    R = create_rotation_matrix_2d(np.pi / 2)
                else:
                    R = np.eye(2)

                rotations.append(R)

        return scale_factors, rotations

    def _get_metadata(self):
        """
        Update ellipsoidal generator specific metadata
        """
        metadata = super()._get_metadata()
        metadata.update(
            {
                "aspect_ratio_mean": self.aspect_ratio_mean,
                "aspect_ratio_std": self.aspect_ratio_std,
                "orientation": self.orientation,
                "base_size": self.base_size,
            }
        )

        return metadata
