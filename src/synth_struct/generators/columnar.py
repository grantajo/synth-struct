# synth-struct/src/synth_struct/generators/columnar.py

"""
This class holds the ColumnarGenerator class that generates an
anisotropic Voronoi generation that has grains elongated in a
given direction. This generator can only be 3D
"""

import numpy as np

from ..orientation import rotation_z_to_x, rotation_z_to_y
from .gen_base import MicrostructureGenerator
from .gen_utils import aniso_voronoi_assignment, get_seed_coordinates


class ColumnarGenerator(MicrostructureGenerator):
    """
    Weighted Voronoi tessellation generation with columnar grain morphology.

    Generates column-like grains elongated along a specified axis.
    Only works for 3D microstructures.
    """

    def __init__(
        self,
        target_num_grains,
        axis="z",
        aspect_ratio=5.0,
        base_size=8.0,
        size_variation=0.2,
        seed=None,
        chunk_size=500_000,
    ):
        """
        Initialize columnar generator.

        Args:
        - num_grains: int - Number of grains
        - axis: srt - Growth direction ('x', 'y', or 'z')
        - aspect_ratio: float - Length/width ratio
        - size_variation: float - Relative variation in grain sizes (0.0-1.0)
        - seed: int or None - Random seed for reproducibility
        - chunk_size: int - Number of voxels to process per chunk for memory efficiency
        """
        super().__init__(
            target_num_grains=target_num_grains, seed=seed, chunk_size=chunk_size
        )
        self.axis = axis.lower()
        self.aspect_ratio = aspect_ratio
        self.base_size = base_size
        self.size_variation = size_variation

        # Validate axis
        if self.axis not in ["x", "y", "z"]:
            raise ValueError(f"Invalid axis: {self.axis}. Must be 'x', 'y', or 'z'")

        # Will store generation data
        self.seeds = None
        self.scale_factors = None
        self.rotations = None

    def _generate_internal(self, micro):
        """
        Generate columnar grains.

        Args:
        - micro: Microstructure - Instance with 'dimensions' and 'grain_ids'
        """

        if self.seed:
            np.random.seed(self.seed)

        ndim = len(micro.dimensions)

        if ndim != 3:
            raise ValueError("Columnar grains only supported for 3D microstructures")

        # Generate random seed points
        self.seeds = get_seed_coordinates(
            self.target_num_grains, micro.dimensions, self.seed
        )

        # Generate columnar parameters
        self.scale_factors, self.rotations = self._generate_columnar_params(
            self.target_num_grains
        )

        aniso_voronoi_assignment(
            micro, self.seeds, self.scale_factors, self.rotations, self.chunk_size
        )

        print(
            f"Generated {self.target_num_grains} columnar grains along {self.axis}-axis"
        )

    def _generate_columnar_params(self, target_num_grains):
        """
        Generate scale factors and rotation matrices for columnar grains.

        Args:
        - target_num_grains: int - Number of grains

        Returns:
        - scale_factors: np.ndarray of shape (target_num_grains, 3)
        - rotations: list of rotation matrices (each is 3x3)
        """

        scale_factors = np.zeros((target_num_grains, 3))
        rotations = []

        long_axis = self.base_size * self.aspect_ratio
        short_axis = self.base_size

        # Calculate size variation range
        var_min = 1.0 - self.size_variation
        var_max = 1.0 + self.size_variation

        for i in range(target_num_grains):
            # Add variation to each grain
            this_long = long_axis * np.random.uniform(var_min, var_max)
            this_short = short_axis * np.random.uniform(var_min, var_max)

            # Scale factors: two short axes, one long axis (aligned with z initially)
            scale_factors[i] = [this_short, this_short, this_long]

            if self.axis == "x":
                R = rotation_z_to_x()
            elif self.axis == "y":
                R = rotation_z_to_y()
            else:  # 'z'
                R = np.eye(3)

            rotations.append(R)

        return scale_factors, rotations

    def _get_metadata(self):
        """
        Update columnar generator specific metadata.
        """
        metadata = super()._get_metadata()
        metadata.update(
            {
                "axis": self.axis,
                "aspect_ratio": self.aspect_ratio,
                "base_size": self.base_size,
                "size_variation": self.size_variation,
            }
        )

        return metadata
