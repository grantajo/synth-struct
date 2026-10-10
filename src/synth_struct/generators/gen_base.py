# synth-struct/src/synth_struct/generators/gen_base.py

"""
This holds the MicrostructureGenerator class and associated basic functions.
"""

import numpy as np


class MicrostructureGenerator:
    """
    Base class for microstructure generators.

    All generator classes should inherit from this and implement:
        generate(micro: Microstructure) -> None

    Provides shared utilities like allocating per-grain arrays
    """

    def __init__(self, target_num_grains, seed=None, chunk_size: int = 500_000):
        """
        Initialize base attributes that subclasses will populate.
        """
        self.target_num_grains = target_num_grains
        self.seed = seed
        self.chunk_size = chunk_size
        self.rng = np.random.default_rng(seed)

        self.seeds = None
        self.weights = None
        self.target_volumes = None
        self.actual_volumes = None

        self.scale_factors = None
        self.rotations = None

    def generate(self, micro):
        """
        Generate grains in the given Microstructure
        Calls _generate_internal() then allocates arrays.
        """
        self._validate_microstructure(micro)

        self._generate_internal(micro)

        micro.metadata["generator"] = self._get_metadata()

        if self.seeds is not None:
            micro.metadata["seed_coordinates"] = self.seeds.copy()

        return micro

    def _validate_microstructure(self, micro):
        """
        Validate that the required attributes exist in the microstructure
        """
        for attr in ("dimensions", "resolution", "units"):
            if not hasattr(micro, attr):
                raise AttributeError(f"micro must define {attr}")

    def _generate_internal(self, micro):
        """
        Internal generation logic. Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the _generate_internal() method."
        )

    def _get_metadata(self):
        return {
            "method": type(self).__name__,
            "seed": self.seed,
            "target_num_grains": self.target_num_grains,
        }

    """
    Might want to move to Microstructure class
    def _allocate_grain_arrays(self, micro):
        ""
        Allocate arrays for per-grain properties in the Microstructure.

        Creates:
            micro.orientations: np.ndarray of shape (num_grains+1, 3) - Euler angles (radians)
            micro.stiffness: np.ndarray of shape (num_grains+1, 6, 6) - Stiffness tensors (GPa)
            micro.phase: np.ndarray of shape (num_grains+1,) - integer phase IDs

        Note: Index 0 is reserved for background; graind Ids 1..num_grains map to indices 1..num_grians
        ""

        n = micro.num_grains + 1  # +1 to include background (ID=0)
        micro.orientations = np.zeros((n, 3), dtype=np.float64)
        micro.stiffness = np.zeros((n, 6, 6), dtype=np.float32)
        micro.phase = np.zeros(n, dtype=np.int8)
    """

    def get_seed_coordinates(self):
        """
        Return the coordinates of the Voronoi seed points
        """
        if self.seeds is None:
            raise AttributeError("Generate the microstructure first.")
        return self.seeds.copy()
