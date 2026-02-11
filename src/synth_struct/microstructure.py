# synth_struct/src/synth_struct/microstructure.py

"""
Microstructure representation for voxel-based synthetic materials.

Provides the ``Microstructure`` class, a container for 2D and 3D
voxelized grain structures with associated physical scaling,
crystallographic symmetry, and attached data fields.
"""

import numpy as np


class Microstructure:
    """
    Data container for a synthetic microstructure
    """

    def __init__(self, dimensions, resolution, units="micron", symmetry=None):
        """
        Initiation of a Microstructure class

        Args:
        - dimensions: tuple - (nx, ny) for 2D, (nx, ny, nz) for 3D
        - resolution: float - physical size per voxel
        - units: str - physical units
        - symmetry: str, optional - Crystal symmetry ('cubic' or 'hexagonal')
        """

        self.dimensions = tuple(dimensions)
        self.resolution = resolution
        self.units = units
        self.symmetry = symmetry

        self.grain_ids = np.zeros(
            self.dimensions, dtype=np.int32
        )  # 0 = background (e.g. unindexed EBSD)

        self.fields = {}
        self.metadata = {}

    @property
    def num_grains(self):
        """Number of grain excluding the background"""
        return int(self.grain_ids.max())

    def attach_field(self, name, array):
        """
        Attach per-grain or per-voxel data (orientations, stiffnesses, etc.)
        """
        self.fields[name] = array

    def get_field(self, name):
        return self.fields[name]
