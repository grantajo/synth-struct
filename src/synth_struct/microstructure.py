# synth-struct/src/synth_struct/microstructure.py

"""
Microstructure representation for voxel-based synthetic materials.

Provides the ``Microstructure`` class, a container for 2D and 3D
voxelized grain structures with associated physical scaling,
crystallographic symmetry, and attached data fields.
"""

import copy

import numpy as np


class Microstructure:
    """
    Data container for a synthetic microstructure
    """

    def __init__(self, dimensions, resolution, units="um", symmetry=None):
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

    @property
    def orientations(self):
        return self.fields.get("orientations")

    @orientations.setter
    def orientations(self, value):
        self.fields["orientations"] = value

    def attach_field(self, name, array):
        """
        Attach per-grain or per-voxel data (orientations, stiffnesses, etc.)
        """
        self.fields[name] = array

    def get_field(self, name):
        """
        Get the data from an attached field in the Microstructure class
        (orientations, stiffnesses, etc.)"""
        return self.fields[name]

    def copy(self):
        """
        Return a deep copy of the Microstructure
        """
        new_micro = Microstructure(
            dimensions=self.dimensions,
            resolution=self.resolution,
            units=self.units,
            symmetry=self.symmetry,
        )
        new_micro.grain_ids = self.grain_ids.copy()
        new_micro.fields = {k: v.copy() for k, v in self.fields.items()}
        new_micro.metadata = copy.deepcopy(self.metadata)
        return new_micro
