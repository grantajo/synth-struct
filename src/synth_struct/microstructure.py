# synth-struct/src/synth_struct/microstructure.py

"""
Microstructure representation for voxel-based synthetic materials.

Provides the ``Microstructure`` class, a container for 2D and 3D
voxelized grain structures with associated physical scaling,
crystallographic symmetry, and attached data fields.
"""

import copy
from typing import Dict, Optional

import numpy as np

from .orientation.phase import Phase
from .orientation.rotation_converter import euler_to_quat


class Microstructure:
    """
    Data container for a synthetic microstructure
    """

    def __init__(self, dimensions, resolution, units="um", phase=None):
        """
        Initiation of a Microstructure class

        Args:
        - dimensions: tuple - (nx, ny) for 2D, (nx, ny, nz) for 3D
        - resolution: float - physical size per voxel
        - units: str - physical units
        - phase =
        """

        self.dimensions = tuple(dimensions)
        self.resolution = resolution
        self.units = units

        self._grain_ids = np.zeros(
            self.dimensions, dtype=np.int32
        )  # 0 = background (e.g. unindexed EBSD)
        
        self._quaternion_cache = None

        self.fields = {}
        self.metadata = {}

        if phase is None:
            self._phases = {}
            self._phase_ids = None
        elif isinstance(phase, Phase):
            # Single phase - no phase_ids array needed
            self._phases = {0: phase}
            self._phase_ids = None
        elif isinstance(phase, dict):
            # Multiphase - caller provides (phase_id: Phase) and
            # must set phase ids
            self._phases = phase
            self._phase_ids = np.zeros(self.dimensions, dtype=np.int32)
        else:
            raise TypeError(
                f"phase must be a Phase or Dict[int, Phase], got " f"{type(phase)}"
            )

    # -------------------------------------------------------------------------
    # Grain IDs
    # -------------------------------------------------------------------------

    @property
    def grain_ids(self):
        return self._grain_ids

    @grain_ids.setter
    def grain_ids(self, value):
        value = np.asarray(value, dtype=np.int32)
        if value.shape != self.dimensions:
            raise ValueError(
                f"grain_ids shape {value.shape} does not match "
                f"dimensions {self.dimensions}"
            )
        self._grain_ids = value

    @property
    def num_grains(self):
        """Number of grains excluding the background"""
        unique = np.unique(self._grain_ids)
        return len(unique[unique != 0])

    # -------------------------------------------------------------------------
    # Phase IDs
    # -------------------------------------------------------------------------

    @property
    def phase_ids(self):
        return self._phase_ids

    @phase_ids.setter
    def phase_ids(self, value):
        value = np.asarray(value, dtype=np.int32)
        if value.shape != self.dimensions:
            raise ValueError(
                f"phase_ids shape {value.shape} does not match "
                f"dimensions {self.dimensions}"
            )
        unknown = set(np.unique(value)) - {-1} - set(self._phases.keys())
        if unknown:
            raise ValueError(
                f"phase_ids contains unknown phase IDs {unknown}. "
                f"Known phases: {list(self._phases.keys())}"
            )
        self._phase_ids = value

    # -------------------------------------------------------------------------
    # Phases
    # -------------------------------------------------------------------------

    @property
    def phases(self) -> Dict[int, Phase]:
        return self._phases

    def add_phase(self, phase_id: int, phase: Phase) -> int:
        """Add or replace a phase by ID."""
        if not isinstance(phase, Phase):
            raise TypeError(f"Expected Phase, got {type(phase)}")
        self._phases[phase_id] = phase
        return phase_id

    def get_phase(self, phase_id: int) -> Phase:
        """Get phase by ID"""
        if phase_id not in self._phases:
            raise KeyError(f"No phase with ID {phase_id}")
        return self._phases[phase_id]

    def get_phase_for_grain(self, grain_id: int) -> Optional[Phase]:
        """
        Return the Phase for a given grain.

        For single-phase microstructures, returns single phase
        For multiphase, looks up via phase_ids at any voxel belonging
        to that grain.
        """
        if len(self._phases) == 1:
            return next(iter(self._phases.values()))

        if self._phase_ids is None:
            return None

        voxels = np.argwhere(self._grain_ids == grain_id)
        if len(voxels) == 0:
            raise ValueError(f"No voxels found for grain_ids {grain_id}")

        phase_id = self._phase_ids[tuple(voxels[0])]

        return self._phases.get(int(phase_id))

    # -------------------------------------------------------------------------
    # Orientations
    # -------------------------------------------------------------------------

    @property
    def orientations(self):
        return self.fields.get("orientations")

    @orientations.setter
    def orientations(self, value):
        self.fields["orientations"] = value

    # -------------------------------------------------------------------------
    # Texture
    # -------------------------------------------------------------------------

    def get_quaternions(self):
        if self._quaternion_cache is None:
            self._quaternion_cache = euler_to_quat(self.orientations)
        return self._quaternion_cache

    def assign_texture(self, texture, grain_ids=None):
        """
        Assigna texture to the microstructure, updating both
        orientations and phase_ids automatically.

        Args:
            - texture: Texture object returned by a generator
            - grain_ids: np.ndarray or None - Grain IDs to assign to.
                If None, assigns to all grains
        """
        self._quaternion_cache = None
        phase = texture.phase

        existing_id = next(
            (
                pid
                for pid, p in self._phases.items()
                if p.name == phase.name and p.space_group == phase.space_group
            ),
            None,
        )

        if existing_id is None:
            phase_id = max(self._phases.keys(), default=-1) + 1
            self.add_phase(phase_id, phase)
        else:
            phase_id = existing_id

        if grain_ids is not None:
            self.orientations[grain_ids] = texture.orientations
        else:
            self.orientations = texture.orientations

        if self._phase_ids is None:
            self._phase_ids = np.zeros(self.dimensions, dtype=np.int32)
        if grain_ids is not None:
            mask = np.isin(self._grain_ids, grain_ids)
            self._phase_ids[mask] = phase_id
        else:
            self._phase_ids[:] = phase_id

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

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
            phase=self.phases,
        )
        new_micro.grain_ids = self.grain_ids.copy()
        new_micro._quaternion_cache = None
        new_micro.fields = {k: v.copy() for k, v in self.fields.items()}
        new_micro.metadata = copy.deepcopy(self.metadata)
        return new_micro
