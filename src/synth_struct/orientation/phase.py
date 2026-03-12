# synth-struct/src/synth_struct/orientation/phase.py

"""
This class holds phase information for the microstructure.

Provides the ```Phase``` class, a container for crystal symmetry,
lattice parameters, and associated crystallographic information.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .phase_constants import KNOWN_PHASES, ALIASES, VALID_POINT_GROUPS


def available_presets() -> list[str]:
    """Return list of available phase presets and aliases."""
    return list(KNOWN_PHASES.keys()) + list(ALIASES.keys())


@dataclass
class Phase:
    """
    Container for crystallographic information.

    Args:
        - name: str - Name of the phase (e.g. 'Ti-alpha', 'austenite')
        - crystal_system: str - One of 'cubic' or 'hexagonal'
        - lattice_params: tuple - (a, b, c) lattice parameters in Angstroms
        - space_group: int or None - International space group number (optional)
        - metadata: dict - Any additional information (composition, reference, etc.)
    """

    name: str
    lattice_params: tuple
    point_group: str
    space_group: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if len(self.lattice_params) != 3:
            raise ValueError(
                f"lattice_params must be (a, b, c), got {self.lattice_params}"
            )

        if any(p <= 0 for p in self.lattice_params):
            raise ValueError(
                f"All lattice parameters must be positive, got {self.lattice_params}"
            )

        if self.point_group not in VALID_POINT_GROUPS:
            raise ValueError(
                f"Unknown point group '{self.point_group}'. "
                f"Expected one of {VALID_POINT_GROUPS}"
            )

        if self.space_group is not None:
            if not (1 <= self.space_group <= 230):
                raise ValueError(
                    f"Space group must be between 1 and 230, got {self.space_group}"
                )

    @property
    def a(self) -> float:
        return self.lattice_params[0]

    @property
    def b(self) -> float:
        return self.lattice_params[1]

    @property
    def c(self) -> float:
        return self.lattice_params[2]

    @property
    def crystal_system(self) -> str:
        """Derive crystal system from point group."""
        cubic = {"m-3m", "m-3", "432", "-43m", "23"}
        hexagonal = {"6/mmm", "6mm", "-6m2", "622", "6/m", "-6", "6"}
        trigonal = {"3m", "-3m", "32", "-3", "3"}
        tetragonal = {"4/mmm", "4mm", "-42m", "422", "4/m", "-4", "4"}
        orthorhombic = {"mmm", "mm2", "222"}
        monoclinic = {"2/m", "m", "2"}
        triclinic = {"1", "-1"}

        if self.point_group in cubic:
            return "cubic"
        if self.point_group in hexagonal:
            return "hexagonal"
        if self.point_group in trigonal:
            return "trigonal"
        if self.point_group in tetragonal:
            return "tetragonal"
        if self.point_group in orthorhombic:
            return "orthorhombic"
        if self.point_group in monoclinic:
            return "monoclinic"
        if self.point_group in triclinic:
            return "triclinic"
        raise ValueError(
            f"Cannot derive crystal system from point group '{self.point_group}'"
        )

    @classmethod
    def from_preset(cls, name: str) -> "Phase":
        """
        Instantiate a Phase from a known material preset.

        Usage:
            phase = Phase.from_preset("Fe-fcc")
        """

        resolved = ALIASES.get(name, name)

        if resolved not in KNOWN_PHASES:
            available = list(KNOWN_PHASES.keys()) + list(ALIASES.keys())
            raise ValueError(
                f"Unknown preset '{name}'. " f"Available presets: {available}"
            )

        if name not in KNOWN_PHASES:
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available presets: {list(KNOWN_PHASES.keys())}"
            )

        data = KNOWN_PHASES[name]
        return cls(
            name=resolved,
            lattice_params=data["lattice_params"],
            space_group=data["space_group"],
            point_group=data["point_group"],
        )

    def __repr__(self):
        return (
            f"Phase(name='{self.name}', "
            f"crystal_system='{self.crystal_system}', "
            f"point_group='{self.point_group}' "
            f"lattice_params={self.lattice_params}, "
            f"space_group={self.space_group}), "
        )
