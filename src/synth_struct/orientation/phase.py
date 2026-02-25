# synth-struct/src/synth_struct/orientation/phase.py

"""
This class holds phase information for the microstructure.

Provides the ```Phase``` class, a container for crystal symmetry,
lattice parameters, and associated crystallographic information.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

VALID_CRYSTAL_SYSTEMS = (
    "cubic",
    "hexagonal",
    # "tetragonal", "orthorhombic", "trigonal", "monoclinic", "triclinic",
    # maybe implement other crystal systems in the future
)

KNOWN_PHASES = {
    "default": {
        "crystal_system": "cubic",
        "lattice_params": (1, 1, 1),
        "space_group": 225,
    },
    "Ti-alpha": {
        "crystal_system": "hexagonal",
        "lattice_params": (2.95, 2.95, 4.68),
        "space_group": 194,
    },
    "Ti-beta": {
        "crystal_system": "cubic",
        "lattice_params": (3.28, 3.28, 3.28),
        "space_group": 229,
    },
    "Fe-bcc": {
        "crystal_system": "cubic",
        "lattice_params": (2.87, 2.87, 2.87),
        "space_group": 229,
    },
    "Fe-fcc": {
        "crystal_system": "cubic",
        "lattice_params": (3.64, 3.64, 3.64),
        "space_group": 225,
    },
    "Al": {
        "crystal_system": "cubic",
        "lattice_params": (4.05, 4.05, 4.05),
        "space_group": 225,
    },
    "Cu": {
        "crystal_system": "cubic",
        "lattice_params": (3.61, 3.61, 3.61),
        "space_group": 225,
    },
    "Mg": {
        "crystal_system": "hexagonal",
        "lattice_params": (3.21, 3.21, 5.21),
        "space_group": 194,
    },
    "Ni": {
        "crystal_system": "cubic",
        "lattice_params": (3.52, 3.52, 3.52),
        "space_group": 225,
    },
}


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
    crystal_system: str
    lattice_params: tuple
    space_group: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.crystal_system not in VALID_CRYSTAL_SYSTEMS:
            raise ValueError(
                f"Unknown crystal system '{self.crystal_system}'. "
                f"Expected one of {VALID_CRYSTAL_SYSTEMS}"
            )

        if len(self.lattice_params) != 3:
            raise ValueError(
                f"lattice_params must be (a, b, c), got {self.lattice_params}"
            )

        if any(p <= 0 for p in self.lattice_params):
            raise ValueError(
                f"All lattice parameters must be positive, got {self.lattice_params}"
            )

        if self.crystal_system == "cubic":
            a, b, c = self.lattice_params
            if not (np.isclose(a, b) and np.isclose(b, c)):
                raise ValueError(
                    f"Cubic phase requires a == b == c, got {self.lattice_params}"
                )

        if self.crystal_system == "hexagonal":
            a, b, c = self.lattice_params
            if not np.isclose(a, b):
                raise ValueError(
                    f"Hexagonal phase requires a == b, got {self.lattice_params}"
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

    @classmethod
    def from_preset(cls, name: str) -> "Phase":
        """
        Instantiate a Phase from a known material preset.

        Usage:
            phase = Phase.from_preset("Fe-fcc")
        """
        if name not in KNOWN_PHASES:
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available presets: {list(KNOWN_PHASES.keys())}"
            )

        data = KNOWN_PHASES[name]
        return cls(
            name=name,
            crystal_system=data["crystal_system"],
            lattice_params=data["lattice_params"],
            space_group=data["space_group"],
        )

    def __repr__(self):
        return (
            f"Phase(name='{self.name}', "
            f"crystal_system='{self.crystal_system}', "
            f"lattice_params={self.lattice_params}, "
            f"space_group={self.space_group})"
        )
