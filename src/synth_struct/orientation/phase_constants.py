# synth-struct/src/synth_struct/orientation/phase.py

"""
A file that contains phase presets, aliases for phase presets, and valid point groups.
"""

VALID_POINT_GROUPS = {
    # Cubic
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
    # Hexagonal
    "6/mmm",
    "6mm",
    "-6m2",
    "622",
    "6/m",
    "-6",
    "6",
    # Trigonal
    "3m",
    "-3m",
    "32",
    "-3",
    "3",
    # Tetragonal
    "4/mmm",
    "4mm",
    "-42m",
    "422",
    "4/m",
    "-4",
    "4",
    # Orthorhombic
    "mmm",
    "mm2",
    "222",
    # Monoclinic
    "2/m",
    "m",
    "2",
    # Triclinic
    "-1",
    "1",
}

KNOWN_PHASES = {
    "default_cubic": {
        "lattice_params": (1, 1, 1),
        "space_group": 225,
        "point_group": "m-3m",
    },
    "default_hexagonal": {
        "lattice_params": (1, 1, 1.633),
        "space_group": 194,
        "point_group": "6/mmm",
    },
    "Ti-alpha": {
        "lattice_params": (2.95, 2.95, 4.68),
        "space_group": 194,
        "point_group": "6/mmm",
    },
    "Ti-beta": {
        "lattice_params": (3.28, 3.28, 3.28),
        "space_group": 229,
        "point_group": "m-3m",
    },
    "Fe-bcc": {
        "lattice_params": (2.87, 2.87, 2.87),
        "space_group": 229,
        "point_group": "m-3m",
    },
    "Fe-fcc": {
        "lattice_params": (3.64, 3.64, 3.64),
        "space_group": 225,
        "point_group": "m-3m",
    },
    "Al": {
        "lattice_params": (4.05, 4.05, 4.05),
        "space_group": 225,
        "point_group": "m-3m",
    },
    "Cu": {
        "lattice_params": (3.61, 3.61, 3.61),
        "space_group": 225,
        "point_group": "m-3m",
    },
    "Mg": {
        "lattice_params": (3.21, 3.21, 5.21),
        "space_group": 194,
        "point_group": "6/mmm",
    },
    "Ni": {
        "lattice_params": (3.52, 3.52, 3.52),
        "space_group": 225,
        "point_group": "m-3m",
    },
    "Zr-alpha": {
        "lattice_params": (3.23, 3.23, 5.15),
        "space_group": 194,
        "point_group": "6/mmm",
    },
    "Zr-beta": {
        "lattice_params": (3.62, 3.62, 3.62),
        "space_group": 229,
        "point_group": "m-3m",
    },
    "W": {
        "lattice_params": (3.16, 3.16, 3.16),
        "space_group": 229,
        "point_group": "m-3m",
    }
}

ALIASES = {
    "ferrite": "Fe-bcc",
    "austenite": "Fe-fcc",
    "alpha-Ti": "Ti-alpha",
    "beta-Ti": "Ti-beta",
}