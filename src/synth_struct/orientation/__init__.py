# synth-struct/src/synth_struct/orientation/__init__.py

"""
Rotation conversion functions
"""

from .rotation_converter import (
    normalize_angle,
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat,
    create_rotation_matrix_2d,
    rotation_z_to_x,
    rotation_z_to_y,
)
from .phase import Phase, available_presets
from .phase_constants import KNOWN_PHASES, VALID_POINT_GROUPS, ALIASES

__all__ = [
    "normalize_angle",
    "euler_to_quat",
    "euler_to_rotation_matrix",
    "quat_to_euler",
    "quat_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_quat",
    "create_rotation_matrix_2d",
    "rotation_z_to_x",
    "rotation_z_to_y",
    "Phase",
    "available_presets",
    "KNOWN_PHASES",
    "VALID_POINT_GROUPS",
    "ALIASES",
]
