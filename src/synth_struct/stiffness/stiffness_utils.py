# synth_struct/src/synth_struct/stiffness/stiffness_utils.py

"""
Utility functions for rotating stiffness tensors using orientation data.
"""

import numpy as np


def rotate_stiffness_tensor(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate a single stiffness tensor using a rotation matrix.

    Uses the 4th-order tensor rotation formula in Voigt notation:
    C'_ij = M_ik * M_jl * C_kl

    where M is the bond transformation matrix derived from rotation matrix R.

    Args:
    - C: Stiffness tensor in Voigt notation, shape (6, 6)
    - R: Rotation matrix, shape (3, 3)

    Returns:
    - C_rotated: Rotated stiffness tensor, shape (6, 6)
    """
    # Build the bond transformation matrix (Voigt notation)
    M = np.zeros((6, 6))

    # Direct terms (normal stresses)
    M[0, 0] = R[0, 0] ** 2
    M[0, 1] = R[0, 1] ** 2
    M[0, 2] = R[0, 2] ** 2
    M[0, 3] = 2 * R[0, 1] * R[0, 2]
    M[0, 4] = 2 * R[0, 0] * R[0, 2]
    M[0, 5] = 2 * R[0, 0] * R[0, 1]

    M[1, 0] = R[1, 0] ** 2
    M[1, 1] = R[1, 1] ** 2
    M[1, 2] = R[1, 2] ** 2
    M[1, 3] = 2 * R[1, 1] * R[1, 2]
    M[1, 4] = 2 * R[1, 0] * R[1, 2]
    M[1, 5] = 2 * R[1, 0] * R[1, 1]

    M[2, 0] = R[2, 0] ** 2
    M[2, 1] = R[2, 1] ** 2
    M[2, 2] = R[2, 2] ** 2
    M[2, 3] = 2 * R[2, 1] * R[2, 2]
    M[2, 4] = 2 * R[2, 0] * R[2, 2]
    M[2, 5] = 2 * R[2, 0] * R[2, 1]

    # Shear terms
    M[3, 0] = R[1, 0] * R[2, 0]
    M[3, 1] = R[1, 1] * R[2, 1]
    M[3, 2] = R[1, 2] * R[2, 2]
    M[3, 3] = R[1, 1] * R[2, 2] + R[1, 2] * R[2, 1]
    M[3, 4] = R[1, 0] * R[2, 2] + R[1, 2] * R[2, 0]
    M[3, 5] = R[1, 0] * R[2, 1] + R[1, 1] * R[2, 0]

    M[4, 0] = R[0, 0] * R[2, 0]
    M[4, 1] = R[0, 1] * R[2, 1]
    M[4, 2] = R[0, 2] * R[2, 2]
    M[4, 3] = R[0, 1] * R[2, 2] + R[0, 2] * R[2, 1]
    M[4, 4] = R[0, 0] * R[2, 2] + R[0, 2] * R[2, 0]
    M[4, 5] = R[0, 0] * R[2, 1] + R[0, 1] * R[2, 0]

    M[5, 0] = R[0, 0] * R[1, 0]
    M[5, 1] = R[0, 1] * R[1, 1]
    M[5, 2] = R[0, 2] * R[1, 2]
    M[5, 3] = R[0, 1] * R[1, 2] + R[0, 2] * R[1, 1]
    M[5, 4] = R[0, 0] * R[1, 2] + R[0, 2] * R[1, 0]
    M[5, 5] = R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]

    # Apply rotation: C' = M @ C @ M^T
    C_rotated = M @ C @ M.T

    return C_rotated


def rotate_stiffness_tensors_batch(
    C_base: np.ndarray, R_matrices: np.ndarray
) -> np.ndarray:
    """
    Rotate multiple stiffness tensors using corresponding rotation matrices.

    Args:
    - C_base: Base stiffness tensor(s), shape (6, 6) or (n, 6, 6)
              If (6, 6), the same base tensor is used for all rotations
    - R_matrices: Rotation matrices, shape (n, 3, 3)

    Returns:
    - C_rotated: Rotated stiffness tensors, shape (n, 6, 6)
    """
    n = R_matrices.shape[0]

    # Handle both single base tensor and multiple base tensors
    if C_base.ndim == 2:
        # Single base tensor - broadcast to all
        C_rotated = np.empty((n, 6, 6))
        for i in range(n):
            C_rotated[i] = rotate_stiffness_tensor(C_base, R_matrices[i])
    elif C_base.ndim == 3:
        # Multiple base tensors
        if C_base.shape[0] != n:
            raise ValueError(
                f"Number of base tensors ({C_base.shape[0]}) must match "
                f"number of rotation matrices ({n})"
            )
        C_rotated = np.empty((n, 6, 6))
        for i in range(n):
            C_rotated[i] = rotate_stiffness_tensor(C_base[i], R_matrices[i])
    else:
        raise ValueError(
            f"C_base must have shape (6, 6) or (n, 6, 6), got {C_base.shape}"
        )

    return C_rotated
