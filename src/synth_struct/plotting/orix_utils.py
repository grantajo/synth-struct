# synth_struct/src/synth_struct/plotting/orix_utils.py

"""
Utility functions for orix-based crystallographic visualization

Handles conversion between microstructure data and orix data structures.
"""

import numpy as np
from orix.crystal_map import CrystalMap, PhaseList, Phase
from orix.quaternion import Orientation

from ..orientation.rotation_converter import euler_to_quat


def create_crystal_map(micro, crystal_structure="cubic", grain_subset=None):
    """
    Convert microstructure class to orix CrystalMap.

    Converts synth_struct microstructures into orix CrystalMap objects for EBSD-style visualization

    Args:
    - micro: Microstructure object with orientations and grain_ids
    - crystal_structure: str - 'cubic', 'fcc', 'bcc', 'hexagonal', or 'hcp'.
    - grain_subset: np.ndarray or None - Array of grain IDs to include. If None, include all grains

    Returns:
    - CrystalMap: orix CrystalMap object

    Example:
        from plotting.orix_utils import create_crystal_map
        crystal_map = create_crystal_map(micro, crystal_structure='cubic')

        # Or for a specific region
        from micro_utils import get_grains_in_region
        region_grains = get_grains_in_region(micro, 'sphere', center=[50,50,50], radius=20)
        crystal_map = create_crystal_map(micro, grain_subset=region_grains)
    """

    if crystal_structure.lower() in ["cubic", "fcc", "bcc"]:
        phase = Phase(name="Cubic", point_group="m-3m")
    elif crystal_structure.lower() in ["hexagonal", "hcp"]:
        phase = Phase(name="Hexagonal", point_group="6/mmm")
    else:
        raise ValueError(
            f"Unkown crystal structure: {crystal_structure}. "
            f"Supported: 'cubic', 'fcc', 'bcc', 'hexagonal', 'hcp'"
        )

    phase_list = PhaseList(phase)
    symmetry = phase.point_group

    # Convert orientations to quaternions
    quaternions = euler_to_quat(micro.orientations)

    grain_ids_flat = micro.grain_ids.flatten()
    num_points = len(grain_ids_flat)

    # Initialize quaternion array
    quaternion_array = quaternions[grain_ids_flat]

    orientations = Orientation(quaternion_array, symmetry=symmetry)

    if len(micro.dimensions) == 3:
        nx, ny, nz = micro.dimensions

        x_coords = np.repeat(np.arange(nx), ny * nz)
        y_coords = np.tile(np.repeat(np.arange(ny), nz), nx)
        z_coords = np.tile(np.arange(nz), nx * ny)

        crystal_map = CrystalMap(
            rotations=orientations,
            phase_id=np.zeros(num_points, dtype=int),
            x=x_coords,
            y=y_coords,
            phase_list=phase_list,
            scan_unit="px",
            prop={"grain_id": grain_ids_flat, "z": z_coords},
        )

    else:  # 2D
        ny, nx = micro.dimensions

        y_coords = np.repeat(np.arange(ny), nx)
        x_coords = np.tile(np.arange(nx), ny)

        crystal_map = CrystalMap(
            rotations=orientations,
            phase_id=np.zeros(num_points, dtype=int),
            x=x_coords,
            y=y_coords,
            phase_list=phase_list,
            scan_unit="px",
            prop={"grain_id": grain_ids_flat},
        )

    return crystal_map


def get_crystal_map_slice(crystal_map, dimensions, slice_idx=None, slice_direction="z"):
    """
    Extract a 2D slice from a 3D crystal map.

    Args:
    - crystal_map: orix CrystalMap object
    - dimensions: tuple - Microstructure dimensions
    - slice_idx: int or None - Slice index. If None, uses middle slice.
    - slice_direction: str - 'x', 'y', or 'z'

    Returns:
    - crystal_map_slice: 2D slice of the crystal map
    - shape: tuple - Shape of the slice (for reshaping images)

    Example:
        crystal_map = create_crystal_map(micro)
        cm_slice, shape = get_crystal_map_slice(crystal_map, micro.dimensions,
                                                slice_idx=50, slice_direction='z')
    """

    if len(dimensions) != 3:
        return crystal_map, dimensions

    if slice_idx is None:
        axis_idx = {"x": 0, "y": 1, "z": 2}[slice_direction.lower()]
        slice_idx = dimensions[axis_idx] // 2

    # Extract slice based on direction
    if slice_direction.lower() == "z":
        crystal_map_slice = crystal_map[crystal_map.prop["z"] == slice_idx]
        shape = (dimensions[1], dimensions[0])
    elif slice_direction.lower() == "y":
        crystal_map_slice = crystal_map[crystal_map.y == slice_idx]
        shape = (dimensions[2], dimensions[0])
    elif slice_direction.lower() == "x":
        crystal_map_slice = crystal_map[crystal_map.x == slice_idx]
        shape = (dimensions[2], dimensions[1])
    else:
        raise ValueError(f"Invalid slice_direction: {slice_direction}")

    return crystal_map_slice, shape


def subsample_orientations(orientations, sample_fraction):
    """
    Randomly subsample orientations for cleaner pole figure visualization.

    Args:
    - orientations: orix Orientation object - Orientations to subsample
    - sample_fraction: float - Fraction of orientations to keep (0-1)

    Returns:
    - Orientation: Subsampled orientations

    Example:
        # Use in pole figure plotting to reduce point density
        orientations_subset = subsample_orientations(crystal_map.orientations, 0.1)
    """
    if not 0 < sample_fraction <= 1:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    n = orientations.size
    n_sample = int(sample_fraction * n)
    idx = np.random.choice(n, size=n_sample, replace=False)

    return orientations[idx]


def get_grain_average_orientations(micro, crystal_structure="cubic"):
    """
    Get one representative orientation per grain (useful for pole figures with large datasets).

    Args:
    - microstructure: Microstructure object
    - crystal_structure: str - Crystal structure type

    Returns:
    - Orientation: orix Orientation object with one orientation per grain

    Example:
        # For very dense pole figures, use grain averages instead of all voxels
        avg_oris = get_grain_average_orientations(micro)
    """

    # Get phase info
    if crystal_structure.lower() in ["cubic", "fcc", "bcc"]:
        phase = Phase(name="Cubic", point_group="m-3m")
    elif crystal_structure.lower() in ["hexagonal", "hcp"]:
        phase = Phase(name="Hexagonal", point_group="6/mmm")
    else:
        raise ValueError(f"Unknown crystal structure: {crystal_structure}")

    symmetry = phase.point_group

    # Get quaternions (already one per grain)
    quaternions = euler_to_quat(micro.orientations)

    # Convert to array
    num_grains = len(quaternions)
    quat_array = np.zeros((num_grains, 4))
    for grain_id, quat in quaternions.items():
        if grain_id > 0:
            quat_array[grain_id - 1] = quat

    return Orientation(quat_array, symmetry=symmetry)


def filter_crystal_map_by_grains(crystal_map, grain_ids):
    """
    Filter crystal map to only include specific grains.

    Args:
    - crystal_map: Full crystal map
    - grain_ids: array-like - Grain IDs to keep

    Returns:
    - CrystalMap: Filtered crystal map

    Example:
        from micro_utils import get_grains_in_region
        region_grains = get_grains_in_region(micro, 'box', x_min=0, x_max=50)
        crystal_map = create_crystal_map(micro)
        filtered_cm = filter_crystal_map_by_grains(crystal_map, region_grains)
    """
    grain_ids = np.asarray(grain_ids)
    mask = np.isin(crystal_map.prop["grain_id"], grain_ids)
    return crystal_map[mask]
