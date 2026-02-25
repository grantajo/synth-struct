# synth-struct/src/synth_struct/plotting/orix_utils.py

"""
Utility functions for orix-based crystallographic visualization

Handles conversion between microstructure data and orix data structures.
"""

import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.crystal_map import Phase as OrixPhase
from orix.quaternion import Orientation

from synth_struct.orientation.rotation_converter import euler_to_quat


def create_crystal_map(micro, grain_subset=None):
    """
    Convert microstructure class to orix CrystalMap.

    Converts synth_struct microstructures into orix CrystalMap objects for EBSD-style visualization

    Args:
    - micro: Microstructure object with orientations and grain_ids
    - grain_subset: np.ndarray or None - Array of grain IDs to include. If None, include all grains

    Returns:
    - CrystalMap: orix CrystalMap object

    Example:
        from plotting.orix_utils import create_crystal_map
        crystal_map = create_crystal_map(micro)

        # Or for a specific region
        from micro_utils import get_grains_in_region
        region_grains = get_grains_in_region(micro, 'sphere', center=[50,50,50], radius=20)
        crystal_map = create_crystal_map(micro, grain_subset=region_grains)
    """

    if len(micro.phases) == 0:
        raise ValueError("Microstructure has no phase information.")

    orix_phases = {}
    for phase_id, phase_obj in micro.phases.items():
        if phase_obj.crystal_system.lower() == "cubic":
            point_group = "m-3m"
        elif phase_obj.crystal_system.lower() == "hexagonal":
            point_group = "6/mmm"
        else:
            raise ValueError(
                f"Unsupported crystal system '{phase_obj.crystal_system}' "
                f"for orix plotting"
            )
        orix_phases[phase_id] = OrixPhase(name=phase_obj.name, point_group=point_group)

    phase_list = PhaseList(phases=orix_phases)

    grain_ids = micro.grain_ids
    grain_ids_flat = grain_ids.flatten()

    if micro.phase_ids is not None:
        phase_ids_flat = micro.phase_ids.flatten().astype(np.int32)
    else:
        phase_ids_flat = np.zeros(grain_ids.size, dtype=np.int32)

    if grain_subset is not None:
        mask_flat = np.isin(grain_ids_flat, grain_subset)
    else:
        mask_flat = np.ones(len(grain_ids_flat), dtype=bool)

    masked_grain_ids = grain_ids_flat[mask_flat]

    quaternions_all = euler_to_quat(micro.orientations)
    quaternions_array = quaternions_all[masked_grain_ids]

    primary_symmetry = orix_phases[0].point_group
    orientations = Orientation(quaternions_array, symmetry=primary_symmetry)

    if len(micro.dimensions) == 3:
        nx, ny, nz = micro.dimensions

        x_coords = np.repeat(np.arange(nx), ny * nz)
        y_coords = np.tile(np.repeat(np.arange(ny), nz), nx)
        z_coords = np.tile(np.arange(nz), nx * ny)

        crystal_map = CrystalMap(
            rotations=orientations,
            phase_id=phase_ids_flat[mask_flat],
            x=x_coords[mask_flat],
            y=y_coords[mask_flat],
            phase_list=phase_list,
            scan_unit="px",
            prop={
                "grain_id": masked_grain_ids,
                "z": z_coords[mask_flat],
                "flat_idx": np.arange(len(grain_ids_flat))[mask_flat],
            },
        )

    else:  # 2D
        ny, nx = micro.dimensions

        y_coords = np.repeat(np.arange(ny), nx)
        x_coords = np.tile(np.arange(nx), ny)

        crystal_map = CrystalMap(
            rotations=orientations,
            phase_id=phase_ids_flat[mask_flat],
            x=x_coords[mask_flat],
            y=y_coords[mask_flat],
            phase_list=phase_list,
            scan_unit="px",
            prop={"grain_id": masked_grain_ids},
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

    rng = np.random.default_rng()
    n = orientations.size
    n_sample = int(sample_fraction * n)
    idx = rng.choice(n, size=n_sample, replace=False)

    return orientations[idx]


def get_grain_average_orientations(micro, crystal_structure="cubic"):
    """
    Get one representative orientation per grain (useful for pole figures with large datasets).

    Args:
    - micro: Microstructure object

    Returns:
    - Orientation: orix Orientation object with one orientation per grain

    Example:
        # For very dense pole figures, use grain averages instead of all voxels
        avg_oris = get_grain_average_orientations(micro)
    """

    if len(micro.phases) == 0:
        raise ValueError("Microstructure has no phase information")

    phase_obj = next(iter(micro.phases.values()))
    if phase_obj.crystal_system.lower() == "cubic":
        point_group = "m-3m"
    elif phase_obj.crystal_system.lower() == "hexagonal":
        point_group = "6/mmm"
    else:
        raise ValueError(f"Unsupported crystal system '{phase_obj.crystal_system}'")

    orix_phase = OrixPhase(name=phase_obj.name, point_group=point_group)
    symmetry = orix_phase.point_group

    quaternions = euler_to_quat(micro.orientations)
    grain_quaternions = quaternions[1:]  # Skip background

    return Orientation(grain_quaternions, symmetry=symmetry)


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
