# synth_struct/src/synth_struct/micro_utils.py

"""
A series of functions to help with microstructure development

1. Get grain IDs from a region of a Microstructure
"""

from typing import Optional, List, Union
import numpy as np


def get_grains_in_region(micro, region_type: str = "box", **kwargs) -> np.ndarray:
    """
    Get grain IDs of grains that are in a specific region

    Args:
    - micro: Microstructure instance
    - region: str - Type of region ('box', 'sphere', 'cylinder', 'custom_mask')
    - **kwargs: Region-specific parameters

    Region types and parameters:
    - 'box':
        - x_min, x_max: X bounds (default: 0, dimensions[0])
        - y_min, y_max: Y bounds (default: 0, dimensions[1])
        - z_min, z_max: Z bounds (default: 0, dimensions[2], 3D only)
    - 'sphere':
        - center: [x, y, z] or [x, y] center coordinates (default: center of microstructure)
        - radius: Radius of sphere (required)
    - 'cylinder': 3D only
        - center: [x, y] center in plane perpendicular to axis (default: center)
        - radius: Radius of cylinder (required)
        - c_min, c_max: Bounds along cylinder axis (default: 0, dimensions[axis])
        - axis: Cylinder axis ('x', 'y', or 'z', default: 'z')
    - 'custom_mask':
        - mask: Boolean array same shape as microstructure (required)

    Returns:
    - grain_ids: np.ndarray - Array of grain IDs in the region (excluding background 0)

    Examples:
    # Box region
    grain = get_grains_in_region(micro, 'box', x_min=10, x_max=50, y_min=20, y_max=80)

    # Sphere in center
    grains = get_grains_in_region(micro, 'sphere', center=[50, 50, 50], radius=30)

    # Cylinder along Z-axis
    grains = get_grains_in_region(micro, 'cylinder', center=[50, 50], radius=20)

    # Custom mask
    mask = my_custom_function(micro.grain_ids)
    grains = get_grains_in_region(micro, 'custom_mask', mask=mask)
    """
    region_type = region_type.lower()

    if region_type == "box":
        mask = _create_box_mask(micro, **kwargs)

    elif region_type == "sphere":
        mask = _create_sphere_mask(micro, **kwargs)

    elif region_type == "cylinder":
        mask = _create_cylinder_mask(micro, **kwargs)

    elif region_type == "custom_mask":
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("'mask' parameter is required for custom_mask region type")
        if mask.shape != micro.grain_ids.shape:
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match microstructure "
                f"shape {micro.grain_ids.shape}"
            )

    else:
        raise ValueError(
            f"Unknown region_type: '{region_type}'. "
            f"Available types: 'box', 'sphere', 'cylinder', 'custom_mask'"
        )

    # Get unique grain IDs in the masked region
    grains_in_region = np.unique(micro.grain_ids[mask])

    # Remove background
    grains_in_region = grains_in_region[grains_in_region > 0]

    return grains_in_region


def _create_box_mask(
    micro,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_min: Optional[int] = None,
    y_max: Optional[int] = None,
    z_min: Optional[int] = None,
    z_max: Optional[int] = None,
) -> np.ndarray:
    """
    Create a box-shaped boolean mask.

    Args:
    - micro: Microstructure instance
    - x_min, x_max, y_min, y_max, z_min, z_max: Box bounds

    Returns:
    - mask: Boolean array matching microstructure dimensions
    """

    ndim = len(micro.dimensions)

    if ndim == 3:
        nx, ny, nz = micro.dimensions
        x_min = x_min if x_min is not None else 0
        x_max = x_max if x_max is not None else nx
        y_min = y_min if y_min is not None else 0
        y_max = y_max if y_max is not None else ny
        z_min = z_min if z_min is not None else 0
        z_max = z_max if z_max is not None else nz

        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        mask = (
            (x >= x_min)
            & (x < x_max)
            & (y >= y_min)
            & (y < y_max)
            & (z >= z_min)
            & (z < z_max)
        )

    elif ndim == 2:
        nx, ny = micro.dimensions
        x_min = x_min if x_min is not None else 0
        x_max = x_max if x_max is not None else nx
        y_min = y_min if y_min is not None else 0
        y_max = y_max if y_max is not None else ny

        x, y = np.mgrid[0:nx, 0:ny]
        mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)

    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")

    return mask


def _create_sphere_mask(
    micro,
    center: Optional[Union[List, np.ndarray]] = None,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Create a spherical (or circular in 2D) boolean mask.

    Args:
    - micro: Microstructure instance
    - center: Center coordinates [x, y] or [x, y, z]
    - radius: Sphere/circle radius

    Returns:
    - mask: Boolean array matching microstructure dimensions
    """

    if radius is None:
        raise ValueError("'radius' parameter is required for sphere region type")

    ndim = len(micro.dimensions)

    if ndim == 3:
        nx, ny, nz = micro.dimensions
        if center is None:
            center = np.array([nx / 2, ny / 2, nz / 2])
        else:
            center = np.array(center)

        if len(center) != 3:
            raise ValueError(
                f"Center for 3D sphere must have 3 coordinates, got {len(center)}"
            )

        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        distances = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        mask = distances <= radius

    elif ndim == 2:
        nx, ny = micro.dimensions
        if center is None:
            center = np.array([nx / 2, ny / 2])
        else:
            center = np.array(center)

        if len(center) != 2:
            raise ValueError(
                f"Center for 2D circle must have 2 coordinates, got {len(center)}"
            )

        x, y = np.mgrid[0:nx, 0:ny]
        distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = distances <= radius

    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")

    return mask


def _create_cylinder_mask(
    micro,
    center: Optional[Union[List, np.ndarray]] = None,
    radius: Optional[float] = None,
    z_min: Optional[int] = None,
    z_max: Optional[int] = None,
    axis: str = "z",
) -> np.ndarray:
    """
    Create a cylindrical boolean mask.

    Args:
    - micro: Microstructure instance
    - center: Center coordinates in plane perpendicular to axis
    - radius: Cylinder radius
    - z_min, z_max: Bounds along cylinder axis
    - axis: Cylinder axis ('x', 'y', or 'z', default: 'z')

    Returns:
    - mask: Boolean array matching microstructure dimensions
    """
    if axis is None:
        axis = "z"
    else:
        axis = axis.lower()

    if radius is None:
        raise ValueError("'radius' parameter is required for cylinder region type")

    if len(micro.dimensions) != 3:
        raise ValueError("Cylinder region only supported for 3D microstructures")

    nx, ny, nz = micro.dimensions
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]

    if axis == "z":
        # Cylinder along z-axis
        if center is None:
            center = np.array([nx / 2, ny / 2])
        else:
            center = np.array(center)

        if len(center) != 2:
            raise ValueError(
                f"Center for z-axis cylinder must have 2 coordinates (x, y), got {len(center)}"
            )

        z_min = z_min if z_min is not None else 0
        z_max = z_max if z_max is not None else nz

        radial_dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = (radial_dist <= radius) & (z >= z_min) & (z < z_max)

    elif axis == "y":
        # Cylinder along y-axis
        if center is None:
            center = np.array([nx / 2, nz / 2])
        else:
            center = np.array(center)

        if len(center) != 2:
            raise ValueError(
                f"Center for y-axis cylinder must have 2 coordinates (x, z), got {len(center)}"
            )

        y_min = z_min if z_min is not None else 0
        y_max = z_max if z_max is not None else ny

        radial_dist = np.sqrt((x - center[0]) ** 2 + (z - center[1]) ** 2)
        mask = (radial_dist <= radius) & (y >= y_min) & (y < y_max)

    elif axis == "x":
        # Cylinder along x-axis
        if center is None:
            center = np.array([ny / 2, nz / 2])
        else:
            center = np.array(center)

        if len(center) != 2:
            raise ValueError(
                f"Center for x-axis cylinder must have 2 coordinates (y, z), got {len(center)}"
            )

        x_min = z_min if z_min is not None else 0
        x_max = z_max if z_max is not None else nx

        radial_dist = np.sqrt((y - center[0]) ** 2 + (z - center[1]) ** 2)
        mask = (radial_dist <= radius) & (x >= x_min) & (x < x_max)

    else:
        raise ValueError(f"Axis must be 'x', 'y', or 'z', got '{axis}'")

    return mask
