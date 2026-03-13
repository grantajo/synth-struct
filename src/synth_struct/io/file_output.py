# synth-struct/src/synth_struct/io/file_output.py

"""
This file has functions for outputting meshes and other files
that are useful for mechanical simulations.
"""

import os
from enum import Enum
from pathlib import Path
import warnings
import tempfile

import numpy as np
from scipy.spatial import cKDTree, Voronoi, ConvexHull
import pyvista as pv
import gmsh
import h5py


class MeshPath(Enum):
    """
    High-level mesh strategy.

    CONFORMING: Analytical grain boundaries -> Gmsh -> conformal Tet mesh

    REGULAR_GRID: Voxel grain_ids -> structured hex8 mesh
    """

    CONFORMING = "conforming"
    REGULAR_GRID = "regular_grid"


class SolverFormat(Enum):
    """
    Target solver output format.

    DAMASK_VTI      : ImageData .vti — DAMASK spectral solver
    DAMASK_HDF5     : Native DAMASK .hdf5 GeomFile
    ABAQUS_STANDARD : .inp — Abaqus Standard (elasticity, CPFEM)
    ABAQUS_EXPLICIT : .inp — Abaqus Explicit (wave propagation)
    FENICS          : .msh (Gmsh format v4) — FEniCS / DOLFINx
    CUSTOM_HDF5     : .h5 — your own CPFE solver
    VTK_SURFACE     : .vtp — intermediate surface, pass to mesher manually
    """

    DAMASK_VTI = "damask_vti"
    DAMASK_HDF5 = "damask_hdf5"
    ABAQUS_STANDARD = "abaqus_standard"
    ABAQUS_EXPLICIT = "abaqus_explicit"
    FENICS = "fenics"
    CUSTOM_HDF5 = "custom_hdf5"
    VTK_SURFACE = "vtk_surface"


class ElementType(Enum):
    """
    FEM element type.

    TET4: Linear tetrahedron - fast, less accurate, wave problems
    TET10: Quadratic tetrahedron — preferred for elasticity/CPFEM. Avoids volumetric locking, better stress recovery
    HEX8: Linear hexahedron — DAMASK spectral, wave problems - Less numerical dispersion than Tet for wave propagation
    """

    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"


_SOLVER_DEFAULTS = {
    SolverFormat.DAMASK_VTI: (MeshPath.REGULAR_GRID, ElementType.HEX8),
    SolverFormat.DAMASK_HDF5: (MeshPath.REGULAR_GRID, ElementType.HEX8),
    SolverFormat.ABAQUS_STANDARD: (MeshPath.CONFORMING, ElementType.TET10),
    SolverFormat.ABAQUS_EXPLICIT: (MeshPath.REGULAR_GRID, ElementType.HEX8),
    SolverFormat.FENICS: (MeshPath.CONFORMING, ElementType.TET10),
    SolverFormat.CUSTOM_HDF5: (MeshPath.REGULAR_GRID, ElementType.HEX8),
    SolverFormat.VTK_SURFACE: (MeshPath.CONFORMING, None),
}

_FORMAT_EXTENSIONS = {
    SolverFormat.DAMASK_VTI: ".vti",
    SolverFormat.DAMASK_HDF5: ".hdf5",
    SolverFormat.ABAQUS_STANDARD: ".inp",
    SolverFormat.ABAQUS_EXPLICIT: ".inp",
    SolverFormat.FENICS: ".msh",
    SolverFormat.CUSTOM_HDF5: ".h5",
    SolverFormat.VTK_SURFACE: ".vtp",
}

# -----------------------------------------------------------------------------
# Main Exporter
# -----------------------------------------------------------------------------


def export_microstructure(
    micro,
    generator,
    filepath,
    solver_format,
    element_type=None,  # None = use solver default
    mesh_path=None,  # None = use solver default
    k_neighbors=24,
    # Wave propagation
    target_frequency_hz=None,  # if set, validates element size
    wave_velocity=6000.0,  # m/s - default steel longitudinal
    # Periodic BCs
    periodic_bc=False,
    # Validation
    validate=False,
    sample_fraction=0.01,
):
    """
    Unified microstructure exporter for FEM solvers.

    Args:
        microstructure      : Microstructure instance
        generator           : VoronoiGenerator or AnisotropicVoronoiGenerator
        filepath            : str - output path (extension auto-corrected)
        solver_format       : SolverFormat enum
        element_type        : ElementType or None (uses solver default)
        mesh_path           : MeshPath or None (uses solver default)
        k_neighbours        : int - neighbourhood size for analytic Voronoi
        target_frequency_hz : float or None - NDE frequency for element
                              size validation
        wave_velocity       : float - wave speed in m/s for λ calculation
        periodic_bc         : bool - generate periodic node sets
        validate            : bool - validate mesh against voxel data
        sample_fraction     : float - voxel sample fraction for validation

    Returns:
        Exported mesh object (type depends on solver_format)
    """
    # --- Resolve defaults ---
    default_path, default_elem = _SOLVER_DEFAULTS[solver_format]
    mesh_path = mesh_path or default_path
    element_type = element_type or default_elem

    # --- Auto-correct extension ---
    filepath = str(Path(filepath).with_suffix(_FORMAT_EXTENSIONS[solver_format]))

    # --- Validate element size for wave problems ---
    if target_frequency_hz is not None:
        _validate_element_size(
            micro,
            target_frequency_hz,
            wave_velocity,
        )

    # --- Dispatch to mesh path ---
    if mesh_path == MeshPath.REGULAR_GRID:
        mesh = _build_regular_grid(micro)

    else:
        mesh = _build_conforming_mesh(
            micro,
            generator,
            k_neighbors,
            element_type,
            periodic_bc,
            solver_format,
            filepath,
        )

    _write_format(
        mesh,
        micro,
        filepath,
        solver_format,
        periodic_bc,
    )

    if validate:
        _validate_against_voxels(
            micro,
            mesh,
            sample_fraction,
        )

    return mesh


# -----------------------------------------------------------------------------
# Mesh Builders
# -----------------------------------------------------------------------------


def _build_regular_grid(micro):
    """Voxel grain_ids -> PyVista ImageData (Hex8 structured grid)."""
    dims = micro.dimensions
    res = micro.resolution

    grid = pv.ImageData()
    if len(dims) == 3:
        grid.dimensions = tuple(d + 1 for d in dims)
    else:
        grid.dimensions = (*dims, 1)

    grid.spacing = (res, res, res)
    grid.origin = (0.0, 0.0, 0.0)

    gids = micro.grain_ids.flatten(order="F")
    grid.cell_data["grain_id"] = gids

    if micro.phase_ids is not None:
        grid.cell_data["phase_id"] = micro.phase_ids.flatten(order="F")

    if micro.orientations is not None:
        grid.cell_data["orientation_euler"] = micro.orientations[gids]

    if hasattr(micro, "stiffness") and micro.stiffness is not None:
        # Flatten 6x6 -> 36 components per voxel for VTK compatibility
        stiffness_per_voxel = micro.stiffness[gids]  # (N, 6, 6)
        grid.cell_data["stiffness"] = stiffness_per_voxel.reshape(-1, 36)

    return grid


def _build_conforming_mesh(
    micro,
    generator,
    k_neighbors,
    element_type,
    periodic_bc,
    solver_format,
    filepath,
):
    """
    Analytic grain surfaces -> Gmsh -> conforming Tet mesh.
    Returns a PyVista UnstructuredGrid.
    """

    surface_meshes = _build_grain_surface_meshes(
        micro,
        generator,
        k_neighbors,
    )

    return _gmsh_conforming_tet(
        surface_meshes,
        micro,
        element_type,
        periodic_bc,
        solver_format,
        filepath,
    )


def _build_grain_surface_meshes(
    micro,
    generator,
    k_neighbors,
):
    """
    Build per-grain triangle surface meshes from analytical Voronoi.
    Returns list of (grain_id, pv.PolyData) tuples.
    """

    seeds = generator.seeds.astype(np.float64)
    scale_factors = generator.scale_factors.astype(np.float64)
    rotations = [R.astype(np.float64) for R in generator.rotations]
    dims = np.array(micro.dimensions)
    res = micro.resolution
    bounds = dims * res
    is_3d = len(dims) == 3
    n_grains = len(seeds)
    tree = cKDTree(seeds)
    meshes = []

    for gid in range(1, n_grains + 1):
        g = gid - 1
        R = rotations[g]
        scale = scale_factors[g]
        k = min(k_neighbors + 1, n_grains)
        _, nbr = tree.query(seeds[g], k=k)

        local_seeds = seeds[nbr]
        shifted = local_seeds - seeds[g]
        rotated = (R.T @ shifted.T).T
        scaled = rotated / scale
        mirrored = _mirror_seeds_local(scaled, k_neighbors)

        try:
            vor = Voronoi(mirrored)
            region_idx = vor.point_region[0]
            region = vor.regions[region_idx]
            if -1 in region or not region:
                continue

            verts_local = vor.vertices[region]
            verts_physical = (R @ (verts_local * scale).T).T + seeds[g]
            mesh = _convex_region_to_clipped_mesh(
                verts_physical,
                bounds,
                is_3d,
            )

            if mesh is not None:
                meshes.append((gid, mesh))

        except Exception:
            continue

    return meshes


def _mirror_seeds_local(
    seeds_local,
    k_neighbors,
):
    """
    Mirror seeds in local metric space across domain faces to
    ensure boundary grain cells are closed.

    Uses the actual seed extent as a proxy for domain size

    Args:
        seeds_local: (k, D) array of seeds in grain's local metric space
        k_neighbors: int - neighborhood size, used as a sanity reference
    """
    mirrored = [seeds_local]
    n_dim = seeds_local.shape[1]
    extent = np.max(np.abs(seeds_local), axis=0) * 2 + 1

    for ax in range(n_dim):
        low = seeds_local.copy()
        high = seeds_local.copy()
        low[:, ax] = -seeds_local[:, ax]
        high[:, ax] = 2 * extent[ax] - seeds_local[:, ax]
        mirrored.extend([low, high])

    return np.vstack(mirrored)


def _convex_region_to_clipped_mesh(
    verts,
    bounds,
    is_3d,
):
    """
    Build a convex hull surface mesh from Voronoi region vertices,
    then clip to the physical domain box.

    Args:
        verts: (V, D) array of vertices in physical space
        bounds: (D,) array of domain extents in physical units
        id_3d: bool
    """
    try:
        if len(verts) < (4 if is_3d else 3):
            return None

        hull = ConvexHull(verts)
        faces_pv = np.hstack(
            [np.full((len(hull.simplices), 1), 3), hull.simplices]
        ).astype(np.int32)

        mesh = pv.PolyData(verts, faces_pv)
        box = pv.Box(bounds=[0, bounds[0], 0, bounds[1], 0, bounds[2] if is_3d else 1])
        clipped = mesh.clip_box(box, invert=False)
        return clipped if clipped.n_cells > 0 else None

    except Exception:
        return None


def _gmsh_conforming_tet(
    surface_meshes,
    micro,
    element_type,
    periodic_bc,
    solver_format,
    filepath,
):
    """
    Feed grain surfaces into Gmsh to produce a conforming tet mesh with
    with grain physical groups
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)

    # Element order: Tet4 = 1, Tet10 = 2
    order = 2 if element_type == ElementType.TET10 else 1
    gmsh.option.setNumber("Mesh.ElementOrder", order)

    # Mesh size from microstructure resolution
    res = micro.resolution
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res * 2.0)

    for gid, poly in surface_meshes:
        verts = np.array(poly.points)
        faces = poly.faces.reshape(-1, 4)[:, 1:]  # strip leading '3'

        # Add vertices
        point_tags = []
        for v in verts:
            tag = gmsh.model.geo.addPoint(v[0], v[1], v[2], res)
            point_tags.append(tag)

        # Add triangular surfaces
        surface_tags = []
        for f in faces:
            line1 = gmsh.model.geo.addLine(point_tags[f[0]], point_tags[f[1]])
            line2 = gmsh.model.geo.addLine(point_tags[f[1]], point_tags[f[2]])
            line3 = gmsh.model.geo.addLine(point_tags[f[2]], point_tags[f[0]])
            loop = gmsh.model.geo.addCurveLoop([line1, line2, line3])
            surf = gmsh.model.geo.addPlaneSurface([loop])
            surface_tags.append(surf)

        # Volume from closed surface shell
        shell = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume = gmsh.model.geo.addVolume([shell])

        # Physical group per grain - carries grain_id into solver
        pg = gmsh.model.addPhysicalGroup(3, [volume], tag=gid)
        gmsh.model.setPhysicalName(3, pg, f"grain_{gid}")

    gmsh.model.geo.synchronize()

    # --- Periodic BCs (node-matching on opposing faces) ---
    if periodic_bc:
        _apply_gmsh_periodic_bc(micro)

    gmsh.model.mesh.generate(3)

    if solver_format == SolverFormat.FENICS:
        # Write .msh directly from Gmsh - PyVista cannot write .msh
        gmsh.write(filepath)
        gmsh.finalize()
        # Still return a PyVista mesh for consistency (read back in)
        mesh = pv.read(filepath)
    else:
        # All other formats: write to temp, load into PyVista, clean up
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            tmp = f.name
        gmsh.write(tmp)
        gmsh.finalize()
        mesh = pv.read(tmp)
        os.unlink(tmp)

    return mesh


def _apply_gmsh_periodic_bc(micro):
    """
    Apply periodic mesh constrains on opposing domain faces.
    """
    dims = np.array(micro.dimensions)
    res = micro.resolution
    bounds = dims * res

    # Translation vectors for each face pair
    face_pairs = [
        ([1, 0, 0], bounds[0]),
        ([0, 1, 0], bounds[1]),
        ([0, 0, 1], bounds[2]),
    ]

    for normal, length in face_pairs:
        translation = [
            1,
            0,
            0,
            normal[0] * length,
            0,
            1,
            0,
            normal[1] * length,
            0,
            0,
            1,
            normal[2] * length,
            0,
            0,
            0,
            1,
        ]
        # Get surfaces on each face via bounding box queries
        lo_surfs = gmsh.model.getEntitiesInBoundingBox(
            -res,
            -res,
            -res,
            bounds[0] * (1 - normal[0]) + res,
            bounds[1] * (1 - normal[1]) + res,
            bounds[2] * (1 - normal[2]) + res,
            dim=2,
        )
        hi_surfs = gmsh.model.getEntitiesInBoundingBox(
            normal[0] * bounds[0] - res,
            normal[1] * bounds[1] - res,
            normal[2] * bounds[2] - res,
            bounds[0] + res,
            bounds[1] + res,
            bounds[2] + res,
            dim=2,
        )
        for (_, lo), (_, hi) in zip(lo_surfs, hi_surfs):
            gmsh.model.mesh.setPeriodic(2, [hi], [lo], translation)


# -----------------------------------------------------------------------------
# Format Writeres
# -----------------------------------------------------------------------------


def _write_format(
    mesh,
    micro,
    filepath,
    solver_format,
    periodic_bc,
):
    """Dispatch to format-specific writer."""
    if solver_format == SolverFormat.DAMASK_VTI:
        mesh.save(filepath)

    elif solver_format == SolverFormat.DAMASK_HDF5:
        _write_damask_hdf5(mesh, micro, filepath)

    elif solver_format in (SolverFormat.ABAQUS_STANDARD, SolverFormat.ABAQUS_EXPLICIT):
        _write_abaqus_inp(
            mesh,
            micro,
            filepath,
            explicit=(solver_format == SolverFormat.ABAQUS_EXPLICIT),
            periodic_bc=periodic_bc,
        )

    elif solver_format == SolverFormat.FENICS:
        pass  # already written directly by Gmsh in _gmsh_conforming_tet

    elif solver_format == SolverFormat.CUSTOM_HDF5:
        _write_custom_hdf5(mesh, micro, filepath)

    elif solver_format == SolverFormat.VTK_SURFACE:
        mesh.save(filepath)


def _write_damask_hdf5(grid, micro, filepath):
    """
    Write DAMASK-compatible GeomFile in HDF5 format.
    DAMASK expects: cells, size, origin, material (= grain_id -1, 0-indexed)
    """
    dims = np.array(micro.dimensions)
    res = micro.resolution

    with h5py.File(filepath, "w") as f:
        f.attrs["creator"] = "microstructure_exporter"
        f.attrs["DADF5_version_major"] = 0
        f.attrs["DADF5_version_minor"] = 1

        geom = f.create_group("geometry")
        geom.create_dataset("cells", data=dims.astype(np.int32))
        geom.create_dataset("size", data=(dims * res).astype(np.float64))
        geom.create_dataset("origin", data=np.zeros(3, dtype=np.float64))

        # DAMASK material index is 0-based grain_id
        material = (micro.grain_ids - 1).astype(np.int32)
        geom.create_dataset("material", data=material)


def _write_abaqus_inp(
    mesh,
    micro,
    filepath,
    explicit,
    periodic_bc,
):
    """
    Write Abaqus .inp file with:
    - Node coordinates
    - Element connectivity with type declaration
    - Element sets per grain (for material assignment)
    - Node sets for boundary conditions
    - Material + selection blocks (placeholder - user fills constitutive)
    """
    if periodic_bc:
        warnings.warn(
            "Periodic BC node equations for Abaqus are not yet implemented. "
            "Node sets are written but *Equation blocks must be added manually.",
            UserWarning,
        )

    dims = np.array(micro.dimensions)
    res = micro.resolution
    nodes = mesh.points
    # Extract cell connectivity — varies by element type
    cells = mesh.cells_dict

    with open(filepath, "w") as f:
        f.write("*Heading\n")
        f.write(
            f" Microstructure export — {'Explicit' if explicit else 'Standard'}\n\n"
        )

        # --- Nodes ---
        f.write("*Node\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"  {i}, {x:.6f}, {y:.6f}, {z:.6f}\n")

        # --- Elements per type ---
        abaqus_type = "C3D8R" if explicit else "C3D10"  # Hex8 explicit, Tet10 standard
        f.write(f"\n*Element, type={abaqus_type}\n")
        elem_id = 1

        for cell_type, connectivity in cells.items():
            for conn in connectivity:
                abaqus_conn = " ".join(str(n + 1) for n in conn)
                f.write(f"  {elem_id}, {abaqus_conn}\n")
                elem_id += 1

        # --- Element sets per grain ---
        if "grain_id" in mesh.cell_data:
            grain_ids = mesh.cell_data["grain_id"]
            for gid in np.unique(grain_ids):
                elems = np.where(grain_ids == gid)[0] + 1
                f.write(f"\n*Elset, elset=Grain_{gid}\n")
                # Abaqus: 16 elements per line max
                for chunk in [elems[i : i + 16] for i in range(0, len(elems), 16)]:
                    f.write("  " + ", ".join(map(str, chunk)) + "\n")

        # --- Boundary node sets (domain faces) ---
        bounds = dims * res
        tol = res * 0.01
        face_defs = {
            "X_LO": lambda p: p[:, 0] < tol,
            "X_HI": lambda p: p[:, 0] > bounds[0] - tol,
            "Y_LO": lambda p: p[:, 1] < tol,
            "Y_HI": lambda p: p[:, 1] > bounds[1] - tol,
            "Z_LO": lambda p: p[:, 2] < tol,
            "Z_HI": lambda p: p[:, 2] > bounds[2] - tol,
        }
        for name, mask_fn in face_defs.items():
            node_ids = np.where(mask_fn(nodes))[0] + 1
            if len(node_ids):
                f.write(f"\n*Nset, nset={name}\n")
                for chunk in [
                    node_ids[i : i + 16] for i in range(0, len(node_ids), 16)
                ]:
                    f.write("  " + ", ".join(map(str, chunk)) + "\n")

        # --- Material placeholders (one per grain) ---
        unique_grains = np.unique(mesh.cell_data.get("grain_id", []))
        for gid in unique_grains:
            f.write(f"\n*Material, name=Grain_{gid}_Material\n")
            f.write("*Elastic\n")
            f.write("** TODO: insert E, nu or full anisotropic stiffness here\n")
            f.write(
                f"\n*Solid Section, elset=Grain_{gid}, material=Grain_{gid}_Material\n"
            )
            f.write(",\n")

        # --- Step block ---
        step_type = "*Dynamic, Explicit" if explicit else "*Static"
        f.write(f"\n*Step\n{step_type}\n")
        f.write("** TODO: define load steps, BCs, output requests\n")
        f.write("*End Step\n")


def _write_custom_hdf5(
    mesh,
    micro,
    filepath,
):
    """
    Write a self-describing HDF5 file.

    Structure:
        /mesh/nodes             (N_nodes, 3)        float64
        /mesh/elements          (N_elem, n)         int32       connectivity
        /mesh/grain_id          (N_elem,)           int32
        /grains/orientations    (N_grains, 3)       float64     Euler Angles
        /grains/stiffness       (N_grains, 6, 6)    float64
        /grains/phase_id        (N_grains,)         int32
        /metadata/resolution    scalar              float64
        /metadata/dimensions    (3,)                int32
        /metadata/units         str
    """
    with h5py.File(filepath, "w") as f:

        # --- Mesh topology ---
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("nodes", data=mesh.points.astype(np.float64))

        if hasattr(mesh, "cells_dict"):
            cells = mesh.cells_dict
            for cell_type, connectivity in cells.items():
                mesh_grp.create_dataset(
                    f"elements_{cell_type}", data=connectivity.astype(np.int32)
                )
        else:
            # ImageData - store grid parameters instead of connectivity
            # Connectivity is implicit from dimensions and spacing
            mesh_grp.create_dataset(
                "dimensions", data=np.array(mesh.dimensions, dtype=np.int32)
            )
            mesh_grp.create_dataset(
                "spacing", data=np.array(mesh.spacing, dtype=np.float64)
            )
            mesh_grp.create_dataset(
                "origin", data=np.array(mesh.origin, dtype=np.float64)
            )

        if "grain_id" in mesh.cell_data:
            mesh_grp.create_dataset(
                "grain_id", data=mesh.cell_data["grain_id"].astype(np.int32)
            )

        # --- Per-grain data ---
        grains_grp = f.create_group("grains")
        n = micro.num_grains + 1  # include background at 0

        if micro.orientations is not None:
            grains_grp.create_dataset(
                "orientations", data=micro.orientations[:n].astype(np.float64)
            )
            grains_grp["orientations"].attrs["convention"] = "Bunge ZXZ radians"

        if hasattr(micro, "stiffness") and micro.stiffness is not None:
            grains_grp.create_dataset(
                "stiffness", data=micro.stiffness[:n].astype(np.float64)
            )
            grains_grp["stiffness"].attrs["convention"] = "Voigt 6x6 GPa"

        if hasattr(micro, "phase") and micro.phase is not None:
            grains_grp.create_dataset("phase_id", data=micro.phase[:n].astype(np.int32))

        # --- Phase names ---
        phases_grp = f.create_group("phases")
        for pid, phase in micro.phases.items():
            pg = phases_grp.create_group(str(pid))
            pg.attrs["name"] = phase.name
            pg.attrs["space_group"] = phase.space_group

        # --- Metadata ---
        meta = f.create_group("metadata")
        meta.create_dataset(
            "dimensions", data=np.array(micro.dimensions, dtype=np.int32)
        )
        meta.create_dataset("resolution", data=micro.resolution)
        meta.attrs["units"] = micro.units


# -----------------------------------------------------------------------------
# Validation (wave propagation and voxels)
# -----------------------------------------------------------------------------


def _validate_element_size(
    micro,
    frequency_hz,
    wave_velocity_ms,
):
    """
    Check voxel resolution statisfies the λ/10 rule for wave propagation.
    Emits a warning if resolution is too coarse for the target frequency.
    """
    wavelength = wave_velocity_ms / frequency_hz  # Units (m)
    max_elem_size = (wavelength / 10) * 1e6  # Units (um)

    res = micro.resolution  # Units (um)

    freq_mhz = frequency_hz / 1e6
    print(
        f"  Wave check: f={freq_mhz:.1f} MHz, λ={wavelength*1e3:.2f} mm, "
        f"max element size={max_elem_size:.1f} μm, "
        f"current resolution={res:.1f} μm"
    )

    if res > max_elem_size:
        warnings.warn(
            f"Resolution {res:.1f} μm is too coarse for {freq_mhz:.1f} MHz "
            f"(need < {max_elem_size:.1f} μm). "
            f"Increase voxel resolution or reduce target frequency. ",
            UserWarning,
        )
    else:
        print(f"  Resolution is sufficient for {freq_mhz:.1f} MHz propagation.")


def _validate_against_voxels(
    micro,
    mesh,
    sample_fraction=0.01,
):
    """
    Spot-check that the exported mesh grain assignments match the
    voxel grain_ids for a random sample of voxel centers.

    Samples voxel center positions in physical space, queries their
    grain_id form the mesh cell data, and compares agenst the ground truth
    grain_ids array.

    Args:
        micro           : Microstructure instance
        mesh            : pv.DataSet — output of export_microstructure
        sample_fraction : float - fraction of voxels to sample (default 1%)

    Returns:
        float - match rate between 0.0 and 1.0
    """
    dims = micro.dimensions
    res = micro.resolution
    grain_ids = micro.grain_ids
    is_3d = len(dims) == 3

    if "grain_id" not in mesh.cell_data and "grain_id" not in mesh.point_data:
        warnings.warn("Mesh has no grain_id field - skipping validation.", UserWarning)
        return None

    # --- Random sample of voxel flat indices ---
    total = int(np.prod(dims))
    n = max(100, int(total * sample_fraction))
    indices = np.random.choice(total, size=n, replace=False)

    # Convert flat indices to voxel coordinates then to physical centers
    coords = np.column_stack(np.unravel_index(indices, dims)).astype(np.float64)
    points_physical = (coords + 0.5) * res  # Voxel centers in physical units

    # Extend to 3D if 2D - PyVista always works in 3D
    if not is_3d:
        points_physical = np.hstack(
            [points_physical, np.zeros((len(points_physical), 1))]
        )

    # --- Sample mesh grain_id at voxel center positions ---
    point_cloud = pv.PolyData(points_physical)

    try:
        sampled = point_cloud.sample(mesh)
    except Exception as e:
        warnings.warn(
            f"Mesh sampling failed during validation: {e}",
            UserWarning,
        )
        return None

    # sample() interpolates - for integer grain_ids we round back
    if "grain_id" in sampled.point_data:
        mesh_gids = np.round(sampled.point_data["grain_id"]).astype(np.int32)
    else:
        warnings.warn(
            "Sampled mesh has no grain_id point data - skipping validation. ",
            UserWarning,
        )
        return None

    # --- Compare against ground truth ---
    voxel_gids = grain_ids.flat[indices].astype(np.int32)
    mismatches = np.sum(mesh_gids != voxel_gids)
    match_rate = 1.0 - mismatches / n

    print(
        f"  Validation: {match_rate:.2%} match over {n} sampled voxels "
        f"({mismatches} mismatches)"
    )

    if match_rate < 0.95:
        warnings.warn(
            f"{mismatches} grain_id mismatches detected ({1 - match_rate:.2%} error). "
            f"For conforming meshes, try increasing k_neighbors. "
            f"For regular grids, this indicates a flattening order mismatch. ",
            UserWarning,
        )
    else:
        print("  Validation Passed.")

    return match_rate
