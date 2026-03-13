# synth-struct/examples/vtk_example.py

"""
This is an example of how to create a VTK file from a microstructure.

This is similar to basic_example_3d
"""

import sys
from pathlib import Path

from synth_struct import Microstructure, VoronoiGenerator
from synth_struct import export_microstructure, SolverFormat

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

output_dir = project_root / "output" / "vtk_examples"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "microstructure"

print("=" * 20, "VTK Output Example", "=" * 20)

print("Generating 3D Microstructure")
# Variables for microstructure generation
dims = (100, 100, 100)
res = 1.0
num_grains = 500

micro = Microstructure(dimensions=dims, resolution=res)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

# ------------------------------------------------------
# 1. DAMASK VTI - Structured voxel grid
# ------------------------------------------------------
print("-" * 60)
print("[1/3]")
print("Exporting DAMASK VTI (.vti)...")

mesh_vti = export_microstructure(
    micro=micro,
    generator=voronoi_gen,
    filepath=str(output_path),
    solver_format=SolverFormat.DAMASK_VTI,
    validate=True,
    sample_fraction=0.01,
)
print("  Saved: {output_dir / 'microstructure.vti'}")

# ------------------------------------------------------
# 2. DAMASK HDF5 - native DAMASK GeomFile
# ------------------------------------------------------
print("-" * 60)
print("[2/3]")
print("Exporting DAMASK HDF5 (.hdf5)...")

mesh_hdf5 = export_microstructure(
    micro=micro,
    generator=voronoi_gen,
    filepath=str(output_path),
    solver_format=SolverFormat.DAMASK_HDF5,
    validate=True,
    sample_fraction=0.01,
)
print("  Saved: {output_dir / 'microstructure.hdf5'}")

# ------------------------------------------------------
# 3. Custom HDF5 - Self-describing HDF5 file
# ------------------------------------------------------
print("-" * 60)
print("[3/3]")
print("Exporting custom HDF5 (.h5)...")

mesh_h5 = export_microstructure(
    micro=micro,
    generator=voronoi_gen,
    filepath=str(output_path),
    solver_format=SolverFormat.CUSTOM_HDF5,
    validate=True,
    sample_fraction=0.01,
)
print(f"  Saved: {output_dir / 'microstructure.h5'}")

# Plotting
mesh_vti.plot(scalars="grain_id", cmap="hsv", show_edges=False)
#mesh_vtk.plot(scalars="grain_id", cmap="hsv", show_edges=False)
mesh_hdf5.plot(scalars="grain_id", cmap="hsv", show_edges=False)
mesh_h5.plot(scalars="grain_id", cmap="hsv", show_edges=False)