# synth-struct/examples/aniso_mesh_output.py

"""
This is an example of how to create a VTK file from a microstructure
generator that is not 'Voronoi' and uses the anisotropic Voronoi
assignments.
"""

import sys
from pathlib import Path

from synth_struct import Microstructure, EllipsoidalGenerator, MixedGenerator
from synth_struct import export_microstructure, SolverFormat

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

output_dir = project_root / "output" / "vtk_examples"
output_dir.mkdir(parents=True, exist_ok=True)
ell_path = output_dir / "ellipsoidal_microstructure"
mix_path = output_dir / "mixed_microstructure"

print("=" * 17, "Aniso VTK Output Example", "=" * 17)
print("=" * 60)

print("Generating Ellipsoidal 3D Microstructure")
# Variables for microstructure generation
dims = (100, 100, 100)
res = 1.0
num_grains = 500

micro_ell = Microstructure(dimensions=dims, resolution=res)
ell_gen = EllipsoidalGenerator(
    num_grains=num_grains,
    aspect_ratio_mean=4.0,
    aspect_ratio_std=0.8,
    orientation="z",
    base_size=8.0,
    seed=42,
)
ell_gen.generate(micro_ell)

# ----------------------------------------
# Ellipsoidal Mesh Generation
# ----------------------------------------
print("Exporting ellipsoidal mesh files")
print("-" * 60)
print("[1/6]")
print("Exporting DAMASK VTI (.vti)...")

mesh1 = export_microstructure(
    micro=micro_ell,
    filepath=str(ell_path),
    solver_format=SolverFormat.DAMASK_VTI,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(ell_path) + '.vti'}")

"""
# This makes a really large file, only enable if you are okay with that
print("-" * 60)
print("[2/6]")
print("Exporting ABAQUS Explicit (.inp)...")

mesh2 = export_microstructure(
    micro=micro_ell,
    filepath=str(ell_path),
    solver_format=SolverFormat.ABAQUS_EXPLICIT,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(ell_path) + '.inp'}")
"""

print("-" * 60)
print("[3/6]")
print("Exporting DAMASK HDF5 (.hdf5)...")

mesh3 = export_microstructure(
    micro=micro_ell,
    filepath=str(ell_path),
    solver_format=SolverFormat.DAMASK_HDF5,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(ell_path) + '.hdf5'}")

# ----------------------------------------
# Mixed Mesh Generation
# ----------------------------------------
print("Generating Mixed 3D Microstructure")
micro_mix = Microstructure(dimensions=dims, resolution=res)
mix_gen = MixedGenerator(
    num_grains=num_grains,
    ellipsoid_fraction=0.5,
    aspect_ratio_mean=5.0,
    aspect_ratio_std=0.5,
    base_size=10.0,
    seed=42,
)
mix_gen.generate(micro_mix)


print("Exporting Mixed Microstructure mesh files")
print("-" * 60)
print("[4/6]")
print("Exporting DAMASK VTI (.vti)...")

mesh4 = export_microstructure(
    micro=micro_mix,
    filepath=str(mix_path),
    solver_format=SolverFormat.DAMASK_VTI,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(mix_path) + '.vti'}")

"""
# This makes a really large file, only enable if you are okay with that
print("-" * 60)
print("[5/6]")
print("Exporting ABAQUS Explicit (.inp)...")

mesh5 = export_microstructure(
    micro=micro_mix,
    filepath=str(mix_path),
    solver_format=SolverFormat.ABAQUS_EXPLICIT,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(mix_path) + '.inp'}")
"""

print("-" * 60)
print("[6/6]")
print("Exporting DAMASK HDF5 (.hdf5)...")

mesh6 = export_microstructure(
    micro=micro_mix,
    filepath=str(mix_path),
    solver_format=SolverFormat.DAMASK_HDF5,
    validate=True,
    sample_fraction=0.1,
)
print(f"  Saved: {str(mix_path) + '.hdf5'}")



