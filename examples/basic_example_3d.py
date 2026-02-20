# synth-struct/examples/basic_example_3d.py

"""
This is a simple example to create a 3D microstructure and visualize slices.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.generators.voronoi import VoronoiGenerator

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=" * 16, "Basic 3D Example", "=" * 16)

start_time = time.time()

dims = (100, 100, 100)
res = 1.0
num_grains = 500

# Create a 2D microstructure
micro = Microstructure(dimensions=dims, resolution=res)

# Initialize Voronoi Generator
voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")
print(f"Execution Time: {elapsed_time:.2f} seconds")

# print('Memory size of voronoi_gen:', sys.getsizeof(voronoi_gen), 'bytes')
# print('Memory size of micro:', sys.getsizeof(micro.grain_ids), 'bytes')

# Visualize the three orthogonal slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# XY slice (middle of Z)
z_slice = dims[2] // 2
axes[0].imshow(micro.grain_ids[:, :, z_slice], cmap="nipy_spectral", origin="lower")
axes[0].set_title(f"XY Slice (z={z_slice})")

# XZ slice (middle of y)
y_slice = dims[1] // 2
axes[1].imshow(micro.grain_ids[:, y_slice, :], cmap="nipy_spectral", origin="lower")
axes[1].set_title(f"XZ Slice (y={y_slice})")

# YZ slice (middle of x)
x_slice = dims[0] // 2
axes[2].imshow(micro.grain_ids[x_slice, :, :], cmap="nipy_spectral", origin="lower")
axes[2].set_title(f"YZ Slice (x={x_slice})")

plt.tight_layout()

repo_root = Path(__file__).resolve().parent.parent
output_dir = repo_root / "output/basic_examples"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "3d_slices.png", dpi=150)

print("-" * 50)
print(f"Saved visualization to:\n{output_dir / '3d_slices.png'}")
print("-" * 50)

# plt.show()
