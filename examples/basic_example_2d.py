# synth_struct/examples/basic_2d_example.py

"""
This is a simple example to create a 2D microstructure.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.generators.voronoi import VoronoiGenerator


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

start_time = time.time()

dims = (200, 200)
res = 1.0
num_grains = 350
# Create a 2D microstructure
micro = Microstructure(dimensions=dims, resolution=res)

# Initialize Voronoi Generator
voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=None, chunk_size=500_000)

# Run Voronoi generator
voronoi_gen.generate(micro)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Created 2D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")
print(f"Execution Time: {elapsed_time:.2f} seconds")

# print('Memory size of voronoi_gen:', sys.getsizeof(voronoi_gen), 'bytes')
# print('Memory size of micro:', sys.getsizeof(micro.grain_ids), 'bytes')

plt.figure(figsize=(6, 6))
plt.imshow(micro.grain_ids, cmap="nipy_spectral", origin="lower")
plt.colorbar(label="Grain ID")
plt.title("2D Example")

repo_root = Path(__file__).resolve().parent.parent
output_dir = repo_root / "output"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "2d_slice.png", dpi=150)

print(f"Saved visualization to {output_dir / '2d_slices.png'}")

plt.show()
