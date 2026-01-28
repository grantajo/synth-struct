import sys
import time
from pathlib import Path

from src.microstructure import Microstructure
from src.generators.voronoi import VoronoiGenerator

import matplotlib.pyplot as plt

start_time = time.time()

dims = 200
res = 1.0 
num_grains = 350
# Create a 2D microstructure
micro = Microstructure(dimensions=(dims, dims), resolution=res)

# Initialize Voronoi Generator
voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=None, chunk_size=500_000)

# Run Voronoi generator
voronoi_gen.generate(micro)

end_time = time.time()
elapsed_time = end_time - start_time

# print('Memory size:', sys.getsizeof(res), 'bytes')

print(f"Execution Time: {elapsed_time:.2f} seconds")


plt.figure(figsize=(6, 6))
plt.imshow(micro.grain_ids, cmap='nipy_spectral', origin='lower')
plt.colorbar(label='Grain ID')
plt.title('2D Example')

repo_root = Path(__file__).resolve().parent.parent
output_dir = repo_root / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '2d_slice.png', dpi=150)
plt.show()

