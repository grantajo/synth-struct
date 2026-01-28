# synth_struct/examples/basic_3d_example.py

import time

import sys
sys.path.insert(0, '../src')

from microstructure import Microstructure
from texture import Texture
from hdf5_writer import write_struct_hdf5
import matplotlib.pyplot as plt

"""
This is a simple example to create a 3D microstructure.
"""

start_time = time.time()

dims = 200
res = 1.0 
num_grains = 350
# Create a 2D microstructure
micro = Microstructure(dimensions=(dims, dims, dims), resolution=res)

# Generate 50 grains
micro.gen_voronoi(num_grains=num_grains, seed=42)

# Assign random orientations
micro.orientations = Texture.random_orientations(num_grains, seed=42)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution Time: {elapsed_time:.2f} seconds")

fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
middle_slice = micro.grain_ids.shape[2] // 2
im = axs[0].imshow(micro.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
axs[0].set_title(f'Z-slice at {middle_slice}')

middle_slice = micro.grain_ids.shape[1] // 2
axs[1].imshow(micro.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
axs[1].set_title(f'Y-slice at {middle_slice}')

middle_slice = micro.grain_ids.shape[0] // 2
axs[2].imshow(micro.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
axs[2].set_title(f'X-slice at {middle_slice}')

fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
fig.suptitle('3D Example')
plt.savefig('../output/3d_slices.png', dpi=150)
plt.close()
