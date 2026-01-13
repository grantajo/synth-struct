import sys
sys.path.insert(0, '../src')

from microstructure import Microstructure
from texture import Texture
from hdf5_writer import write_struct_hdf5
import matplotlib.pyplot as plt

dims = 200
res = 1.0 
num_grains = 50
# Create a 2D microstructure
micro = Microstructure(dimensions=(dims, dims), resolution=res)

# Generate 50 grains
micro.generate_voronoi_grains(num_grains=num_grains, seed=42)

# Assign random orientations
micro.orientations = Texture.random_orientations(num_grains, seed=42)

plt.figure(figsize=(8, 8))
plt.imshow(micro.grain_ids, cmap='tab20')
plt.colorbar(label='Grain ID')
plt.title('Generated Microstructure')
plt.savefig('../output/microstructure.png', dpi=150)
plt.show()

write_microstructure_hdf5(micro, '../output/microstructure.h5')
print('Microstructure saved to output/microstructure.h5')
