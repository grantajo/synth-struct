import sys
sys.path.insert(0, '../src')

import matplotlib
import matplotlib.pyplot as plt

from microstructure import Microstructure
from texture import Texture
from hdf5_writer import write_d3d, write_hdf5
import time

def main():
    print("="*50)
    print("3D Microstructure for DREAM.3D")
    print("="*50)
    
    total_start = time.time()
    
    # Create 3D microstructure
    print("\n[1/4] Creating 3D microstructure...")
    start = time.time()
    micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    print(f"      Done in {time.time() - start:.3f}s")
    
    # Generate grains using optimized method for 3D
    print("\n[2/4] Generating grains...")
    start = time.time()
    micro.gen_voronoi(num_grains=200, chunk_size=100000, seed=42)
    print(f"      Done in {time.time() - start:.3f}s")
    
    # Assign orientations
    print("\n[3/4] Assigning orientations...")
    start = time.time()
    micro.orientations = Texture.random_orientations(200, seed=42)
    print(f"      Done in {time.time() - start:.3f}s")
    
    # Save in DREAM.3D format
    print("\n[4/4] Saving to DREAM.3D format...")
    start = time.time()
    write_hdf5(micro, '../output/microstructure_3d.h5')
    # write_raw_binary(micro, '../output/microstructure')
    print(f"      Done in {time.time() - start:.3f}s")
    
    # Quick 2D slice visualization
    print("\nCreating 2D slice preview...")
    plt.figure(figsize=(8, 8))
    middle_slice = micro.grain_ids.shape[2] // 2
    plt.imshow(micro.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    plt.colorbar(label='Grain ID')
    plt.title(f'Z-slice at {middle_slice}')
    plt.savefig('../output/slice_preview.png', dpi=150)
    plt.close()
    
    total_time = time.time() - total_start
    print("\n" + "="*50)
    print(f"Total execution time: {total_time:.3f}s")
    print("="*50)
    print("\nOutput files:")
    print("  - ../output/microstructure.dream3d (open in DREAM.3D)")
    print("  - ../output/microstructure_grains.raw")
    print("  - ../output/slice_preview.png")

if __name__ == "__main__":
    main()
