import sys
sys.path.insert(0, '../src')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from microstructure import Microstructure
from texture import Texture


def main():
    micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro.gen_voronoi(num_grains=75, seed=42)
    
    print("\nCreating 2D XY slice preview for equiaxed (unweighted Voronoi)...")
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
    fig.suptitle('Equiaxed Microstructure')
    plt.savefig('../output/slice_preview_equiaxed.png', dpi=150)
    plt.close()
    

    # Example 2: Ellipsoidal grains
    micro = Microstructure(dimensions=(100, 100, 200), resolution=1.0)
    micro.gen_voronoi_w(
        num_grains=50,
        grain_shapes='ellipsoidal',
        shape_params={
            'aspect_ratio_mean': 5.0,
            'aspect_ratio_std': 1.0,
            'orientation': 'z',
            'base_size': 8.0
        },
        seed=42
    )
    
    print("\nCreating 2D XY slice preview for ellisoidal...")
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
    fig.suptitle('Ellipsoidal Microstructure')
    plt.savefig('../output/slice_preview_ellipsoidal.png', dpi=150)
    plt.close()
    

    # Example 3: Columnar grains (like solidification structure)
    micro = Microstructure(dimensions=(100, 100, 200), resolution=1.0)
    micro.gen_voronoi_w(
        num_grains=50,
        grain_shapes='columnar',
        shape_params={
            'axis': 'z',
            'aspect_ratio': 10.0,
            'base_size': 8.0
        },
        seed=42
    )
    
    print("\nCreating 2D XY slice preview for columnar...")
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
    fig.suptitle('Columnar Microstructure')
    plt.savefig('../output/slice_preview_columnar.png', dpi=150)
    plt.close()


    # Example 5: Mixed morphology
    micro = Microstructure(dimensions=(100, 100, 200), resolution=1.0)
    micro.gen_voronoi_w(
        num_grains=120,
        grain_shapes='mixed',
        shape_params={
            'ellipsoid_fraction': 0.4,
            'aspect_ratio_mean': 4.0,
            'base_size': 8.0
        },
        seed=42
    )
    
    print("\nCreating 2D XY slice preview for mixed...")
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
    fig.suptitle('Mixed Microstructure')
    plt.savefig('../output/slice_preview_mixed.png', dpi=150)
    plt.close()
    
    """
    This is an artifact from a previous time of creating custom weights for the Voronoi generator
    May be brought back eventually
    
    # Example 6: Custom weights for specific control
    custom_weights = np.random.gamma(2, 2, size=100)  # Gamma distribution
    micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro.gen_voronoi_w(
        num_grains=100,
        grain_shapes='custom',
        shape_params={'weights': custom_weights},
        seed=42
    )
    
    print("\nCreating 2D XY slice preview for custom...")
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
    fig.suptitle('Custom Microstructure')
    plt.savefig('../output/slice_preview_custom.png', dpi=150)
    plt.close()
    """
    
    
if __name__ == "__main__":
    main()

