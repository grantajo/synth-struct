import sys
sys.path.insert(0, '../src')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from microstructure import Microstructure
from texture import Texture


def main():
    # Example 1: Spherical grains with size variation
    micro_sph = Microstructure(dimensions=(100,100,100), resolution=1.0)
    micro_sph.gen_voronoi_w(
        num_grains=150, 
        grain_shapes='spherical',
        shape_params={'size_variation': 0.3},
        seed=42
    )

    # Example 2: Ellipsoidal grains
    micro_ell = Microstructure(dimensions=(1000, 1000, 1000), resolution=1.0)
    micro_ell.gen_voronoi_w(
        num_grains=1000,
        grain_shapes='ellipsoidal',
        shape_params={
            'aspect_ratio_mean': 4.0,
            'aspect_ratio_std': 0.25,
            'orientation': 'z'  # Elongated along Z
        },
        seed=42
    )

    # Example 3: Columnar grains (like solidification structure)
    micro_col = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro_col.gen_voronoi_w(
        num_grains=80,
        grain_shapes='columnar',
        shape_params={
            'axis': 'z',
            'aspect_ratio': 10.0
        },
        seed=42
    )


    # Example 4: Equiaxed grains (recrystallized structure)
    micro_eq = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro_eq.gen_voronoi_w(
        num_grains=200,
        grain_shapes='equiaxed',
        shape_params={'size_variation': 0.05},
        seed=42
    )

    # Example 5: Mixed morphology
    micro_mix = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro_mix.gen_voronoi_w(
        num_grains=120,
        grain_shapes='mixed',
        shape_params={
            'ellipsoid_fraction': 0.4,
            'aspect_ratio_mean': 2.0
        },
        seed=42
    )

    # Example 6: Custom weights for specific control
    custom_weights = np.random.gamma(2, 2, size=100)  # Gamma distribution
    micro_custom = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
    micro_custom.gen_voronoi_w(
        num_grains=100,
        grain_shapes='custom',
        shape_params={'weights': custom_weights},
        seed=42
    )

    # Quick 2D slice visualizations
    print("\nCreating 2D XY slice preview for spherical...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_sph.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_sph.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_sph.grain_ids.shape[1] // 2
    axs[1].imshow(micro_sph.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_sph.grain_ids.shape[0] // 2
    axs[2].imshow(micro_sph.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Spherical Microstructure')
    plt.savefig('../output/slice_preview_spherical.png', dpi=150)
    plt.close()
    
    
    print("\nCreating 2D XY slice preview for ellisoidal...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_ell.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_ell.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_ell.grain_ids.shape[1] // 2
    axs[1].imshow(micro_ell.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_ell.grain_ids.shape[0] // 2
    axs[2].imshow(micro_ell.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Ellipsoidal Microstructure')
    plt.savefig('../output/slice_preview_ellipsoidal.png', dpi=150)
    plt.close()
    
    
    print("\nCreating 2D XY slice preview for columnar...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_col.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_col.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_col.grain_ids.shape[1] // 2
    axs[1].imshow(micro_col.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_col.grain_ids.shape[0] // 2
    axs[2].imshow(micro_col.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Columnar Microstructure')
    plt.savefig('../output/slice_preview_columnar.png', dpi=150)
    plt.close()
    
    
    print("\nCreating 2D XY slice preview for equiaxed...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_eq.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_eq.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_eq.grain_ids.shape[1] // 2
    axs[1].imshow(micro_eq.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_eq.grain_ids.shape[0] // 2
    axs[2].imshow(micro_eq.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Equiaxed Microstructure')
    plt.savefig('../output/slice_preview_equiaxed.png', dpi=150)
    plt.close()
    
    
    print("\nCreating 2D XY slice preview for mixed...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_mix.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_mix.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_mix.grain_ids.shape[1] // 2
    axs[1].imshow(micro_mix.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_mix.grain_ids.shape[0] // 2
    axs[2].imshow(micro_mix.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Mixed Microstructure')
    plt.savefig('../output/slice_preview_mixed.png', dpi=150)
    plt.close()
    
    print("\nCreating 2D XY slice preview for custom...")
    fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
    middle_slice = micro_custom.grain_ids.shape[2] // 2
    im = axs[0].imshow(micro_custom.grain_ids[:, :, middle_slice], cmap='nipy_spectral')
    axs[0].set_title(f'Z-slice at {middle_slice}')
    
    middle_slice = micro_custom.grain_ids.shape[1] // 2
    axs[1].imshow(micro_custom.grain_ids[:, middle_slice, :], cmap='nipy_spectral')
    axs[1].set_title(f'Y-slice at {middle_slice}')
    
    middle_slice = micro_custom.grain_ids.shape[0] // 2
    axs[2].imshow(micro_custom.grain_ids[middle_slice, :, :], cmap='nipy_spectral')
    axs[2].set_title(f'X-slice at {middle_slice}')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', label='Grain ID', pad=0.08, aspect=40)
    fig.suptitle('Custom Microstructure')
    plt.savefig('../output/slice_preview_custom.png', dpi=150)
    plt.close()
       
    
    
if __name__ == "__main__":
    main()

