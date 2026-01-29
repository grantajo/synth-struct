# synth_struct/examples/shapes.py

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.microstructure import Microstructure
from src.generators.voronoi import VoronoiGenerator
from src.generators.ellipsoidal import EllipsoidalGenerator
from src.generators.columnar import ColumnarGenerator
from src.generators.mixed import MixedGenerator
from src.generators.lath import LathGenerator

import matplotlib.pyplot as plt
import numpy as np


"""
This example creates a microstructure with each of the various Microstructure generator classes

Generates both 2D and 3D examples for each type.
"""

def main():
    repo_root = project_root
    output_dir = repo_root / 'output'
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Microstructure generator examples")
    print("="*60)

    dims_2d = (200, 200)
    dims_3d = (100, 100, 100)
    resolution = 1.0
    num_grains = 300
    seed = 42

    micro_2d = Microstructure(dimensions=dims_2d, resolution=resolution)
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    
    # ===========================
    # Voronoi
    # ===========================
    print("\n1. Standard Voronoi:")
    print("-"*60)

    # 2D Voronoi
    gen = VoronoiGenerator(num_grains=num_grains, seed=seed)
    fig, elapsed = generate_and_visualize_2d(gen, micro_2d, "Voronoi", "voronoi_2d")
    fig.savefig(output_dir / 'voronoi_2d.png', dpi=100, bbox_inches='tight')
    print(f"    2D: {elapsed:.2f}s")

    # 3D Voronoi
    gen = VoronoiGenerator(num_grains=num_grains, seed=seed)
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Voronoi", "voronoi_3d")
    fig.savefig(output_dir / 'voronoi_3d.png', dpi=100, bbox_inches='tight')
    print(f"    3D: {elapsed:.2f}s")


    # ===========================
    # Ellipsoidal
    # ===========================
    print()
    print("="*60)
    print("2. Ellipsoidal:")
    print("-"*60)

    #2D Ellipsoidal
    gen = EllipsoidalGenerator(
        num_grains=num_grains,
        aspect_ratio_mean=4.0,
        aspect_ratio_std=0.8,
        orientation='z',
        base_size=8.0,
        seed=seed
    )
    fig, elapsed = generate_and_visualize_2d(gen, micro_2d, "Ellipsoidal", "ellipsoidal_2d")
    fig.savefig(output_dir / 'ellipsoidal_2d.png', dpi=150, bbox_inches='tight')
    print(f"    2D: {elapsed:.2f}s")

    gen = EllipsoidalGenerator(
        num_grains=num_grains,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=1.0,
        orientation='z',
        base_size=10.0,
        seed=seed,
        chunk_size=1_000_000
    )
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Ellipsoidal", "ellipsoidal_3d")
    fig.savefig(output_dir / 'ellipsoidal_3d.png', dpi=150, bbox_inches='tight')
    print(f"    3D: {elapsed:.2f}s")

    # ===========================
    # Columnar
    # ===========================
    print()
    print("="*60)
    print("3. Columnar:")
    print("-"*60)

    gen = ColumnarGenerator(
        num_grains=num_grains,
        axis='z',
        aspect_ratio=8.0,
        base_size=8.0,
        size_variation=0.2,
        seed=seed,
        chunk_size=1_000_000
    )
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Columnar (Z-axis)", "columnar_z")
    fig.savefig(output_dir / 'columnar_z.png', dpi=150, bbox_inches='tight')
    print(f"    Z-axis: {elapsed:.2f}s")

    gen = ColumnarGenerator(
        num_grains=num_grains,
        axis='x',
        aspect_ratio=8.0,
        base_size=8.0,
        seed=seed,
        chunk_size=1_000_000
    )
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Columnar (X-axis)", "columnar_x")
    fig.savefig(output_dir / 'columnar_x.png', dpi=150, bbox_inches='tight')
    print(f"    X-axis: {elapsed:.2f}s")
    
    # ===========================
    # Mixed
    # ===========================
    print()
    print("="*60)
    print("4. Mixed (ellipsoidal + equiaxed):")
    print("-"*60)

    gen = MixedGenerator(
        num_grains=num_grains,
        ellipsoid_fraction=0.6,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=1.0,
        base_size=10.0,
        seed=seed
    )
    fig, elapsed = generate_and_visualize_2d(gen, micro_2d, "Mixed (60% Ellipsoidal)", "mixed_2d")
    fig.savefig(output_dir / 'mixed_2d.png', dpi=150, bbox_inches='tight')
    print(f"    2D: {elapsed:.2f}s")

    # 3D Mixed
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    gen = MixedGenerator(
        num_grains=num_grains,
        ellipsoid_fraction=0.5,
        aspect_ratio_mean=6.0,
        base_size=10.0,
        seed=seed,
        chunk_size=1_000_000
    )
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Mixed (50% Ellipsoidal)", "mixed_3d")
    fig.savefig(output_dir / 'mixed_3d.png', dpi=150, bbox_inches='tight')
    print(f"    3D: {elapsed:.2f}s")
    
    # ===========================
    # Lath
    # ===========================
    print()
    print("="*60)
    print("5. Lath (Martensitic/Bainitic):")
    print("-"*60)

    gen = LathGenerator(
        num_grains=400,
        num_colonies=30,
        aspect_ratio_mean=12.0,
        aspect_ratio_std=2.0,
        width_mean=2.0,
        width_std=0.5,
        colony_misorientation=10.0,
        seed=seed,
        chunk_size=1_000_000
    )
    fig, elapsed = generate_and_visualize_3d(gen, micro_3d, "Lath (30 Colonies)", "lath_3d")
    fig.savefig(output_dir / 'lath_3d.png', dpi=150, bbox_inches='tight')
    print(f"    3D: {elapsed:.2f}s")
    
    # Get colony information
    colony_info = gen.get_colony_info()
    print(f"    Colonies: {colony_info['num_colonies']}")
    print(f"    Avg laths per colony: {len(colony_info['colony_ids']) / colony_info['num_colonies']:.1f}")
    
    print("\n" + "="*60)
    print(f"All examples saved to: {output_dir}")
    print("="*60)
    
    plt.show()


"""
Helper functions
"""
def generate_and_visualize_2d(generator, micro, title, filename):
    """Helper function to generate and visualize 2D microstructure"""
    start = time.time()
    generator.generate(micro)
    elapsed = time.time() - start
    
    plt.figure(figsize=(6,6))
    plt.imshow(micro.grain_ids, cmap='nipy_spectral', origin='lower')
    plt.colorbar(label='Grain ID')
    plt.title(f'{title} (2D)\n{micro.num_grains} grains')
    
    return plt.gcf(), elapsed
    
def generate_and_visualize_3d(generator, micro, title, filename):
    """Helper function to generate and visualize 3D microstructure"""
    start = time.time()
    generator.generate(micro)
    elapsed = time.time() - start
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    
    z_mid = micro.dimensions[2] // 2
    y_mid = micro.dimensions[1] // 2
    x_mid = micro.dimensions[0] // 2
    
    axes[0].imshow(micro.grain_ids[:, :, z_mid], cmap='nipy_spectral', origin='lower')
    axes[0].set_title(f'XY Slice')
    
    axes[1].imshow(micro.grain_ids[:, y_mid, :], cmap='nipy_spectral', origin='lower')
    axes[1].set_title(f'XZ Slice')
    
    axes[2].imshow(micro.grain_ids[x_mid, :, :], cmap='nipy_spectral', origin='lower')
    axes[2].set_title(f'YZ Slice')
    
    fig.suptitle(f'{title} (3D) - {micro.num_grains} grains', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig, elapsed
    
if __name__ == '__main__':
    main()

