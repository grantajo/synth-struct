# synth_struct/examples/shapes.py

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from synth_struct.microstructure import Microstructure
from synth_struct.generators.voronoi import VoronoiGenerator
from synth_struct.generators.ellipsoidal import EllipsoidalGenerator
from synth_struct.generators.columnar import ColumnarGenerator
from synth_struct.generators.mixed import MixedGenerator
from synth_struct.generators.lath import LathGenerator
from synth_struct.plotting.gen_plot import Plotter

import matplotlib.pyplot as plt

"""
This example creates a microstructure with each of the various Microstructure generator classes

Generates both 2D and 3D examples for each type.
"""


def main():
    repo_root = project_root
    output_dir = repo_root / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Microstructure generator examples")
    print("=" * 60)

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
    print("-" * 60)
    print("Starting Standard Voronoi 2D")

    # 2D Voronoi
    gen = VoronoiGenerator(num_grains=num_grains, seed=seed)
    gen.generate(micro_2d)
    fig, ax = plt.subplots()
    Plotter.plot_grain_ids(ax, micro_2d)
    fig.savefig(output_dir / "voronoi_2d.png", dpi=100, bbox_inches="tight")
    print(f"    Voronoi 2D saved to: {output_dir / 'voronoi_2d.png'}")

    # 3D Voronoi
    print()
    print("Starting Standard Voronoi 3D")
    gen = VoronoiGenerator(num_grains=num_grains, seed=seed)
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "voronoi_3d.png", dpi=100, bbox_inches="tight")
    print(f"    Voronoi 3D saved to: {output_dir / 'voronoi_3d.png'}")

    # ===========================
    # Ellipsoidal
    # ===========================
    micro_2d = Microstructure(dimensions=dims_2d, resolution=resolution)
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    print()
    print("=" * 60)
    print("2. Ellipsoidal:")
    print("-" * 60)
    print("Starting Ellipsoidal 2D")

    # 2D Ellipsoidal
    gen = EllipsoidalGenerator(
        num_grains=num_grains,
        aspect_ratio_mean=4.0,
        aspect_ratio_std=0.8,
        orientation="z",
        base_size=8.0,
        seed=seed,
    )
    gen.generate(micro_2d)
    fig, ax = plt.subplots()
    Plotter.plot_grain_ids(ax, micro_2d)
    fig.savefig(output_dir / "ellipsoidal_2d.png", dpi=150, bbox_inches="tight")
    print(f"    Ellipsoidal 2D saved to: {output_dir / 'ellipsoidal_2d.png'}")

    print()
    print("Starting Ellipsoidal 3D")
    gen = EllipsoidalGenerator(
        num_grains=num_grains,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=1.0,
        orientation="z",
        base_size=10.0,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "ellipsoidal_3d.png", dpi=150, bbox_inches="tight")
    print(f"    Ellipsoidal 3D saved to: {output_dir / 'ellipsoidal_3d.png'}")

    # ===========================
    # Columnar
    # ===========================
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    print()
    print("=" * 60)
    print("3. Columnar:")
    print("-" * 60)
    print("Starting Columnar Z-axis")

    gen = ColumnarGenerator(
        num_grains=num_grains,
        axis="z",
        aspect_ratio=8.0,
        base_size=8.0,
        size_variation=0.2,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "columnar_z.png", dpi=150, bbox_inches="tight")
    print(f"    Columnar in Z-axis saved to: {output_dir / 'columnar_z.png'}")

    print()
    print("Starting Columnar X-axis")
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    gen = ColumnarGenerator(
        num_grains=num_grains,
        axis="x",
        aspect_ratio=8.0,
        base_size=8.0,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "columnar_x.png", dpi=150, bbox_inches="tight")
    print(f"    Columnar in X-axis saved to: {output_dir / 'columnar_x.png'}")

    # ===========================
    # Mixed
    # ===========================
    micro_2d = Microstructure(dimensions=dims_2d, resolution=resolution)
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    print()
    print("=" * 60)
    print("4. Mixed (ellipsoidal + equiaxed):")
    print("-" * 60)
    print("Starting Mixed 2D")

    gen = MixedGenerator(
        num_grains=num_grains,
        ellipsoid_fraction=0.6,
        aspect_ratio_mean=5.0,
        aspect_ratio_std=1.0,
        base_size=10.0,
        seed=seed,
    )
    fig, ax = plt.subplots()
    Plotter.plot_grain_ids(ax, micro_2d)
    fig.savefig(output_dir / "mixed_2d.png", dpi=150, bbox_inches="tight")
    print(f"    Mixed 2D saved to: {output_dir / 'mixed_2d.png'}")

    # 3D Mixed
    print()
    print("Starting Mixed 3D")
    gen = MixedGenerator(
        num_grains=num_grains,
        ellipsoid_fraction=0.5,
        aspect_ratio_mean=6.0,
        base_size=10.0,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "mixed_3d.png", dpi=150, bbox_inches="tight")
    print(f"    Mixed 3D saved to: {output_dir / 'mixed_3d.png'}")

    # ===========================
    # Lath
    # ===========================
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    print()
    print("=" * 60)
    print("5. Lath (Martensitic/Bainitic):")
    print("-" * 60)
    print("Starting Lath Pure Colony")

    gen = LathGenerator(
        num_grains=400,
        num_colonies=25,
        aspect_ratio_mean=12.0,
        aspect_ratio_std=1.0,
        width_mean=2.0,
        width_std=0.5,
        colony_misorientation=5.0,
        basketweave_fraction=0.0,
        bw_variants=24,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "lath_colony.png", dpi=150, bbox_inches="tight")
    print(f"  Pure Colony saved to: {output_dir / 'lath_colony.png'}")

    # Get colony information
    colony_info = gen.get_colony_info()
    print(f"    Colonies: {colony_info['num_colonies']}")
    print(
        f"    Avg laths per colony: {len(colony_info['colony_ids']) / colony_info['num_colonies']:.1f}"
    )

    print()
    print("Starting Lath Pure Basketweave")
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    gen = LathGenerator(
        num_grains=400,
        num_colonies=25,
        aspect_ratio_mean=12.0,
        aspect_ratio_std=1.0,
        width_mean=2.0,
        width_std=0.5,
        colony_misorientation=5.0,
        basketweave_fraction=1.0,
        bw_variants=24,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "lath_basketweave.png", dpi=150, bbox_inches="tight")
    print(f"  Pure Basketweave saved to: {output_dir / 'lath_basketweave.png'}")

    # Get colony information
    colony_info = gen.get_colony_info()
    print(f"    Colonies: {colony_info['num_colonies']}")
    print(
        f"    Avg laths per colony: {len(colony_info['colony_ids']) / colony_info['num_colonies']:.1f}"
    )

    print()
    print("Starting Lath Mixed Colony/Basketweave")
    micro_3d = Microstructure(dimensions=dims_3d, resolution=resolution)
    gen = LathGenerator(
        num_grains=400,
        num_colonies=25,
        aspect_ratio_mean=12.0,
        aspect_ratio_std=1.0,
        width_mean=2.0,
        width_std=0.5,
        colony_misorientation=5.0,
        basketweave_fraction=0.7,
        bw_variants=24,
        seed=seed,
        chunk_size=1_000_000,
    )
    gen.generate(micro_3d)
    fig = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig, micro_3d, shuffle=True)
    fig.savefig(output_dir / "lath_mixed.png", dpi=150, bbox_inches="tight")
    print(f"  Mixed Basketweave/Colony saved to: {output_dir / 'lath_mixed.png'}")

    # Get colony information
    colony_info = gen.get_colony_info()
    print(f"    Colonies: {colony_info['num_colonies']}")
    print(
        f"    Avg laths per colony: {len(colony_info['colony_ids']) / colony_info['num_colonies']:.1f}"
    )

    print("\n" + "=" * 60)
    print(f"All examples saved to: {output_dir}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
