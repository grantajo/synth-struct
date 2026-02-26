# synth-struct/examples/shapes.py

"""
This example creates a microstructure with each of the various Microstructure generator classes

Generates both 2D and 3D examples for each type.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct import (
    Microstructure,
    VoronoiGenerator,
    EllipsoidalGenerator,
    ColumnarGenerator,
    MixedGenerator,
    LathGenerator,
    Plotter,
)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    repo_root = project_root
    output_dir = repo_root / "output/shapes"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("=" * 12, "Microstructure generator examples", "=" * 13)
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
    fig2dv, ax2dv = plt.subplots()
    Plotter.plot_grain_ids(ax2dv, micro_2d)
    fig2dv.suptitle("Voronoi 2D")
    plt.tight_layout()
    fig2dv.savefig(output_dir / "voronoi_2d.png", dpi=100, bbox_inches="tight")
    print("    Voronoi 2D saved to:")
    print(f"    {output_dir / 'voronoi_2d.png'}")

    # 3D Voronoi
    print()
    print("Starting Standard Voronoi 3D")
    gen = VoronoiGenerator(num_grains=num_grains, seed=seed)
    gen.generate(micro_3d)
    fig3dv = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig3dv, micro_3d, shuffle=True)
    fig3dv.suptitle("Voronoi 3D")
    plt.tight_layout()
    fig3dv.savefig(output_dir / "voronoi_3d.png", dpi=100, bbox_inches="tight")
    print("    Voronoi 3D saved to:")
    print(f"    {output_dir / 'voronoi_3d.png'}")

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
    fig2e, ax2e = plt.subplots()
    Plotter.plot_grain_ids(ax2e, micro_2d)
    fig2e.suptitle("Ellipsoidal 2D")
    plt.tight_layout()
    fig2e.savefig(output_dir / "ellipsoidal_2d.png", dpi=150, bbox_inches="tight")
    print("    Ellipsoidal 2D saved to:")
    print(f"    {output_dir / 'ellipsoidal_2d.png'}")

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
    fig3e = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig3e, micro_3d, shuffle=True)
    fig3e.suptitle("Ellipsoidal 3D")
    plt.tight_layout()
    fig3e.savefig(output_dir / "ellipsoidal_3d.png", dpi=150, bbox_inches="tight")
    print("    Ellipsoidal 3D saved to:")
    print(f"    {output_dir / 'ellipsoidal_3d.png'}")

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
    fig3cz = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig3cz, micro_3d, shuffle=True)
    fig3cz.suptitle("Columnar 3D Z-axis")
    plt.tight_layout()
    fig3cz.savefig(output_dir / "columnar_z.png", dpi=150, bbox_inches="tight")
    print("    Columnar in Z-axis saved to:")
    print(f"    {output_dir / 'columnar_z.png'}")

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
    fig3cx = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig3cx, micro_3d, shuffle=True)
    fig3cx.suptitle("Columnar 3D X-axis")
    plt.tight_layout()
    fig3cx.savefig(output_dir / "columnar_x.png", dpi=150, bbox_inches="tight")
    print("    Columnar in X-axis saved to:")
    print(f"    {output_dir / 'columnar_x.png'}")

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
    gen.generate(micro_2d)
    fig2m, ax2m = plt.subplots()
    Plotter.plot_grain_ids(ax2m, micro_2d)
    fig2m.suptitle("Mixed 2D")
    plt.tight_layout()
    fig2m.savefig(output_dir / "mixed_2d.png", dpi=150, bbox_inches="tight")
    print("    Mixed 2D saved to:")
    print(f"    {output_dir / 'mixed_2d.png'}")

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
    fig3m = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(fig3m, micro_3d, shuffle=True)
    fig3m.suptitle("Mixed 3D")
    plt.tight_layout()
    fig3m.savefig(output_dir / "mixed_3d.png", dpi=150, bbox_inches="tight")
    print("    Mixed 3D saved to:")
    print(f"    {output_dir / 'mixed_3d.png'}")

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
    figlc = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(figlc, micro_3d, shuffle=True)
    figlc.suptitle("Lath Pure Colony")
    plt.tight_layout()
    figlc.savefig(output_dir / "lath_colony.png", dpi=150, bbox_inches="tight")
    print("  Pure Colony saved to:")
    print(f"  {output_dir / 'lath_colony.png'}")

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
    figlb = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(figlb, micro_3d, shuffle=True)
    figlb.suptitle("Lath Pure Basketweave")
    plt.tight_layout()
    figlb.savefig(output_dir / "lath_basketweave.png", dpi=150, bbox_inches="tight")
    print("  Pure Basketweave saved to:")
    print(f"  {output_dir / 'lath_basketweave.png'}")

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
    figlm = plt.figure(figsize=(15, 5))
    Plotter.plot_3d_slices(figlm, micro_3d, shuffle=True)
    figlm.suptitle("Lath Mixed Basketweave/Colony")
    plt.tight_layout()
    figlm.savefig(output_dir / "lath_mixed.png", dpi=150, bbox_inches="tight")
    print("  Mixed Basketweave/Colony saved to:")
    print(f"  {output_dir / 'lath_mixed.png'}")

    # Get colony information
    colony_info = gen.get_colony_info()
    print(f"    Colonies: {colony_info['num_colonies']}")
    print(
        f"    Avg laths per colony: {len(colony_info['colony_ids'])}"
        f" {colony_info['num_colonies']:.1f}"
    )

    print()
    print("=" * 60)
    print(f"All examples saved to: \n{output_dir}")
    print("=" * 60)

    # plt.show()


if __name__ == "__main__":
    main()
