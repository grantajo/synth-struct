# synth-struct/examples/pole_figure_plotting.py

"""
This is an example that shows how to use the pole figure plotting
feature. This example then saves the pole figures
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from orix.plot import register_projections

from synth_struct import (
    Microstructure,
    Phase,
    VoronoiGenerator,
    RandomTexture,
    CubicTexture,
    get_grains_in_region,
)
from synth_struct import pole_figures as PFplot
from synth_struct import ipf_maps as IPFmap
from synth_struct import inverse_pole_figures as IPFplot

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
output_dir = repo_root / "output/pole_figures"
output_dir.mkdir(exist_ok=True)

register_projections()

print("=" * 14, "Pole Figure Plotting Examples", "=" * 15)

dims = (100, 100, 100)
res = 1.0
num_grains = 250

default_phase = Phase.from_preset("default_cubic")

micro = Microstructure(
    dimensions=dims,
    resolution=res,
    phase=default_phase,
)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

random_texture = RandomTexture(phase=default_phase)
base_texture = random_texture.generate(micro)
micro.assign_texture(base_texture)

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

print("-" * 60)
print("Plotting figures for base random texture")
print("-Plotting IPF maps...")
base_ipf_fig, base_ipf_ax = plt.subplots(1, 3, figsize=(15, 5))
IPFmap.plot_multiple_ipf_maps(base_ipf_ax, micro)
base_ipf_fig.suptitle("IPF Maps for Random Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "base_texture_ipf_maps.png", dpi=150, bbox_inches="tight")
print(f"  Saved IPF maps to 'base_texture_ipf_maps.png'")

print("-Plotting pole figures...")
base_pf_fig, base_pf_ax = plt.subplots(
    1, 3, 
    figsize=(15, 5), 
    subplot_kw={"projection": "stereographic"}
)
PFplot.plot_multiple_pole_figures(
    base_pf_ax,
    micro,
    phase_id=0,
    miller_indices=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
    sample_fraction=0.005,
    plot_type='scatter',
)
base_pf_fig.suptitle("Pole Figures for Random Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "base_texture_pole_figures.png", dpi=150, bbox_inches="tight")
print(f"  Saved pole figures to 'base_texture_pole_figures.png'")

# ========================================================
# Cube texture with 5° spread
# ========================================================
print()
print("-" * 60)
print("Creating texture in middle of the microstructure with 5° spread...")

micro_5deg = micro.copy()
texture = "cube"

middle_grains = get_grains_in_region(micro_5deg, "sphere", center=[50, 50, 50], radius=30)
print(f"  Middle region contains {len(middle_grains)} grains")

# Generate texture for the middle
middle_reg_texture = CubicTexture(texture, phase=default_phase, degspread=5.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the box region
micro_5deg.assign_texture(middle_orientations, grain_ids=middle_grains)

fig_5deg_ipf, ax_5deg_ipf = plt.subplots(1, 3, figsize=(15, 5))
IPFmap.plot_multiple_ipf_maps(ax_5deg_ipf, micro_5deg)
fig_5deg_ipf.suptitle("IPF Maps with Cube Texture (5° Spread)", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "cube_texture_5deg_ipf_maps.png", dpi=150, bbox_inches="tight")
print(f"  Saved IPF maps to 'cube_texture_5deg_ipf_maps.png'")

fig_5deg_pf, ax_5deg_pf = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "stereographic"})
PFplot.plot_multiple_pole_figures(
    ax_5deg_pf,
    micro_5deg,
    phase_id=0,
    miller_indices=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],
    sample_fraction=0.005,
    plot_type='density',
    sigma=5.0,
)
fig_5deg_pf.suptitle("Pole Figures with Cube Texture (5° Spread)", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "cube_texture_5deg_pole_figures.png", dpi=150, bbox_inches="tight")
print(f"  Saved pole figures to 'cube_texture_5deg_pole_figures.png'")

# ========================================================
# Cube texture with 10° spread
# ========================================================

print()
print("-" * 60)
print("Creating texture in middle of the microstructure with 10° spread...")

micro_10deg = micro.copy()
texture = "cube"

middle_grains = get_grains_in_region(micro_10deg, "sphere", center=[50, 50, 50], radius=30)
print(f"  Middle region contains {len(middle_grains)} grains")

# Generate texture for the middle
middle_reg_texture = CubicTexture(texture, phase=default_phase, degspread=10.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the box region
micro_10deg.assign_texture(middle_orientations, grain_ids=middle_grains)

fig_10deg_ipf, ax_10deg_ipf = plt.subplots(1, 3, figsize=(15, 5))
IPFmap.plot_multiple_ipf_maps(ax_10deg_ipf, micro_10deg)
fig_10deg_ipf.suptitle("IPF Maps with Cube Texture (10° Spread)", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "cube_texture_10deg_ipf_maps.png", dpi=150, bbox_inches="tight")
print(f"  Saved IPF maps to 'cube_texture_10deg_ipf_maps.png'")

fig_10deg_pf = plt.figure(figsize=(15, 5))
ax_10deg_pf = IPFplot.create_ipf_axes(fig_10deg_pf, 3, layout="row")
IPFplot.plot_multiple_ipfs(
    ax_10deg_pf,
    micro_10deg,
    directions=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    phase_id=0,
    sample_fraction=0.005,
    plot_type='density',
)
fig_10deg_pf.suptitle("Inverse Pole Figures with Cube Texture (10° Spread)", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "cube_texture_10deg_inverse_pole_figures.png", dpi=150, bbox_inches="tight")
print(f"  Saved inverse pole figures to 'cube_texture_10deg_inverse_pole_figures.png'")

print()
print("=" * 60)
print(f"All figures saved to: {output_dir}")

plt.show()