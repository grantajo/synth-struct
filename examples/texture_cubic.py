# synth-struct/examples/texture_cubic.py

"""
This is an example that shows each of the cubic textures and saves plots
for each of them.

Cubic textures:
"cube"
"goss"
"brass"
"copper"
"s"
"p"
"rotated_cube"
"rotated_goss"
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.micro_utils import get_grains_in_region
from synth_struct.generators.voronoi import VoronoiGenerator
from synth_struct.orientation.texture.random import RandomTexture
from synth_struct.orientation.texture.cubic import CubicTexture
import synth_struct.plotting.ipf_maps as IPFplot

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
example_base_dir = repo_root / "output/texture"
example_base_dir.mkdir(exist_ok=True)
output_dir = repo_root / "output/texture/cubic"
output_dir.mkdir(exist_ok=True)

print("=" * 21, "Texture Examples", "=" * 21)

# Variables for microstructure generation
dims = (200, 200)
resolution = 1.0
num_grains = 300

# Create microstructure
micro = Microstructure(dimensions=dims, resolution=resolution)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

# Generate random texture for entire microstructure
random_texture = RandomTexture()
random_texture.generate(micro)

print(f"Created 2D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

base_fig, base_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(base_axes, micro)
base_fig.suptitle("Random Base Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_random.png", dpi=150, bbox_inches="tight")

# Get grains in the middle and display how many grains are changed
middle_grains = get_grains_in_region(micro, "sphere", center=[100, 100], radius=50)

print(f"  Middle region contains {len(middle_grains)} grains")

# ========================================================
# Cube texture
# ========================================================
print()
print("=" * 60)
print("1. Cube Texture:")
print("-" * 60)

# Reassign microstructure
cube_micro = micro.copy()
texture = "cube"

# Generate texture for the middle
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the box region
cube_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
cube_fig, cube_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(cube_axes, cube_micro)
cube_fig.suptitle("Cube Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_cube.png", dpi=150, bbox_inches="tight")

print("  Cube texture filename: 'texture_cube.png'")

# ========================================================
# Goss texture
# ========================================================
print()
print("=" * 60)
print("2. Goss Texture:")
print("-" * 60)

# Reassign microstructure
goss_micro = micro.copy()
texture = "goss"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
goss_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
goss_fig, goss_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(goss_axes, goss_micro)
goss_fig.suptitle("Goss Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_goss.png", dpi=150, bbox_inches="tight")

print("  Goss texture filename: 'texture_goss.png'")

# ========================================================
# Brass texture
# ========================================================
print()
print("=" * 60)
print("3. Brass Texture:")
print("-" * 60)

# Reassign microstructure
brass_micro = micro.copy()
texture = "brass"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
brass_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
brass_fig, brass_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(brass_axes, brass_micro)
brass_fig.suptitle("Brass Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_brass.png", dpi=150, bbox_inches="tight")

print("  Brass texture filename: 'texture_brass.png'")

# ========================================================
# Copper texture
# ========================================================
print()
print("=" * 60)
print("4. Copper Texture:")
print("-" * 60)

# Reassign microstructure
cu_micro = micro.copy()
texture = "copper"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
cu_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
cu_fig, cu_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(cu_axes, cu_micro)
cu_fig.suptitle("Copper Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_copper.png", dpi=150, bbox_inches="tight")

print("  Copper texture filename: 'texture_copper.png'")

# ========================================================
# S texture
# ========================================================
print()
print("=" * 60)
print("5. S Texture:")
print("-" * 60)

# Reassign microstructure
s_micro = micro.copy()
texture = "s"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
s_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
s_fig, s_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(s_axes, s_micro)
s_fig.suptitle("S Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_s.png", dpi=150, bbox_inches="tight")

print("  S texture filename: 'texture_s.png'")

# ========================================================
# P texture
# ========================================================
print()
print("=" * 60)
print("6. P Texture:")
print("-" * 60)

# Reassign microstructure
p_micro = micro.copy()
texture = "p"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
p_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
p_fig, p_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(p_axes, p_micro)
p_fig.suptitle("P Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_p.png", dpi=150, bbox_inches="tight")

print("  P texture filename: 'texture_p.png'")

# ========================================================
# Rotated Cube texture
# ========================================================
print()
print("=" * 60)
print("7. Rotated Cube Texture:")
print("-" * 60)

# Reassign microstructure
rot_cube_micro = micro.copy()
texture = "rotated_cube"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
rot_cube_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
rot_cube_fig, rot_cube_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(rot_cube_axes, rot_cube_micro)
rot_cube_fig.suptitle("Rotated Cube Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_rotated_cube.png", dpi=150, bbox_inches="tight")

print("  Rotated Cube texture filename: 'texture_rotated_cube.png'")

# ========================================================
# Rotated Goss texture
# ========================================================
print()
print("=" * 60)
print("8. Rotated Goss Texture:")
print("-" * 60)

# Reassign microstructure
rot_goss_micro = micro.copy()
texture = "rotated_goss"

# Generate texture for the box region
middle_reg_texture = CubicTexture(texture, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the middle
rot_goss_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
rot_goss_fig, rot_goss_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(rot_goss_axes, rot_goss_micro)
rot_goss_fig.suptitle("Rotated Goss Texture", fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_rotated_goss.png", dpi=150, bbox_inches="tight")

print("  Rotated Goss texture filename: 'texture_rotated_goss.png'")

# ========================================================
# Show output location one final time
print()
print("-" * 60)
print("Saved all texture figures to:")
print(f"{output_dir}")

plt.show()
