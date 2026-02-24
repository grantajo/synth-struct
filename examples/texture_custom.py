# synth-struct/examples/texture_custom.py

"""
This is an example that shows how to use the custom textures
feature. This example then saves IPF maps.

Custom texture takes in hkl and uvw values with
(hkl) || ND and [uvw] || RD
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.micro_utils import get_grains_in_region
from synth_struct.generators.voronoi import VoronoiGenerator
from synth_struct.orientation.texture.random import RandomTexture
from synth_struct.orientation.texture.custom import CustomTexture
import synth_struct.plotting.ipf_maps as IPFplot

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
example_base_dir = repo_root / "output/texture"
example_base_dir.mkdir(exist_ok=True)
output_dir = repo_root / "output/texture/custom"
output_dir.mkdir(exist_ok=True)

print("=" * 17, "Custom Texture Examples", "=" * 18)

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

print("-" * 60)
print("  Base texture filename: 'texture_custom_base.png'")

# ========================================================
# Custom texture (Cubic)
# ========================================================

# This is the cube texture for testing
# hkl = [0, 0, 1]
# uvw = [1, 0, 0]

# This is an arbitrary texture
hkl = [1, 2, 3]
uvw = [1, 1, -1]


print("=" * 60)
print("Custom Texture (Cubic):")
print(f"(hkl) = ({hkl[0]}{hkl[1]}{hkl[2]})")
print(f"[uvw] = [{uvw[0]}{uvw[1]}{uvw[2]}]")
print("-" * 60)

# Reassign microstructure
cubic_micro = micro.copy()

# Generate texture for the middle
middle_reg_texture = CustomTexture(
    hkl, 
    uvw, 
    crystal_system="cubic",
    degspread=5.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the box region
cubic_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
cubic_fig, cubic_axes = plt.subplots(1, 3, figsize=(13, 5))
IPFplot.plot_multiple_ipf_maps(cubic_axes, cubic_micro)
cubic_fig.suptitle(
    f"Custom Texture (Cubic)\n"
    f"(hkl) = ({hkl[0]}{hkl[1]}{hkl[2]})\n"
    f"[uvw] = ({uvw[0]}{uvw[1]}{uvw[2]})",
    fontsize=15,
)
plt.tight_layout()
plt.savefig(output_dir / "texture_custom_cubic.png", dpi=150, bbox_inches="tight")


print("  Custom cubic texture filename: 'texture_custom_cubic.png'")

# Uncomment when orix plotting is fixed for hexagonal
# ========================================================
# Custom texture (Hexagonal)
# ========================================================

# Example from Ti
hkil = [1, 0, -1, 0]
uvtw = [1, 2, -3, 0]
lattice_params = (2.95, 2.95, 4.68)

print()
print("=" * 60)
print("Custom Texture (Hexagonal):")
print(f"(hkil) = ({hkil[0]}{hkil[1]}{hkil[2]}{hkil[3]})")
print(f"[uvtw] = [{uvtw[0]}{uvtw[1]}{uvtw[2]}{uvtw[3]}]")
print("-" * 60)

hex_micro = micro.copy()
middle_reg_texture = CustomTexture(
    hkil, 
    uvtw,
    crystal_system="hexagonal",
    lattice_params=lattice_params)

middle_orientations = middle_reg_texture.generate(middle_grains)

hex_micro.orientations[middle_grains] = middle_orientations.orientations

hex_fig, hex_axes = plt.subplots(1, 3, figsize=(14, 5))
IPFplot.plot_multiple_ipf_maps(hex_axes, hex_micro)
hex_fig.suptitle(
    f"Custom Texture (Hexagonal)\n"
    f"(hkil) = ({hkil[0]}{hkil[1]}{hkil[2]}{hkil[3]})\n"
    f"[uvw] = ({uvtw[0]}{uvtw[1]}{uvtw[2]}{uvtw[3]})",
    fontsize=15,
)
plt.tight_layout()
plt.savefig(output_dir / "texture_custom_hex.png", dpi=150, bbox_inches="tight")

print("  Custom Hexagonal texture filename: 'texture_custom_hex.png'")

print("-" * 60)
print(f"All files saved to: \n{output_dir}")

plt.show()
