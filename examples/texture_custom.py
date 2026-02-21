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

# ========================================================
# Custom texture
# ========================================================

hkl = [1, 2, 3]
uvw = [1, 1, -1]

print()
print("=" * 60)
print("Custom Texture:")
print(f"(hkl) = ({hkl[0]}{hkl[1]}{hkl[2]})")
print(f"[uvw] = [{uvw[0]}{uvw[1]}{uvw[2]}]")
print("-" * 60)

# Reassign microstructure
custom_micro = micro.copy()

# Generate texture for the middle
middle_reg_texture = CustomTexture(hkl, uvw, degspread=2.0)
middle_orientations = middle_reg_texture.generate(middle_grains)

# Reassign textures for the grains in the box region
custom_micro.orientations[middle_grains] = middle_orientations.orientations

# Plot
custom_fig, custom_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(custom_axes, custom_micro)
custom_fig.suptitle(f"Custom Texture\n"
                   f"(hkl) = ({hkl[0]}{hkl[1]}{hkl[2]})\n"
                   f"[uvw] = ({uvw[0]}{uvw[1]}{uvw[2]})", 
                   fontsize=15)
plt.tight_layout()
plt.savefig(output_dir / "texture_custom.png", dpi=150, bbox_inches="tight")

print("  Custom texture filename: 'texture_custom.png'")

plt.show()
