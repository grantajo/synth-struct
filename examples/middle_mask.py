# /synth_struct/examples/middle_mask.py

"""
This is an example of creating a 3D microstructure and changing the orientation
of the middle grains
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.micro_utils import get_grains_in_region
from synth_struct.generators.voronoi import VoronoiGenerator
from synth_struct.orientation.texture.random import RandomTexture
from synth_struct.orientation.texture.cubic import CubicTexture
from synth_struct.plotting.gen_plot import Plotter
import synth_struct.plotting.ipf_maps as IPFplot

"""
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

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
output_dir = repo_root / "output/middle_mask"
output_dir.mkdir(exist_ok=True)

print("=" * 11, "Middle Orientation Example", "=" * 11)

# Variables for microstructure generation
dims = (200, 200, 200)
resolution = 1.0
num_grains = 500
mid_text = "rotated_cube"

# Create microstructure
micro = Microstructure(dimensions=dims, resolution=resolution)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

# Generate random texture for entire microstructure
random_texture = RandomTexture()
random_texture.generate(micro)

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

# Separate middle grains and apply Cubic texture
middle_grains = get_grains_in_region(micro, "sphere", center=[100, 100, 100], radius=60)
middle_texture = CubicTexture(mid_text, degspread=5.0)
middle_texture = middle_texture.generate(middle_grains)
micro.orientations[middle_grains] = middle_texture.orientations

print(f"Middle region contains {len(middle_grains)} grains")

print()

#
# Plotting
#

print("Plotting microstructure:")
# Plot the microstructure grain IDs
print("  Plotting grain IDs")
fig1 = plt.figure(figsize=(15, 5))
Plotter.plot_3d_slices(fig1, micro)
plt.savefig(output_dir / "middle_grain_structure.png", dpi=150, bbox_inches="tight")
print("    Done plotting grain IDs")


# Plot the IPF-Z maps
print("  Plotting IPF maps")
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(axes, micro)
print("Done plotting IPF maps")
plt.tight_layout()
plt.savefig(output_dir / "middle_texture_IPFs.png", dpi=150, bbox_inches="tight")

print("-" * 50)
print(f"    Saved figures to: \n{output_dir}")
print("-" * 50)

# plt.show()
