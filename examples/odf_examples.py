# synth-struct/examples/odf_examples.py

"""
This example shows the capability to plot ODFs for various
microstructure examples.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synth_struct import (
    Microstructure,
    Phase,
    VoronoiGenerator,
    RandomTexture,
    CubicTexture,
    HexagonalTexture,
    get_grains_in_region,
)
from synth_struct import ipf_maps as IPFplot
from synth_struct import odf_plot as ODFplot

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

Hexagonal textures:
"basal"
"prismatic"
"pyramidal"
"""

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
output_dir = repo_root / "output/odf"
output_dir.mkdir(exist_ok=True)

print("=" * 12, "Orientation Distribution Functions", "=" * 12)

# Variables for microstructure generation
dims = (200, 200)
res = 1.0
num_grains = 200
default_phase = Phase.from_preset("default")

# Create microstructure
micro = Microstructure(dimensions=dims, resolution=res, phase=default_phase)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

# Generate random texture for entire microstructure
random_texture = RandomTexture()
base_texture = random_texture.generate(micro)
micro.assign_texture(base_texture)

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

print("-" * 60)
print("Plotting Base IPF maps:")
base_fig, base_axes = plt.subplots(1, 3, figsize=(15, 5))
IPFplot.plot_multiple_ipf_maps(base_axes, micro)
base_fig.suptitle("Random Base Texture IPF maps")
plt.tight_layout()
plt.savefig(output_dir / "ipf_maps_base.png", dpi=150, bbox_inches="tight")
print("IPF map filename: 'ipf_maps_base.png'")

print("-" * 60)
print("Plotting Base ODF")
odf_random_fig, odf_random_axes = ODFplot.create_odf_figure(
    micro, filename="odf_base.png"
)
odf_random_fig.suptitle("Random Base Texture ODF")
plt.tight_layout()
plt.savefig(output_dir / "odf_base.png", dpi=150, bbox_inches="tight")
print("ODF filename: 'odf_base.png'")

plt.show()
