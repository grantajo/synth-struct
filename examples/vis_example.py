import sys

sys.path.insert(0, "../src")

import matplotlib.pyplot as plt


from microstructure import Microstructure
from texture import Texture
from plotting import OrixVisualizer

"""
This example shows how to plot IPF maps, pole figures, and ODFs for a 3D microstructure
"""

micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=150, seed=42)
micro.orientations = Texture.random_orientations(150, seed=42)

center_grains = micro.get_grains_in_region("sphere", center=[50, 50, 50], radius=25)
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=center_grains,
    texture_type="brass",
    degspread=10,
)

"""
# IPF map
OrixVisualizer.plot_ipf_map(
    micro,
    direction='z',
    crystal_structure='cubic',
    slice_idx=50,
    slice_direction='z',
    filename='example_ipf_orix.png'
)

# Muliple IPF maps
for direction in ['x', 'y', 'z']:
    OrixVisualizer.plot_ipf_map(
        micro,
        direction=direction,
        crystal_structure='cubic',
        slice_idx=50,
        filename=f'example_ipf_{direction}_orix.png'
    )
"""
# Pole Figures
miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

n_miller = len(miller_indices)
fig = plt.figure(figsize=(n_miller * 3, 4))

axes = OrixVisualizer.create_pole_figure_axes(fig, len(miller_indices))
artists = OrixVisualizer.plot_all_pole_figures(axes, miller_indices, micro, subset=0.5)

fig.suptitle("Pole Figures", fontsize=14)
fig.tight_layout()
fig.savefig("../output/pole_figures.png", dpi=150)
print("Saved pole figure to '../output/pole_figures.png'")

"""
# ODF
OrixVisualizer.plot_odf(
    micro,
    crystal_structure='cubic',
    filename='example_odf.png'
)
"""
