import sys
sys.path.insert(0, '../src')

import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar

from microstructure import Microstructure
from texture import Texture
from ipf_visualization import OrixIPFVisualizer

micro = Microstructure(dimensions=(100,100,100), resolution=1.0)
micro.gen_voronoi(num_grains=150, seed=42)
micro.orientations = Texture.random_orientations(150, seed=42)

center_grains = micro.get_grains_in_region('sphere', center=[50,50,50], radius=25)
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=center_grains,
    texture_type='brass',
    degspread=10
)

# IPF map
OrixIPFVisualizer.plot_ipf_map(
    micro,
    direction='z',
    crystal_structure='cubic',
    slice_idx=50,
    slice_direction='z',
    filename='example_ipf_orix.png'
)

# Muliple IPF maps
for direction in ['x', 'y', 'z']:
    OrixIPFVisualizer.plot_ipf_map(
        micro,
        direction=direction,
        crystal_structure='cubic',
        slice_idx=50,
        filename=f'example_ipf_{direction}_orix.png'
    )
    
# Pole Figures
OrixIPFVisualizer.plot_pole_figure(
    micro,
    crystal_structure='cubic',
    miller_indices=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    filename='example_pole_figures.png'
)

# ODF
OrixIPFVisualizer.plot_odf(
    micro,
    crystal_structure='cubic',
    filename='example_odf.png'
)


