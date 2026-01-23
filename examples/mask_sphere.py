import sys
sys.path.insert(0, '../src')

from microstructure import Microstructure
from texture import Texture
from visualization import IPFVisualizer

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Spherical region in center
sphere_grains = micro.get_grains_in_region(
    'sphere',
    center=[50, 50, 50],
    radius=15
)
print(f"Sphere region contains {len(sphere_grains)} grains")

micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=sphere_grains,
    texture_type='cube',
    degspread=5
)

# Create IPF-Z map
IPFVisualizer.plot_ipf_map(
    micro,
    direction='z',
    crystal_structure='cubic',
    slice_idx=50,
    slice_direction='z',
    filename='ipf_z_map_sphere.png',
    show_legend=True
)

# Show multiple IPF directions
IPFVisualizer.plot_multiple_ipf_maps(
    micro,
    directions=['x', 'y', 'z'],
    crystal_structure='cubic',
    slice_idx=50,
    filename='ipf_xyz_maps_sphere.png'
)
