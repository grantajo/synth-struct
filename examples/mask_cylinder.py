import sys
sys.path.insert(0, '../src')

from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Cylindrical region along Z-axis
cylinder_grains = micro.get_grains_in_region(
    'cylinder',
    center=[50, 50],  # XY center
    radius=20,
    z_min=30,
    z_max=70,
    axis='z'
)
print(f"Cylinder region contains {len(cylinder_grains)} grains")

micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=sphere_grains,
    texture_type='cube',
    degspread=5
)
