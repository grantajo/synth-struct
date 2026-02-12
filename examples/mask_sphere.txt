import sys

sys.path.insert(0, "../src")

from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Spherical region in center
sphere_grains = micro.get_grains_in_region("sphere", center=[50, 50, 50], radius=15)
print(f"Sphere region contains {len(sphere_grains)} grains")

micro.orientations = Texture.apply_texture_to_region(
    micro.orientations, region_grain_ids=sphere_grains, texture_type="cube", degspread=5
)
