import sys

sys.path.insert(0, "../src")

from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Multiple layers with different textures

# Bottom third
bottom_grains = micro.get_grains_in_region("box", z_min=0, z_max=33)
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations, region_grain_ids=bottom_grains, texture_type="brass"
)

# Middle third
middle_grains = micro.get_grains_in_region("box", z_min=34, z_max=66)
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations, region_grain_ids=middle_grains, texture_type="goss"
)

# Top third
top_grains = micro.get_grains_in_region("box", z_min=67, z_max=100)
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations, region_grain_ids=top_grains, texture_type="cube"
)

print(f"Bottom: {len(bottom_grains)} grains (brass)")
print(f"Middle: {len(middle_grains)} grains (goss)")
print(f"Top: {len(top_grains)} grains (cube)")
