from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Box region (center portion)
box_grains = micro.get_grains_in_region(
    'box',
    x_min=25, x_max=75,
    y_min=25, y_max=75,
    z_min=25, z_max=75
)
print(f"Box region contains {len(box_grains)} grains")

# Apply texture to box region
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=box_grains,
    texture_type='brass',
    degspread=10
)
