from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

x, y, z = np.mgrid[0:100, 0:100, 0:100]
custom_mask = x > 50  # Right half
custom_grains = micro.get_grains_in_region('custom_mask', mask=custom_mask)
print(f"Custom region contains {len(custom_grains)} grains")

# Apply texture to box region
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations,
    region_grain_ids=box_grains,
    texture_type='brass',
    degspread=5
)
