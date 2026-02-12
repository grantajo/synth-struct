import sys

sys.path.insert(0, "../src")

from microstructure import Microstructure
from texture import Texture

# Create microstructure
micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
micro.gen_voronoi(num_grains=200, seed=42)
micro.orientations = Texture.random_orientations(200, seed=42)

# Example: Box region (center portion)
box_grains = micro.get_grains_in_region(
    "box", x_min=25, x_max=75, y_min=25, y_max=75, z_min=25, z_max=75
)
print(f"Box region contains {len(box_grains)} grains")

# Apply texture to box region
micro.orientations = Texture.apply_texture_to_region(
    micro.orientations, region_grain_ids=box_grains, texture_type="brass", degspread=10
)

fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
middle_slice = micro.grain_ids.shape[2] // 2
im = axs[0].imshow(micro.grain_ids[:, :, middle_slice], cmap="nipy_spectral")
axs[0].set_title(f"Z-slice at {middle_slice}")

middle_slice = micro.grain_ids.shape[1] // 2
axs[1].imshow(micro.grain_ids[:, middle_slice, :], cmap="nipy_spectral")
axs[1].set_title(f"Y-slice at {middle_slice}")

middle_slice = micro.grain_ids.shape[0] // 2
axs[2].imshow(micro.grain_ids[middle_slice, :, :], cmap="nipy_spectral")
axs[2].set_title(f"X-slice at {middle_slice}")

fig.colorbar(
    im, ax=axs, orientation="horizontal", label="Grain ID", pad=0.08, aspect=40
)
fig.suptitle("Box Mask")
plt.savefig("../output/3d_slices.png", dpi=150)
plt.close()
