# /synth-struct/examples/masks.py

"""
This example shows the various mask types that are available by changing
the orientation of the grains for each mask.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synth_struct.microstructure import Microstructure
from synth_struct.micro_utils import get_grains_in_region
from synth_struct.generators.voronoi import VoronoiGenerator
from synth_struct.orientation.texture.random import RandomTexture
from synth_struct.orientation.texture.cubic import CubicTexture
from synth_struct.plotting.gen_plot import Plotter
import synth_struct.plotting.ipf_maps as IPFplot

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

Mask types:
"box"
"sphere"
"cylinder"
"custom_mask"
"""

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

repo_root = project_root
output_dir = repo_root / "output/masks"
output_dir.mkdir(exist_ok=True)

print("=" * 23, "Mask Examples", "=" * 22)

# Variables for microstructure generation
dims = (200, 200, 200)
resolution = 1.0
num_grains = 800

# Create microstructure
micro = Microstructure(dimensions=dims, resolution=resolution)

voronoi_gen = VoronoiGenerator(num_grains=num_grains, seed=42)
voronoi_gen.generate(micro)

# Generate random texture for entire microstructure
random_texture = RandomTexture()
random_texture.generate(micro)

print(f"Created 3D Microstructure: {dims}")
print(f"Number of grains: {micro.num_grains}")

base_fig, base_axes = plt.subplots(1, 3, figsize=(15, 5))
# Figure 0
IPFplot.plot_multiple_ipf_slices(base_axes, micro, 
    slice_indices=[50, 100, 150]
) 
plt.tight_layout()
plt.savefig(output_dir / "base_micro.png", dpi=150, bbox_inches="tight")
print("  Base microstructure filename: 'base_micro.png'")

# ========================================================
# Box mask example
# ========================================================
print()
print("=" * 60)
print("1. Box example:")
print("-" * 60)

# Reassign microstructure
box_micro = micro.copy()
box_texture = "brass"

# Get grains in box mask
box_grains = get_grains_in_region(box_micro, "box", 
    x_min=125, x_max=175,
    y_min=25, y_max=75,
    z_min=50, z_max=150
)

# Generate texture for the box region
box_reg_texture = CubicTexture(box_texture, degspread=2.0)
box_orientations = box_reg_texture.generate(box_grains)

# Reassign textures for the grains in the box region
box_micro.orientations[box_grains] = box_orientations.orientations

print(f"  Box region contains {len(box_grains)} grains")
print()

# Plot box microstructure
print("  Plotting box mask orientation IPF-Z maps")

box_fig, box_axes = plt.subplots(1, 3, figsize=(15, 5))
# Figure 1
IPFplot.plot_multiple_ipf_slices(box_axes, box_micro, 
    slice_indices=[50, 100, 150]
) 

plt.tight_layout()
plt.savefig(output_dir / "mask_box.png", dpi=150, bbox_inches="tight")
print("  Box mask example filename: 'mask_box.png'")


# ========================================================
# Sphere mask example
# ========================================================
print()
print("=" * 60)
print("2. Sphere example:")
print("-" * 60)

# Reassign microstructure
sphere_micro = micro.copy()
sphere_texture = "rotated_goss"

# Get grains in sphere mask
sphere_grains = get_grains_in_region(sphere_micro, "sphere",
    center=[75, 75, 75], radius=50
)

# Generate texture for the masked region
sphere_reg_texture = CubicTexture(sphere_texture, degspread=2.0)
sphere_orientations = sphere_reg_texture.generate(sphere_grains)

# Reassign textures for grains in spherical mask
sphere_micro.orientations[sphere_grains] = sphere_orientations.orientations

print(f"  Spherical region contains {len(sphere_grains)} grains")
print()

# Plot box microstructure
print("  Plotting spherical mask orientation IPF-Z maps")

sphere_fig, sphere_axes = plt.subplots(1, 5, figsize=(15, 5))
# Figure 2
IPFplot.plot_multiple_ipf_slices(sphere_axes, sphere_micro, 
    slice_indices=[25, 50, 75, 100, 150]
) 

plt.tight_layout()
plt.savefig(output_dir / "mask_sphere.png", dpi=150, bbox_inches="tight")
print("  Sphere mask example filename: 'mask_sphere.png'")

# ========================================================
# Cylinder mask example
# ========================================================
print()
print("=" * 60)
print("3. Cylinder example:")
print("-" * 60)

# Reassign microstructure
cyl_micro = micro.copy()
cyl_texture = "cube"

# Get grains in sphere mask
cyl_grains = get_grains_in_region(cyl_micro, "cylinder",
    center=[75, 100], radius=50,
    c_min=50, c_max=150,
    axis="y"
)

# Generate texture for the masked region
cyl_reg_texture = CubicTexture(cyl_texture, degspread=2.0)
cyl_orientations = cyl_reg_texture.generate(cyl_grains)

# Reassign textures for grains in spherical mask
cyl_micro.orientations[cyl_grains] = cyl_orientations.orientations

print(f"  Cylinder region contains {len(cyl_grains)} grains")
print()

# Plot box microstructure
print("  Plotting cylindrical mask orientation IPF-Z maps")

# Plot with slices in z direction
cyl_fig, cyl_axes = plt.subplots(1, 3, figsize=(15, 5))
# Figure 3
IPFplot.plot_multiple_ipf_slices(cyl_axes, cyl_micro, 
    slice_indices=[75, 100, 125], slice_direction="z"
) 
plt.tight_layout()
plt.savefig(output_dir / "mask_cylinder_zslices.png", dpi=150, bbox_inches="tight")

print("  Cylinder mask example filename: 'mask_cylinder_zslices.png'")

# Plot with slices in the y direction
cyl_fig1, cyl_axes1 = plt.subplots(1, 3, figsize=(15, 5))
# Figure 4
IPFplot.plot_multiple_ipf_slices(cyl_axes1, cyl_micro, 
    slice_indices=[50, 100, 150], slice_direction="y"
) 
plt.tight_layout()
plt.savefig(output_dir / "mask_cylinder_yslices.png", dpi=150, bbox_inches="tight")

print("  Cylinder mask example filename: 'mask_cylinder_yslices.png'")


# ========================================================
# Custom tetrahedron mask example
# ========================================================
print()
print("=" * 60)
print("4. Custom (tetrahedron) example:")
print("-" * 60)

# Utility function for getting the points inside of a tetrahedron for 
# custom mask example
def points_in_tetrahedron(p, v0, v1, v2, v3):
    """
    Returns True where points p are inside a tetrahedron defined by
    points v0, v1, v2, and v3. This will return the custom mask.
    
    p shape: (..., 3)
    """
    A = np.array([v1 - v0, v2 - v0, v3 - v0]).T 
    
    p_shifted = p - v0
    orig_shape = p_shifted.shape
    flat = p_shifted.reshape(-1, 3)
    
    coords = np.linalg.solve(A, flat.T).T
    s, t, u = coords[:, 0], coords[:, 1], coords[:, 2]
    
    inside = (s >= 0) & (t >= 0) & (u >= 0) & (s + t + u <= 1)
    return inside.reshape(orig_shape[:-1])

# Reassign microstructure
tet_micro = micro.copy()
tet_texture = "copper"

# Initialize boolean grid and tetrahedron points
x, y, z = np.meshgrid(
    np.arange(0, dims[0], resolution),
    np.arange(0, dims[1], resolution),
    np.arange(0, dims[2], resolution),
    indexing='ij'
)

points = np.stack([x, y, z], axis=-1)

v0 = np.array([30, 30, 30])
v1 = np.array([170, 30, 30])
v2 = np.array([100, 170, 30])
v3 = np.array([100, 100, 170])

# Get the points in tetrahedron and create mask
tet_mask = points_in_tetrahedron(points, v0, v1, v2, v3)
tet_grains = get_grains_in_region(tet_micro, "custom_mask",
    mask=tet_mask
)

# Generate texture for the masked region
tet_reg_texture = CubicTexture(tet_texture, degspread=2.0)
tet_orientations = tet_reg_texture.generate(tet_grains)

# Reassign textures for grains in spherical mask
tet_micro.orientations[tet_grains] = tet_orientations.orientations

print(f"  Tetrahedron contains {len(tet_grains)} grains")
print()

# Plot box microstructure
print("  Plotting custom tetrahedron layer mask orientation IPF-Z maps")

tet_fig, tet_axes = plt.subplots(1, 3, figsize=(15, 5))
# Figure 5
IPFplot.plot_multiple_ipf_slices(tet_axes, tet_micro, 
    slice_indices=[70, 100, 130], slice_direction="x"
) 

plt.tight_layout()
plt.savefig(output_dir / "mask_custom_tetrahedron.png", dpi=150, bbox_inches="tight")


print("  Custom tetrahedron mask example filename: 'mask_custom_tetrahedron.png'")


# ========================================================
# Layer (box) mask example
# ========================================================
print()
print("=" * 60)
print("5. Multilayer (box) example:")
print("-" * 60)

# Reassign microstructure
layer_micro = micro.copy()
layer1_texture = "s"
layer2_texture = "p"

# Get grains in sphere mask
layer1_grains = get_grains_in_region(layer_micro, "box",
    x_min=0, x_max=200,
    y_min=0, y_max=200,
    z_min=0, z_max=50
)
layer2_grains = get_grains_in_region(layer_micro, "box",
    x_min=0, x_max=200,
    y_min=0, y_max=200,
    z_min=150, z_max=200
)

# Generate texture for the masked region
layer1_reg_texture = CubicTexture(layer1_texture, degspread=2.0)
layer1_orientations = layer1_reg_texture.generate(layer1_grains)

layer2_reg_texture = CubicTexture(layer2_texture, degspread=2.0)
layer2_orientations = layer2_reg_texture.generate(layer2_grains)

# Reassign textures for grains in spherical mask
layer_micro.orientations[layer1_grains] = layer1_orientations.orientations
layer_micro.orientations[layer2_grains] = layer2_orientations.orientations

print(f"  Layer 1 (bottom) contains {len(layer1_grains)} grains")
print(f"  Layer 2 (top) contains {len(layer2_grains)} grains")
print()

# Plot box microstructure
print("  Plotting custom layer mask orientation IPF-Z maps")

layer_fig, layer_axes = plt.subplots(1, 3, figsize=(15, 5))
# Figure 6
IPFplot.plot_multiple_ipf_slices(layer_axes, layer_micro, 
    slice_indices=[50, 100, 150], slice_direction="y"
) 

plt.tight_layout()
plt.savefig(output_dir / "mask_layers.png", dpi=150, bbox_inches="tight")


print("  Layer mask example filename: 'mask_layers.png'")

# ========================================================
# Show output location one final time
print()
print("-" * 60)
print(f"Saved all mask figures to:")
print(f"{output_dir}")

plt.show()

"""
Figure 1: Base Micro
Figure 2: Box
Figure 3: Sphere
Figure 4: Cylinder (x)
Figure 5: Cylinder (y)
Figure 6: Tetrahedron
Figure 7: Layers
"""
