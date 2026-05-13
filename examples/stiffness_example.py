# synth-struct/examples/stiffness_example.py

"""
This example creates a microstructure with a texture and computes
the stiffness tensor for each grain in the microstructure.

While there are no outputs for this example, it serves as a 
demonstration for how to use a stiffness generator.
"""

from synth_struct import (
    Microstructure,
    Phase,
    get_grains_in_region,
    VoronoiGenerator,
    Texture,
    RandomTexture,
    CubicTexture,
    CubicStiffnessGenerator,
)

def main():
    print("=" * 17, "Stiffness Tensor Example", "=" * 17)

    # Microstructure parameters
    dims = (100, 100, 100)
    resolution = 1.0
    num_grains = 150
    
    # Initialize microstructure
    micro = Microstructure(
        dimensions=dims, 
        resolution=resolution, 
        phase=Phase.from_preset("Fe-fcc")
    )
    # Generate microstructure grains
    voronoi_gen = VoronoiGenerator(num_grains=num_grains)
    voronoi_gen.generate(micro)

    # Get grain IDs for the middle of the microstructure
    middle_grains = get_grains_in_region(micro, "sphere", center=(50, 50, 50), radius=15)

    # Assign random texture to the whole microstructure
    random_texture = RandomTexture()
    random_texture = random_texture.generate(micro)
    
    # Assign cubic texture to the middle grains in the microstructure
    texture = "cube"
    middle_reg_texture = CubicTexture(texture, degspread=10)
    middle_orientations = middle_reg_texture.generate(middle_grains)
    micro.assign_texture(middle_orientations, grain_ids=middle_grains)

    # Generate stiffness tensors
    stiffness_gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
    stiffness_gen.generate(
        micro, 
        micro.orientations,
    )
    print("Stiffness tensors generated for each grain in the microstructure.")

if __name__ == "__main__":
    main()