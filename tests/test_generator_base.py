# synth_struct/tests/test_generator_base.py

import numpy as np
import pytest
from src.microstructure import Microstructure
from src.generators.gen_base import MicrostructureGenerator
from src.generators.gen_utils import get_seed_coordinates, aniso_voronoi_assignment

class DummyGenerator(MicrostructureGenerator):
    """
    A dummy generator for testing MicrostructureGenerator base class
    """
    def _generate_internal(self,micro):
        # Simple implementation for testing
        micro.grain_ids = np.random.randint(1,10,micro.dimensions)
        
class TestMicrostructureGenerator:
    def test_base_generator_allocation(self):
        """
        Test that base generator correctly allocates per-grain arrays
        """
        # Create a microstructure
        micro = Microstructure(dimensions=(100, 100), resolution=0.1)
        
        # Set a dummy number of grains
        micro.grain_ids = np.random.randint(1, 10, (100, 100))
        
        # Create generator and generate
        generator = DummyGenerator()
        generator.generate(micro)
        
        # Check array allocations
        assert hasattr(micro, 'orientations'), "Orientations not allocated"
        assert hasattr(micro, 'stiffness'), "Stiffness not allocated"
        assert hasattr(micro, 'phase'), "Phase not allocated"
        
        # Check array shapes
        assert micro.orientations.shape == (micro.num_grains + 1, 3), "Incorrect orientations shape"
        assert micro.stiffness.shape == (micro.num_grains + 1, 6, 6), "Incorrect stiffness shape"
        assert micro.phase.shape == (micro.num_grains + 1,), "Incorrect phase shape"
        
        # Check data types
        assert micro.orientations.dtype == np.float64, "Incorrect orientations dtype"
        assert micro.stiffness.dtype == np.float32, "Incorrect stiffness dtype"
        assert micro.phase.dtype == np.int8, "Incorrect phase dtype"

    def test_base_generator_not_implemented(self):
        """
        Test that base generator raises NotImplementedError 
        if _generate_internal is not implemented
        """
        # Create a microstructure
        micro = Microstructure(dimensions=(100, 100), resolution=0.1)
        
        # Create base generator (which doesn't implement _generate_internal)
        generator = MicrostructureGenerator()
        
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            generator.generate(micro)

class TestGeneratorUtils:
    @pytest.mark.parametrize("dimensions,num_grains", [
        ((100, 100), 10),     # 2D case
        ((50, 50, 50), 20),   # 3D case
        ((10, 10), 5)         # Small 2D case
    ])
    def test_get_seed_coordinates(self, dimensions, num_grains):
        """
        Test seed coordinate generation
        """
        # Generate seeds
        seeds = get_seed_coordinates(num_grains, dimensions)
        
        # Check shape
        assert seeds.shape == (num_grains, len(dimensions)), "Incorrect seeds shape"
        
        # Check coordinate ranges
        for dim in range(len(dimensions)):
            assert np.all(seeds[:, dim] >= 0), f"Coordinates below 0 in dimension {dim}"
            assert np.all(seeds[:, dim] < dimensions[dim]), f"Coordinates exceed dimension {dim}"
    
    def test_get_seed_coordinates_reproducibility(self):
        """
        Test that seed generation is reproducible with same seed
        """
        # First generation
        seeds1 = get_seed_coordinates(10, (100, 100), seed=42)
        
        # Second generation with same seed
        seeds2 = get_seed_coordinates(10, (100, 100), seed=42)
        
        # Check exact equality
        np.testing.assert_array_equal(seeds1, seeds2, "Seeds not reproducible")
    
    def test_aniso_voronoi_assignment_basic(self):
        """
        Basic test for anisotropic Voronoi assignment
        """
        # Create microstructure
        micro = Microstructure(dimensions=(100, 100), resolution=0.1)
        
        # Generate seeds
        num_grains = 10
        seeds = get_seed_coordinates(num_grains, micro.dimensions)
        
        # Create scale factors and rotations
        scale_factors = np.ones((num_grains, 2))
        rotations = [np.eye(2) for _ in range(num_grains)]
        
        # Perform assignment
        aniso_voronoi_assignment(micro, seeds, scale_factors, rotations)
        
        # Checks
        assert micro.grain_ids.shape == micro.dimensions, "Incorrect grain_ids shape"
        assert np.min(micro.grain_ids) >= 1, "Grain IDs should start from 1"
        assert np.max(micro.grain_ids) <= num_grains, "Grain IDs exceed number of grains"
        
        # Verify total number of unique grains
        unique_grains = np.unique(micro.grain_ids)
        assert len(unique_grains) <= num_grains + 1, "Too many unique grain IDs"

    @pytest.mark.parametrize("dimensions", [
        (50, 50),       # 2D small
        (30, 30, 30)    # 3D small
    ])
    def test_aniso_voronoi_assignment_coverage(self, dimensions):
        """
        Test that Voronoi assignment covers entire microstructure
        """
        # Create microstructure
        micro = Microstructure(dimensions=dimensions, resolution=0.1)
        
        # Generate seeds
        num_grains = 5
        seeds = get_seed_coordinates(num_grains, micro.dimensions)
        
        # Create scale factors and rotations
        ndim = len(dimensions)
        scale_factors = np.ones((num_grains, ndim))
        rotations = [np.eye(ndim) for _ in range(num_grains)]
        
        # Perform assignment
        aniso_voronoi_assignment(micro, seeds, scale_factors, rotations)
        
        # Check coverage
        assert not np.any(micro.grain_ids == 0), "Some voxels not assigned a grain"
