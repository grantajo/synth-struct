# synth_struct/tests/test_columnar.py

import numpy as np
import pytest
from src.microstructure import Microstructure
from src.generators.columnar import ColumnarGenerator

class TestColumnarGenerator:
    
    def test_initialization_default(self):
        """Test generator initialization with default parameters"""
        gen = ColumnarGenerator(num_grains=10)
        
        assert gen.num_grains == 10
        assert gen.axis == 'z'
        assert gen.aspect_ratio == 5.0
        assert gen.base_size == 8.0
        assert gen.size_variation == 0.2
        assert gen.seed is None
        assert gen.chunk_size == 500_000
        
    def test_initialization_custom(self):
        """Test generator initialization with custom parameters"""
        gen = ColumnarGenerator(
            num_grains=20,
            axis='x',
            aspect_ratio=10.0,
            base_size=12.0,
            size_variation=0.3,
            seed=42,
            chunk_size=100_000
        )
        
        assert gen.num_grains == 20
        assert gen.axis == 'x'
        assert gen.aspect_ratio == 10.0
        assert gen.base_size == 12.0
        assert gen.size_variation == 0.3
        assert gen.seed == 42
        assert gen.chunk_size == 100_000
        
    def test_invalid_axis(self):
        """Test that invalid axis raises ValueError"""
        with pytest.raises(ValueError, match="Invalid axis"):
            ColumnarGenerator(num_grains=10, axis='w')
            
    def test_axis_case_insensitive(self):
        """Test that axis is case-insensitive"""
        gen_upper = ColumnarGenerator(num_grains=10, axis='X')
        gen_lower = ColumnarGenerator(num_grains=10, axis='x')
        
        assert gen_upper.axis == 'x'
        assert gen_lower.axis == 'x'
        
    def test_generate_3d(self):
        """Test 3D columnar grain generation"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = ColumnarGenerator(num_grains=10, seed=42)
        
        gen.generate(micro)
        
        # Check that grains were assigned
        assert micro.num_grains >= 1
        assert micro.num_grains <= 10
        assert np.any(micro.grain_ids > 0)
        
    def test_2d_raises_error(self):
        """Test that 2D microstructure raises ValueError"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = ColumnarGenerator(num_grains=10)
        
        with pytest.raises(ValueError, match="only supported for 3D"):
            gen.generate(micro)
            
    def test_seed_reproducibility(self):
        """Test that using the same seed produces identical results"""
        micro1 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        micro2 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        
        gen1 = ColumnarGenerator(num_grains=8, seed=123)
        gen2 = ColumnarGenerator(num_grains=8, seed=123)
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        np.testing.assert_array_equal(micro1.grain_ids, micro2.grain_ids)
        
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        micro1 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        micro2 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        
        gen1 = ColumnarGenerator(num_grains=8, seed=123)
        gen2 = ColumnarGenerator(num_grains=8, seed=456)
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        assert not np.array_equal(micro1.grain_ids, micro2.grain_ids)
        
    @pytest.mark.parametrize("axis", ['x', 'y', 'z'])
    def test_axis_options(self, axis):
        """Test different axis options"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=5, axis=axis, seed=42)
        
        gen.generate(micro)
        
        assert micro.num_grains >= 1
        assert gen.rotations is not None
        assert len(gen.rotations) == 5
        
    def test_scale_factors_generation(self):
        """Test that scale factors are generated correctly"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(
            num_grains=10,
            aspect_ratio=6.0,
            base_size=5.0,
            seed=42
        )
        
        gen.generate(micro)
        
        assert gen.scale_factors is not None
        assert gen.scale_factors.shape == (10, 3)
        
        # Check that scale factors reflect columnar elongation
        for i in range(10):
            # One axis should be significantly longer than the others
            max_scale = np.max(gen.scale_factors[i])
            min_scale = np.min(gen.scale_factors[i])
            assert max_scale > min_scale * 2  # At least 2x difference
            
    def test_rotation_matrices_z_axis(self):
        """Test rotation matrices for z-axis (identity)"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=5, axis='z', seed=42)
        
        gen.generate(micro)
        
        # All rotation matrices should be identity for z-axis
        for R in gen.rotations:
            np.testing.assert_array_almost_equal(R, np.eye(3))
            
    def test_rotation_matrices_x_axis(self):
        """Test rotation matrices for x-axis"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=5, axis='x', seed=42)
        
        gen.generate(micro)
        
        # Check that all rotation matrices are the same (z to x rotation)
        expected_R = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        
        for R in gen.rotations:
            np.testing.assert_array_almost_equal(R, expected_R)
            
    def test_rotation_matrices_y_axis(self):
        """Test rotation matrices for y-axis"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=5, axis='y', seed=42)
        
        gen.generate(micro)
        
        # Check that all rotation matrices are the same (z to y rotation)
        expected_R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0,-1, 0]
        ])
        
        for R in gen.rotations:
            np.testing.assert_array_almost_equal(R, expected_R)
            
    def test_rotation_matrices_orthogonal(self):
        """Test that all rotation matrices are orthogonal"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=5, axis='x', seed=42)
        
        gen.generate(micro)
        
        # Check that all rotation matrices are orthogonal (R^T * R = I)
        for R in gen.rotations:
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)
            
    def test_size_variation(self):
        """Test that size variation affects grain dimensions"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = ColumnarGenerator(
            num_grains=20,
            base_size=10.0,
            size_variation=0.3,
            seed=42
        )
        
        gen.generate(micro)
        
        # Check that scale factors show variation
        scale_std = np.std(gen.scale_factors, axis=0)
        assert np.any(scale_std > 0)  # Should have variation
        
    def test_zero_size_variation(self):
        """Test with zero size variation (all grains same size)"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(
            num_grains=10,
            base_size=10.0,
            aspect_ratio=5.0,
            size_variation=0.0,
            seed=42
        )
        
        gen.generate(micro)
        
        # With zero variation and same seed, short axes should be identical
        short_axes = np.sort(gen.scale_factors, axis=1)[:, :2]
        # All short axes should be close to base_size
        assert np.all(np.abs(short_axes - 10.0) < 0.1)
        
    def test_aspect_ratio_effect(self):
        """Test that aspect_ratio affects grain elongation"""
        micro1 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        
        gen1 = ColumnarGenerator(
            num_grains=5,
            aspect_ratio=3.0,
            base_size=10.0,
            size_variation=0.0,
            seed=42
        )
        gen2 = ColumnarGenerator(
            num_grains=5,
            aspect_ratio=8.0,
            base_size=10.0,
            size_variation=0.0,
            seed=42
        )
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        # Higher aspect ratio should result in longer columns
        max1 = np.max(gen1.scale_factors, axis=1)
        max2 = np.max(gen2.scale_factors, axis=1)
        
        assert np.mean(max2) > np.mean(max1)
        
    def test_seeds_generation(self):
        """Test that seed coordinates are generated"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = ColumnarGenerator(num_grains=15, seed=42)
        
        gen.generate(micro)
        
        assert gen.seeds is not None
        assert gen.seeds.shape == (15, 3)
        
        # Check that seeds are within bounds
        for dim in range(3):
            assert np.all(gen.seeds[:, dim] >= 0)
            assert np.all(gen.seeds[:, dim] < 40)
            
    def test_all_voxels_assigned(self):
        """Test that all voxels are assigned to grains"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=10, seed=42)
        
        gen.generate(micro)
        
        # All voxels should be assigned (no zeros remaining)
        assert np.all(micro.grain_ids > 0)
        
    @pytest.mark.parametrize("num_grains", [1, 5, 10, 20])
    def test_varying_grain_counts(self, num_grains):
        """Test generation with different numbers of grains"""
        micro = Microstructure(dimensions=(35, 35, 35), resolution=1.0)
        gen = ColumnarGenerator(num_grains=num_grains, seed=42)
        
        gen.generate(micro)
        
        assert micro.num_grains >= 1
        assert micro.num_grains <= num_grains
        
    def test_base_size_effect(self):
        """Test that base_size affects grain dimensions"""
        micro1 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        
        gen1 = ColumnarGenerator(
            num_grains=5,
            base_size=5.0,
            size_variation=0.0,
            seed=42
        )
        gen2 = ColumnarGenerator(
            num_grains=5,
            base_size=15.0,
            size_variation=0.0,
            seed=42
        )
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        # Larger base_size should result in larger scale factors
        assert np.mean(gen2.scale_factors) > np.mean(gen1.scale_factors)
        
    def test_columnar_shape_preservation(self):
        """Test that columns maintain consistent cross-section"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(
            num_grains=5,
            aspect_ratio=5.0,
            base_size=10.0,
            size_variation=0.0,
            seed=42
        )
        
        gen.generate(micro)
        
        # For each grain, two dimensions should be similar (short axes)
        # and one should be much longer (long axis)
        for i in range(5):
            sorted_scales = np.sort(gen.scale_factors[i])
            # First two should be similar (cross-section)
            assert np.abs(sorted_scales[0] - sorted_scales[1]) < 1e-5
            # Third should be much larger (length)
            assert sorted_scales[2] > sorted_scales[1] * 3
            
    def test_num_grains_stored(self):
        """Test that num_grains is properly calculated from grain_ids"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = ColumnarGenerator(num_grains=12, seed=42)
        
        gen.generate(micro)
        
        # num_grains should be calculated from grain_ids, not set directly
        # It may be <= 12 since some grains might not appear in the final structure
        assert hasattr(micro, 'num_grains')
        assert micro.num_grains >= 1
        assert micro.num_grains <= 12
