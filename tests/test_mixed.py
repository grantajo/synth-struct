# synth_struct/tests/test_mixed.py

import numpy as np
import pytest
from synth_struct.microstructure import Microstructure
from synth_struct.generators.mixed import MixedGenerator


class TestMixedGenerator:
    
    def test_initialization_default(self):
        """Test generator initialization with default parameters"""
        gen = MixedGenerator(num_grains=10)
        
        assert gen.num_grains == 10
        assert gen.ellipsoid_fraction == 0.5
        assert gen.aspect_ratio_mean == 5.0
        assert gen.aspect_ratio_std == 0.5
        assert gen.base_size == 10.0
        assert gen.seed is None
        assert gen.chunk_size == 500_000
        
    def test_initialization_custom(self):
        """Test generator initialization with custom parameters"""
        gen = MixedGenerator(
            num_grains=20,
            ellipsoid_fraction=0.7,
            aspect_ratio_mean=3.0,
            aspect_ratio_std=1.0,
            base_size=15.0,
            seed=42,
            chunk_size=100_000
        )
        
        assert gen.num_grains == 20
        assert gen.ellipsoid_fraction == 0.7
        assert gen.aspect_ratio_mean == 3.0
        assert gen.aspect_ratio_std == 1.0
        assert gen.base_size == 15.0
        assert gen.seed == 42
        assert gen.chunk_size == 100_000
        
    def test_ellipsoid_fraction_clipping(self):
        """Test that ellipsoid_fraction is clipped to [0, 1]"""
        gen1 = MixedGenerator(num_grains=10, ellipsoid_fraction=-0.5)
        gen2 = MixedGenerator(num_grains=10, ellipsoid_fraction=1.5)
        
        assert gen1.ellipsoid_fraction == 0.0
        assert gen2.ellipsoid_fraction == 1.0
        
    def test_generate_2d(self):
        """Test 2D mixed grain generation"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = MixedGenerator(num_grains=10, seed=42)
        
        gen.generate(micro)
        
        # Check that grains were assigned
        assert micro.num_grains >= 1
        assert micro.num_grains <= 10
        assert np.any(micro.grain_ids > 0)
        
    def test_generate_3d(self):
        """Test 3D mixed grain generation"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(num_grains=10, seed=42)
        
        gen.generate(micro)
        
        # Check that grains were assigned
        assert micro.num_grains >= 1
        assert micro.num_grains <= 10
        assert np.any(micro.grain_ids > 0)
        
    def test_seed_reproducibility(self):
        """Test that using the same seed produces identical results"""
        micro1 = Microstructure(dimensions=(40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40), resolution=1.0)
        
        gen1 = MixedGenerator(num_grains=8, seed=123)
        gen2 = MixedGenerator(num_grains=8, seed=123)
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        np.testing.assert_array_equal(micro1.grain_ids, micro2.grain_ids)
        
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        micro1 = Microstructure(dimensions=(40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40), resolution=1.0)
        
        gen1 = MixedGenerator(num_grains=8, seed=123)
        gen2 = MixedGenerator(num_grains=8, seed=456)
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        assert not np.array_equal(micro1.grain_ids, micro2.grain_ids)
        
    def test_all_ellipsoidal(self):
        """Test generation with all ellipsoidal grains"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=1.0,
            seed=42
        )
        
        gen.generate(micro)
        
        assert micro.num_grains >= 1
        assert gen.scale_factors is not None
        
        # All grains should have elongated shapes
        for i in range(10):
            assert np.max(gen.scale_factors[i]) > np.min(gen.scale_factors[i])
            
    def test_all_equiaxed(self):
        """Test generation with all equiaxed grains"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.0,
            seed=42
        )
        
        gen.generate(micro)
        
        assert micro.num_grains >= 1
        assert gen.scale_factors is not None
        
        # All grains should have equal scale factors
        for i in range(10):
            np.testing.assert_array_almost_equal(
                gen.scale_factors[i],
                np.ones(2) * gen.base_size
            )
            
    def test_mixed_50_50(self):
        """Test generation with 50/50 mix"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.5,
            seed=42
        )
        
        gen.generate(micro)
        
        # Check that we have both types
        num_ellipsoidal = 0
        num_equiaxed = 0
        
        for i in range(10):
            max_scale = np.max(gen.scale_factors[i])
            min_scale = np.min(gen.scale_factors[i])
            
            if max_scale > min_scale * 1.2:  # Ellipsoidal (with some tolerance)
                num_ellipsoidal += 1
            else:  # Equiaxed
                num_equiaxed += 1
                
        assert num_ellipsoidal == 5
        assert num_equiaxed == 5
        
    def test_scale_factors_2d(self):
        """Test that scale factors are generated correctly for 2D"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.5,
            seed=42
        )
        
        gen.generate(micro)
        
        assert gen.scale_factors is not None
        assert gen.scale_factors.shape == (10, 2)
        
    def test_scale_factors_3d(self):
        """Test that scale factors are generated correctly for 3D"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.5,
            seed=42
        )
        
        gen.generate(micro)
        
        assert gen.scale_factors is not None
        assert gen.scale_factors.shape == (10, 3)
        
    def test_aspect_ratio_clipping(self):
        """Test that aspect ratios are clipped to reasonable bounds"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=100,
            ellipsoid_fraction=1.0,  # All ellipsoidal
            aspect_ratio_mean=5.0,
            aspect_ratio_std=20.0,  # Large std to potentially exceed bounds
            seed=42
        )
        
        gen.generate(micro)
        
        # Calculate actual aspect ratios from scale factors
        aspect_ratios = np.max(gen.scale_factors, axis=1) / np.min(gen.scale_factors, axis=1)
        
        # Check that all aspect ratios are within [1.5, 15.0]
        assert np.all(aspect_ratios >= 1.4)  # Slight tolerance
        assert np.all(aspect_ratios <= 15.1)
        
    def test_rotation_matrices_2d(self):
        """Test that rotation matrices are valid for 2D"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(num_grains=10, ellipsoid_fraction=1.0, seed=42)
        
        gen.generate(micro)
        
        # Check that all rotation matrices are orthogonal
        for R in gen.rotations:
            assert R.shape == (2, 2)
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=5)
            
    def test_rotation_matrices_3d(self):
        """Test that rotation matrices are valid for 3D"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(num_grains=10, ellipsoid_fraction=1.0, seed=42)
        
        gen.generate(micro)
        
        # Check that all rotation matrices are orthogonal
        for R in gen.rotations:
            assert R.shape == (3, 3)
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=5)
            
    def test_equiaxed_identity_rotation(self):
        """Test that equiaxed grains use identity rotation"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.0,  # All equiaxed
            seed=42
        )
        
        gen.generate(micro)
        
        # All rotation matrices should be identity for equiaxed grains
        for R in gen.rotations:
            np.testing.assert_array_almost_equal(R, np.eye(3))
            
    def test_ellipsoidal_random_rotation(self):
        """Test that ellipsoidal grains have varied rotations"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=1.0,  # All ellipsoidal
            seed=42
        )
        
        gen.generate(micro)
        
        # Not all rotation matrices should be the same
        all_same = True
        first_R = gen.rotations[0]
        
        for R in gen.rotations[1:]:
            if not np.allclose(R, first_R):
                all_same = False
                break
                
        assert not all_same, "Expected varied rotations for ellipsoidal grains"
        
    def test_seeds_generation(self):
        """Test that seed coordinates are generated"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = MixedGenerator(num_grains=15, seed=42)
        
        gen.generate(micro)
        
        assert gen.seeds is not None
        assert gen.seeds.shape == (15, 2)
        
        # Check that seeds are within bounds
        assert np.all(gen.seeds[:, 0] >= 0)
        assert np.all(gen.seeds[:, 0] < 50)
        assert np.all(gen.seeds[:, 1] >= 0)
        assert np.all(gen.seeds[:, 1] < 50)
        
    def test_all_voxels_assigned(self):
        """Test that all voxels are assigned to grains"""
        micro = Microstructure(dimensions=(30, 30), resolution=1.0)
        gen = MixedGenerator(num_grains=10, seed=42)
        
        gen.generate(micro)
        
        # All voxels should be assigned (no zeros remaining)
        assert np.all(micro.grain_ids > 0)
        
    @pytest.mark.parametrize("num_grains", [1, 5, 10, 20])
    def test_varying_grain_counts(self, num_grains):
        """Test generation with different numbers of grains"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(num_grains=num_grains, seed=42)
        
        gen.generate(micro)
        
        assert micro.num_grains >= 1
        assert micro.num_grains <= num_grains
        
    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_varying_ellipsoid_fractions(self, fraction):
        """Test generation with different ellipsoid fractions"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=20,
            ellipsoid_fraction=fraction,
            seed=42
        )
        
        gen.generate(micro)
        
        expected_ellipsoidal = int(20 * fraction)
        
        # Count ellipsoidal vs equiaxed grains
        num_ellipsoidal = 0
        for i in range(20):
            max_scale = np.max(gen.scale_factors[i])
            min_scale = np.min(gen.scale_factors[i])
            
            if max_scale > min_scale * 1.2:
                num_ellipsoidal += 1
                
        assert num_ellipsoidal == expected_ellipsoidal
        
    def test_base_size_effect(self):
        """Test that base_size affects grain dimensions"""
        micro1 = Microstructure(dimensions=(50, 50), resolution=1.0)
        micro2 = Microstructure(dimensions=(50, 50), resolution=1.0)
        
        gen1 = MixedGenerator(num_grains=5, base_size=5.0, seed=42)
        gen2 = MixedGenerator(num_grains=5, base_size=15.0, seed=42)
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        # Larger base_size should result in larger scale factors
        assert np.mean(gen2.scale_factors) > np.mean(gen1.scale_factors)
        
    def test_mixed_distribution_consistency(self):
        """Test that ellipsoidal grains come first, equiaxed second"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=0.3,  # First 3 should be ellipsoidal
            aspect_ratio_mean=5.0,
            seed=42
        )
        
        gen.generate(micro)
        
        # First 3 grains should be ellipsoidal
        for i in range(3):
            max_scale = np.max(gen.scale_factors[i])
            min_scale = np.min(gen.scale_factors[i])
            assert max_scale > min_scale * 1.5
            
        # Remaining 7 should be equiaxed
        for i in range(3, 10):
            np.testing.assert_array_almost_equal(
                gen.scale_factors[i],
                np.ones(3) * gen.base_size,
                decimal=5
            )
            
    def test_odd_number_grains_with_fraction(self):
        """Test handling of odd number of grains with non-integer ellipsoid count"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = MixedGenerator(
            num_grains=11,
            ellipsoid_fraction=0.5,  # Should give 5 ellipsoidal, 6 equiaxed
            seed=42
        )
        
        gen.generate(micro)
        
        # Count actual distribution
        num_ellipsoidal = 0
        num_equiaxed = 0
        
        for i in range(11):
            max_scale = np.max(gen.scale_factors[i])
            min_scale = np.min(gen.scale_factors[i])
            
            if max_scale > min_scale * 1.2:
                num_ellipsoidal += 1
            else:
                num_equiaxed += 1
                
        assert num_ellipsoidal == 5
        assert num_equiaxed == 6
        
    def test_aspect_ratio_mean_effect(self):
        """Test that aspect_ratio_mean affects ellipsoidal grain elongation"""
        micro1 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        
        gen1 = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=1.0,
            aspect_ratio_mean=3.0,
            aspect_ratio_std=0.1,
            seed=42
        )
        gen2 = MixedGenerator(
            num_grains=10,
            ellipsoid_fraction=1.0,
            aspect_ratio_mean=8.0,
            aspect_ratio_std=0.1,
            seed=42
        )
        
        gen1.generate(micro1)
        gen2.generate(micro2)
        
        # Calculate average aspect ratios
        aspect1 = np.mean(np.max(gen1.scale_factors, axis=1) / np.min(gen1.scale_factors, axis=1))
        aspect2 = np.mean(np.max(gen2.scale_factors, axis=1) / np.min(gen2.scale_factors, axis=1))
        
        assert aspect2 > aspect1
