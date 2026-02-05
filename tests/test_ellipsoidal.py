# synth_struct/tests/test_ellipsoidal.py

import numpy as np
import pytest
from synth_struct.microstructure import Microstructure
from synth_struct.generators.ellipsoidal import EllipsoidalGenerator


class TestEllipsoidalGenerator:

    def test_initialization_default(self):
        """Test generator initialization with default parameters"""
        gen = EllipsoidalGenerator(num_grains=10)

        assert gen.num_grains == 10
        assert gen.aspect_ratio_mean == 5.0
        assert gen.aspect_ratio_std == 0.5
        assert gen.orientation == "z"
        assert gen.base_size == 10.0
        assert gen.seed is None
        assert gen.chunk_size == 500_000

    def test_initialization_custom(self):
        """Test generator initialization with custom parameters"""
        gen = EllipsoidalGenerator(
            num_grains=20,
            aspect_ratio_mean=3.0,
            aspect_ratio_std=1.0,
            orientation="x",
            base_size=15.0,
            seed=42,
            chunk_size=100_000,
        )

        assert gen.num_grains == 20
        assert gen.aspect_ratio_mean == 3.0
        assert gen.aspect_ratio_std == 1.0
        assert gen.orientation == "x"
        assert gen.base_size == 15.0
        assert gen.seed == 42
        assert gen.chunk_size == 100_000

    def test_generate_2d(self):
        """Test 2D ellipsoidal grain generation"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=5, seed=42)

        gen.generate(micro)

        # Check that grains were assigned
        assert micro.num_grains >= 1
        assert micro.num_grains <= 5
        assert np.any(micro.grain_ids > 0)

    def test_generate_3d(self):
        """Test 3D ellipsoidal grain generation"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=10, seed=42)

        gen.generate(micro)

        # Check that grains were assigned
        assert micro.num_grains >= 1
        assert micro.num_grains <= 10
        assert np.any(micro.grain_ids > 0)

    def test_seed_reproducibility(self):
        """Test that using the same seed produces identical results"""
        micro1 = Microstructure(dimensions=(40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40), resolution=1.0)

        gen1 = EllipsoidalGenerator(num_grains=8, seed=123)
        gen2 = EllipsoidalGenerator(num_grains=8, seed=123)

        gen1.generate(micro1)
        gen2.generate(micro2)

        np.testing.assert_array_equal(micro1.grain_ids, micro2.grain_ids)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        micro1 = Microstructure(dimensions=(40, 40), resolution=1.0)
        micro2 = Microstructure(dimensions=(40, 40), resolution=1.0)

        gen1 = EllipsoidalGenerator(num_grains=8, seed=123)
        gen2 = EllipsoidalGenerator(num_grains=8, seed=456)

        gen1.generate(micro1)
        gen2.generate(micro2)

        assert not np.array_equal(micro1.grain_ids, micro2.grain_ids)

    @pytest.mark.parametrize("orientation", ["x", "y", "z", "random"])
    def test_orientation_options_3d(self, orientation):
        """Test different orientation options in 3D"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=5, orientation=orientation, seed=42)

        gen.generate(micro)

        assert micro.num_grains >= 1
        assert gen.rotations is not None
        assert len(gen.rotations) == 5

    @pytest.mark.parametrize("orientation", ["x", "y", "random"])
    def test_orientation_options_2d(self, orientation):
        """Test different orientation options in 2D"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=5, orientation=orientation, seed=42)

        gen.generate(micro)

        assert micro.num_grains >= 1
        assert gen.rotations is not None
        assert len(gen.rotations) == 5

    def test_scale_factors_2d(self):
        """Test that scale factors are generated correctly for 2D"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = EllipsoidalGenerator(
            num_grains=10, aspect_ratio_mean=4.0, base_size=5.0, seed=42
        )

        gen.generate(micro)

        assert gen.scale_factors is not None
        assert gen.scale_factors.shape == (10, 2)

        # Check that scale factors reflect elongation
        for i in range(10):
            # One axis should be longer than the other
            assert np.max(gen.scale_factors[i]) > np.min(gen.scale_factors[i])

    def test_scale_factors_3d(self):
        """Test that scale factors are generated correctly for 3D"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = EllipsoidalGenerator(
            num_grains=10, aspect_ratio_mean=4.0, base_size=5.0, seed=42
        )

        gen.generate(micro)

        assert gen.scale_factors is not None
        assert gen.scale_factors.shape == (10, 3)

        # Check that scale factors reflect elongation
        for i in range(10):
            # One axis should be longer than the others
            assert np.max(gen.scale_factors[i]) > np.min(gen.scale_factors[i])

    def test_aspect_ratio_clipping(self):
        """Test that aspect ratios are clipped to reasonable bounds"""
        micro = Microstructure(dimensions=(40, 40, 40), resolution=1.0)
        gen = EllipsoidalGenerator(
            num_grains=100,
            aspect_ratio_mean=5.0,
            aspect_ratio_std=10.0,  # Large std to potentially exceed bounds
            seed=42,
        )

        gen.generate(micro)

        # Calculate actual aspect ratios from scale factors
        aspect_ratios = np.max(gen.scale_factors, axis=1) / np.min(
            gen.scale_factors, axis=1
        )

        # Check that all aspect ratios are within [1.5, 10.0]
        assert np.all(aspect_ratios >= 1.5)
        assert np.all(aspect_ratios <= 10.0)

    def test_seeds_generation(self):
        """Test that seed coordinates are generated"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=15, seed=42)

        gen.generate(micro)

        assert gen.seeds is not None
        assert gen.seeds.shape == (15, 2)

        # Check that seeds are within bounds
        assert np.all(gen.seeds[:, 0] >= 0)
        assert np.all(gen.seeds[:, 0] < 50)
        assert np.all(gen.seeds[:, 1] >= 0)
        assert np.all(gen.seeds[:, 1] < 50)

    def test_rotation_matrices_3d(self):
        """Test that rotation matrices are valid for 3D"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=5, orientation="random", seed=42)

        gen.generate(micro)

        # Check that all rotation matrices are orthogonal (R^T * R = I)
        for R in gen.rotations:
            assert R.shape == (3, 3)
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=5)

    def test_rotation_matrices_2d(self):
        """Test that rotation matrices are valid for 2D"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=5, orientation="random", seed=42)

        gen.generate(micro)

        # Check that all rotation matrices are orthogonal
        for R in gen.rotations:
            assert R.shape == (2, 2)
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=5)

    def test_all_voxels_assigned(self):
        """Test that all voxels are assigned to grains"""
        micro = Microstructure(dimensions=(30, 30), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=10, seed=42)

        gen.generate(micro)

        # All voxels should be assigned (no zeros remaining)
        assert np.all(micro.grain_ids > 0)

    @pytest.mark.parametrize("num_grains", [1, 5, 10, 20])
    def test_varying_grain_counts(self, num_grains):
        """Test generation with different numbers of grains"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        gen = EllipsoidalGenerator(num_grains=num_grains, seed=42)

        gen.generate(micro)

        assert micro.num_grains >= 1
        assert micro.num_grains <= num_grains

    def test_base_size_effect(self):
        """Test that base_size affects grain dimensions"""
        micro1 = Microstructure(dimensions=(50, 50), resolution=1.0)
        micro2 = Microstructure(dimensions=(50, 50), resolution=1.0)

        gen1 = EllipsoidalGenerator(num_grains=5, base_size=5.0, seed=42)
        gen2 = EllipsoidalGenerator(num_grains=5, base_size=15.0, seed=42)

        gen1.generate(micro1)
        gen2.generate(micro2)

        # Larger base_size should result in larger scale factors
        assert np.mean(gen2.scale_factors) > np.mean(gen1.scale_factors)
