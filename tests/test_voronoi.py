# synth_struct/tests/test_voronoi.py

import pytest
import numpy as np

from synth_struct.microstructure import Microstructure
from synth_struct.generators.voronoi import VoronoiGenerator


class TestVoronoiGeneratorInitialization:
    """Test suite for VoronoiGenerator initialization"""

    def test_init_basic(self):
        """Test basic initialization"""
        gen = VoronoiGenerator(num_grains=10)
        assert gen.num_grains == 10
        assert gen.seed is None
        assert gen.chunk_size == 500_000
        assert gen.seeds is None

    def test_init_with_seed(self):
        """Test initialization with random seed"""
        gen = VoronoiGenerator(num_grains=20, seed=42)
        assert gen.num_grains == 20
        assert gen.seed == 42

    def test_init_with_custom_chunk_size(self):
        """Test initialization with custom chunk size"""
        gen = VoronoiGenerator(num_grains=15, chunk_size=100_000)
        assert gen.chunk_size == 100_000

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters"""
        gen = VoronoiGenerator(num_grains=25, seed=123, chunk_size=250_000)
        assert gen.num_grains == 25
        assert gen.seed == 123
        assert gen.chunk_size == 250_000


class TestVoronoiGenerator2D:
    """Test suite for 2D Voronoi generation"""

    @pytest.fixture
    def micro_2d_small(self):
        """Create a small 2D microstructure"""
        return Microstructure(dimensions=(50, 50), resolution=1.0)

    @pytest.fixture
    def micro_2d_medium(self):
        """Create a medium 2D microstructure"""
        return Microstructure(dimensions=(100, 100), resolution=1.0)

    @pytest.fixture
    def micro_2d_large(self):
        """Create a large 2D microstructure"""
        return Microstructure(dimensions=(200, 200), resolution=1.0)

    def test_generate_2d_basic(self, micro_2d_small):
        """Test basic 2D Voronoi generation"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d_small)

        # Check that grain IDs are assigned
        assert micro_2d_small.grain_ids.shape == (50, 50)
        assert micro_2d_small.grain_ids.dtype == np.int32

        # Check that all grains are represented
        unique_grains = np.unique(micro_2d_small.grain_ids)
        assert len(unique_grains) == 10
        assert np.all(unique_grains == np.arange(1, 11))

    def test_generate_2d_correct_num_grains(self, micro_2d_medium):
        """Test that correct number of grains are generated"""
        gen = VoronoiGenerator(num_grains=25, seed=42)
        gen.generate(micro_2d_medium)

        unique_grains = np.unique(micro_2d_medium.grain_ids)
        assert len(unique_grains) == 25
        assert micro_2d_medium.num_grains == 25

    def test_generate_2d_no_background(self, micro_2d_small):
        """Test that no background (ID=0) pixels exist"""
        gen = VoronoiGenerator(num_grains=15, seed=42)
        gen.generate(micro_2d_small)

        # Voronoi should fill entire space - no background
        assert 0 not in micro_2d_small.grain_ids

    def test_generate_2d_all_voxels_assigned(self, micro_2d_small):
        """Test that all voxels are assigned to a grain"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d_small)

        # All voxels should have a grain ID >= 1
        assert np.all(micro_2d_small.grain_ids >= 1)
        assert np.all(micro_2d_small.grain_ids <= 10)

    def test_generate_2d_reproducibility(self, micro_2d_medium):
        """Test that same seed produces same results"""
        gen1 = VoronoiGenerator(num_grains=20, seed=42)
        micro1 = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen1.generate(micro1)

        gen2 = VoronoiGenerator(num_grains=20, seed=42)
        micro2 = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen2.generate(micro2)

        # Should be identical
        assert np.array_equal(micro1.grain_ids, micro2.grain_ids)
        assert np.array_equal(gen1.seeds, gen2.seeds)

    def test_generate_2d_different_seeds(self, micro_2d_medium):
        """Test that different seeds produce different results"""
        gen1 = VoronoiGenerator(num_grains=20, seed=42)
        micro1 = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen1.generate(micro1)

        gen2 = VoronoiGenerator(num_grains=20, seed=123)
        micro2 = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen2.generate(micro2)

        # Should be different
        assert not np.array_equal(micro1.grain_ids, micro2.grain_ids)
        assert not np.array_equal(gen1.seeds, gen2.seeds)

    def test_generate_2d_single_grain(self, micro_2d_small):
        """Test generation with single grain"""
        gen = VoronoiGenerator(num_grains=1, seed=42)
        gen.generate(micro_2d_small)

        # All voxels should be grain 1
        assert np.all(micro_2d_small.grain_ids == 1)
        assert micro_2d_small.num_grains == 1

    def test_generate_2d_many_grains(self, micro_2d_medium):
        """Test generation with many grains"""
        gen = VoronoiGenerator(num_grains=100, seed=42)
        gen.generate(micro_2d_medium)

        unique_grains = np.unique(micro_2d_medium.grain_ids)
        assert len(unique_grains) == 100

    def test_seeds_stored_2d(self, micro_2d_small):
        """Test that seed coordinates are stored"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d_small)

        assert gen.seeds is not None
        assert gen.seeds.shape == (10, 2)  # 10 seeds, 2D coordinates

    def test_get_seed_coordinates_2d(self, micro_2d_small):
        """Test get_seed_coordinates method"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d_small)

        seeds = gen.get_seed_coordinates()
        assert seeds is not None
        assert seeds.shape == (10, 2)
        assert np.array_equal(seeds, gen.seeds)


class TestVoronoiGenerator3D:
    """Test suite for 3D Voronoi generation"""

    @pytest.fixture
    def micro_3d_small(self):
        """Create a small 3D microstructure"""
        return Microstructure(dimensions=(30, 30, 30), resolution=1.0)

    @pytest.fixture
    def micro_3d_medium(self):
        """Create a medium 3D microstructure"""
        return Microstructure(dimensions=(50, 50, 50), resolution=1.0)

    def test_generate_3d_basic(self, micro_3d_small):
        """Test basic 3D Voronoi generation"""
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro_3d_small)

        # Check that grain IDs are assigned
        assert micro_3d_small.grain_ids.shape == (30, 30, 30)
        assert micro_3d_small.grain_ids.dtype == np.int32

        # Check that all grains are represented
        unique_grains = np.unique(micro_3d_small.grain_ids)
        assert len(unique_grains) == 20

    def test_generate_3d_correct_num_grains(self, micro_3d_medium):
        """Test that correct number of grains are generated in 3D"""
        gen = VoronoiGenerator(num_grains=50, seed=42)
        gen.generate(micro_3d_medium)

        unique_grains = np.unique(micro_3d_medium.grain_ids)
        assert len(unique_grains) == 50
        assert micro_3d_medium.num_grains == 50

    def test_generate_3d_no_background(self, micro_3d_small):
        """Test that no background exists in 3D"""
        gen = VoronoiGenerator(num_grains=25, seed=42)
        gen.generate(micro_3d_small)

        assert 0 not in micro_3d_small.grain_ids

    def test_generate_3d_all_voxels_assigned(self, micro_3d_small):
        """Test that all voxels are assigned in 3D"""
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro_3d_small)

        assert np.all(micro_3d_small.grain_ids >= 1)
        assert np.all(micro_3d_small.grain_ids <= 20)

    def test_generate_3d_reproducibility(self, micro_3d_small):
        """Test reproducibility in 3D"""
        gen1 = VoronoiGenerator(num_grains=15, seed=42)
        micro1 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen1.generate(micro1)

        gen2 = VoronoiGenerator(num_grains=15, seed=42)
        micro2 = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        gen2.generate(micro2)

        assert np.array_equal(micro1.grain_ids, micro2.grain_ids)
        assert np.array_equal(gen1.seeds, gen2.seeds)

    def test_seeds_stored_3d(self, micro_3d_small):
        """Test that seed coordinates are stored in 3D"""
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro_3d_small)

        assert gen.seeds is not None
        assert gen.seeds.shape == (20, 3)  # 20 seeds, 3D coordinates

    def test_get_seed_coordinates_3d(self, micro_3d_small):
        """Test get_seed_coordinates method in 3D"""
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro_3d_small)

        seeds = gen.get_seed_coordinates()
        assert seeds is not None
        assert seeds.shape == (20, 3)
        assert np.array_equal(seeds, gen.seeds)


class TestVoronoiGeneratorArrayAllocation:
    """Test that proper arrays are allocated after generation"""

    @pytest.fixture
    def micro_2d(self):
        return Microstructure(dimensions=(50, 50), resolution=1.0)

    @pytest.fixture
    def micro_3d(self):
        return Microstructure(dimensions=(30, 30, 30), resolution=1.0)

    def test_orientations_allocated_2d(self, micro_2d):
        """Test that orientations array is allocated in 2D"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d)

        assert hasattr(micro_2d, "orientations")
        assert micro_2d.orientations.shape == (11, 3)  # num_grains + 1 (background)
        assert micro_2d.orientations.dtype == np.float64
        assert np.all(micro_2d.orientations == 0)  # Initially zeros

    def test_stiffness_allocated_2d(self, micro_2d):
        """Test that stiffness array is allocated in 2D"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d)

        assert hasattr(micro_2d, "stiffness")
        assert micro_2d.stiffness.shape == (11, 6, 6)
        assert micro_2d.stiffness.dtype == np.float32
        assert np.all(micro_2d.stiffness == 0)

    def test_phase_allocated_2d(self, micro_2d):
        """Test that phase array is allocated in 2D"""
        gen = VoronoiGenerator(num_grains=10, seed=42)
        gen.generate(micro_2d)

        assert hasattr(micro_2d, "phase")
        assert micro_2d.phase.shape == (11,)
        assert micro_2d.phase.dtype == np.int8
        assert np.all(micro_2d.phase == 0)

    def test_all_arrays_allocated_3d(self, micro_3d):
        """Test that all arrays are allocated in 3D"""
        gen = VoronoiGenerator(num_grains=25, seed=42)
        gen.generate(micro_3d)

        assert hasattr(micro_3d, "orientations")
        assert hasattr(micro_3d, "stiffness")
        assert hasattr(micro_3d, "phase")

        assert micro_3d.orientations.shape == (26, 3)
        assert micro_3d.stiffness.shape == (26, 6, 6)
        assert micro_3d.phase.shape == (26,)

    def test_background_index_reserved(self, micro_2d):
        """Test that index 0 is reserved for background"""
        gen = VoronoiGenerator(num_grains=5, seed=42)
        gen.generate(micro_2d)

        # Arrays should have size num_grains + 1
        assert len(micro_2d.orientations) == 6
        assert len(micro_2d.phase) == 6

        # Index 0 is for background (even though Voronoi has no background pixels)
        # Indices 1-5 are for grains 1-5


class TestVoronoiGeneratorChunking:
    """Test chunking behavior for memory efficiency"""

    def test_small_chunk_size(self):
        """Test generation with small chunk size"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        gen = VoronoiGenerator(num_grains=10, seed=42, chunk_size=500)
        gen.generate(micro)

        # Should still produce valid result
        assert np.all(micro.grain_ids >= 1)
        assert len(np.unique(micro.grain_ids)) == 10

    def test_chunk_size_larger_than_domain(self):
        """Test with chunk size larger than total voxels"""
        micro = Microstructure(dimensions=(20, 20), resolution=1.0)
        gen = VoronoiGenerator(num_grains=5, seed=42, chunk_size=1_000_000)
        gen.generate(micro)

        # Should work fine with oversized chunk
        assert len(np.unique(micro.grain_ids)) == 5

    def test_chunk_size_equals_domain(self):
        """Test with chunk size exactly equal to domain size"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        total_voxels = 50 * 50
        gen = VoronoiGenerator(num_grains=10, seed=42, chunk_size=total_voxels)
        gen.generate(micro)

        assert len(np.unique(micro.grain_ids)) == 10

    def test_different_chunk_sizes_same_result(self):
        """Test that different chunk sizes produce same result"""
        # Generate with default chunk size
        micro1 = Microstructure(dimensions=(60, 60), resolution=1.0)
        gen1 = VoronoiGenerator(num_grains=15, seed=42, chunk_size=500_000)
        gen1.generate(micro1)

        # Generate with small chunk size
        micro2 = Microstructure(dimensions=(60, 60), resolution=1.0)
        gen2 = VoronoiGenerator(num_grains=15, seed=42, chunk_size=1000)
        gen2.generate(micro2)

        # Results should be identical regardless of chunk size
        assert np.array_equal(micro1.grain_ids, micro2.grain_ids)


class TestVoronoiGeneratorGrainSizes:
    """Test grain size distributions"""

    def test_grain_size_variation(self):
        """Test that grains have varying sizes"""
        micro = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro)

        # Calculate grain sizes
        grain_sizes = []
        for grain_id in range(1, 21):
            size = np.sum(micro.grain_ids == grain_id)
            grain_sizes.append(size)

        # Grains should have different sizes (not all equal)
        assert len(set(grain_sizes)) > 1

        # All grains should have at least 1 voxel
        assert all(size > 0 for size in grain_sizes)

    def test_all_grains_have_voxels(self):
        """Test that every grain has at least one voxel"""
        micro = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen = VoronoiGenerator(num_grains=30, seed=42)
        gen.generate(micro)

        for grain_id in range(1, 31):
            count = np.sum(micro.grain_ids == grain_id)
            assert count > 0, f"Grain {grain_id} has no voxels"

    def test_total_voxels_sum(self):
        """Test that grain voxels sum to total voxels"""
        micro = Microstructure(dimensions=(80, 80), resolution=1.0)
        gen = VoronoiGenerator(num_grains=25, seed=42)
        gen.generate(micro)

        total_voxels = 80 * 80
        sum_grain_voxels = 0
        for grain_id in range(1, 26):
            sum_grain_voxels += np.sum(micro.grain_ids == grain_id)

        assert sum_grain_voxels == total_voxels


class TestVoronoiGeneratorEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_microstructure(self):
        """Test with very small microstructure"""
        micro = Microstructure(dimensions=(5, 5), resolution=1.0)
        gen = VoronoiGenerator(num_grains=3, seed=42)
        gen.generate(micro)

        assert micro.grain_ids.shape == (5, 5)
        assert len(np.unique(micro.grain_ids)) == 3

    def test_rectangular_2d_microstructure(self):
        """Test with non-square 2D microstructure"""
        micro = Microstructure(dimensions=(100, 50), resolution=1.0)
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro)

        assert micro.grain_ids.shape == (100, 50)
        assert len(np.unique(micro.grain_ids)) == 20

    def test_rectangular_3d_microstructure(self):
        """Test with non-cubic 3D microstructure"""
        micro = Microstructure(dimensions=(40, 30, 20), resolution=1.0)
        gen = VoronoiGenerator(num_grains=15, seed=42)
        gen.generate(micro)

        assert micro.grain_ids.shape == (40, 30, 20)
        assert len(np.unique(micro.grain_ids)) == 15

    def test_more_grains_than_voxels_2d(self):
        """Test with more grains than voxels (edge case)"""
        micro = Microstructure(dimensions=(5, 5), resolution=1.0)
        # 25 voxels, 30 grains requested
        gen = VoronoiGenerator(num_grains=30, seed=42)
        gen.generate(micro)

        # Some grains may not have any voxels assigned
        unique_grains = np.unique(micro.grain_ids)
        # Should have at most 25 unique grains (number of voxels)
        assert len(unique_grains) <= 25


class TestVoronoiGeneratorSeedBounds:
    """Test that seeds are within domain bounds"""

    def test_seeds_within_bounds_2d(self):
        """Test that all seeds are within microstructure bounds in 2D"""
        micro = Microstructure(dimensions=(100, 100), resolution=1.0)
        gen = VoronoiGenerator(num_grains=20, seed=42)
        gen.generate(micro)

        seeds = gen.get_seed_coordinates()
        assert np.all(seeds[:, 0] >= 0)
        assert np.all(seeds[:, 0] < 100)
        assert np.all(seeds[:, 1] >= 0)
        assert np.all(seeds[:, 1] < 100)

    def test_seeds_within_bounds_3d(self):
        """Test that all seeds are within microstructure bounds in 3D"""
        micro = Microstructure(dimensions=(50, 60, 70), resolution=1.0)
        gen = VoronoiGenerator(num_grains=25, seed=42)
        gen.generate(micro)

        seeds = gen.get_seed_coordinates()
        assert np.all(seeds[:, 0] >= 0)
        assert np.all(seeds[:, 0] < 50)
        assert np.all(seeds[:, 1] >= 0)
        assert np.all(seeds[:, 1] < 60)
        assert np.all(seeds[:, 2] >= 0)
        assert np.all(seeds[:, 2] < 70)
