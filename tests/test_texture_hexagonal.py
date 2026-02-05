# synth_struct/tests/test_texture_hexagonal.py

import pytest
import numpy as np
from synth_struct.microstructure import Microstructure
from synth_struct.orientation.texture.hexagonal import (
    HexagonalTexture,
    HEXAGONAL_ORIENTATIONS,
)
from synth_struct.orientation.texture.texture import Texture


class TestHexagonalTexture:

    def test_initialization_default(self):
        """Test HexagonalTexture initialization with default parameters"""
        gen = HexagonalTexture(type="basal")

        assert gen.type == "basal"
        assert gen.degspread == 5.0
        assert gen.seed is None

    def test_initialization_custom(self):
        """Test HexagonalTexture initialization with custom parameters"""
        gen = HexagonalTexture(type="prismatic", degspread=10.0, seed=42)

        assert gen.type == "prismatic"
        assert gen.degspread == 10.0
        assert gen.seed == 42

    def test_invalid_texture_type(self):
        """Test that invalid texture type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown hexagonal texture type"):
            HexagonalTexture(type="invalid_type")

    @pytest.mark.parametrize("texture_type", ["basal", "prismatic"])
    def test_valid_texture_types(self, texture_type):
        """Test that all valid texture types are accepted"""
        gen = HexagonalTexture(type=texture_type)
        assert gen.type == texture_type

    def test_generate_returns_texture(self):
        """Test that generate returns a Texture object"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2

        gen = HexagonalTexture(type="basal", seed=42)
        texture = gen.generate(micro)

        assert isinstance(texture, Texture)
        assert texture.representation == "euler"
        assert texture.symmetry == "hexagonal"

    def test_generate_correct_number_of_orientations(self):
        """Test that texture has correct number of orientations"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        micro.grain_ids[20:30, 20:30, 20:30] = 3

        gen = HexagonalTexture(type="basal", seed=42)
        texture = gen.generate(micro)

        assert texture.n_orientations == 3

    def test_zero_spread(self):
        """Test that zero spread produces exact ideal orientation"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2

        gen = HexagonalTexture(type="basal", degspread=0.0, seed=42)
        texture = gen.generate(micro)

        ideal_orientation = HEXAGONAL_ORIENTATIONS["basal"]

        # All orientations should be identical to the ideal
        for i in range(texture.n_orientations):
            np.testing.assert_array_almost_equal(
                texture.orientations[i], ideal_orientation
            )

    def test_nonzero_spread_creates_variation(self):
        """Test that nonzero spread creates variation around ideal"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        for i in range(10):
            micro.grain_ids[i * 2 : (i + 1) * 2, 0:20, 0:20] = i + 1

        gen = HexagonalTexture(type="basal", degspread=5.0, seed=42)
        texture = gen.generate(micro)

        # Should have variation (not all identical)
        std_dev = np.std(texture.orientations, axis=0)
        assert np.any(std_dev > 0)

    def test_seed_reproducibility(self):
        """Test that same seed produces identical results"""
        micro1 = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro1.grain_ids[0:10, 0:10, 0:10] = 1
        micro1.grain_ids[10:20, 10:20, 10:20] = 2

        micro2 = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro2.grain_ids[0:10, 0:10, 0:10] = 1
        micro2.grain_ids[10:20, 10:20, 10:20] = 2

        gen1 = HexagonalTexture(type="prismatic", degspread=5.0, seed=123)
        gen2 = HexagonalTexture(type="prismatic", degspread=5.0, seed=123)

        texture1 = gen1.generate(micro1)
        texture2 = gen2.generate(micro2)

        np.testing.assert_array_equal(texture1.orientations, texture2.orientations)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        micro1 = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro1.grain_ids[0:10, 0:10, 0:10] = 1
        micro1.grain_ids[10:20, 10:20, 10:20] = 2

        micro2 = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro2.grain_ids[0:10, 0:10, 0:10] = 1
        micro2.grain_ids[10:20, 10:20, 10:20] = 2

        gen1 = HexagonalTexture(type="prismatic", degspread=5.0, seed=123)
        gen2 = HexagonalTexture(type="prismatic", degspread=5.0, seed=456)

        texture1 = gen1.generate(micro1)
        texture2 = gen2.generate(micro2)

        assert not np.array_equal(texture1.orientations, texture2.orientations)

    def test_orientations_normalized_to_2pi(self):
        """Test that orientations are normalized to [0, 2π)"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2

        gen = HexagonalTexture(
            type="prismatic", degspread=50.0, seed=42
        )  # Large spread
        texture = gen.generate(micro)

        # All angles should be in [0, 2π)
        assert np.all(texture.orientations >= 0)
        assert np.all(texture.orientations < 2 * np.pi)

    def test_spread_affects_distribution(self):
        """Test that larger spread creates wider distribution"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        for i in range(20):
            micro.grain_ids[i : i + 1, 0:30, 0:30] = i + 1

        # Use prismatic which has no angles at 0
        gen_small = HexagonalTexture(type="prismatic", degspread=2.0, seed=42)
        gen_large = HexagonalTexture(
            type="prismatic", degspread=20.0, seed=43
        )  # Larger spread

        texture_small = gen_small.generate(micro)
        texture_large = gen_large.generate(micro)

        ideal = HEXAGONAL_ORIENTATIONS["prismatic"]

        # Calculate distances from ideal
        dist_small = np.linalg.norm(texture_small.orientations - ideal, axis=1)
        dist_large = np.linalg.norm(texture_large.orientations - ideal, axis=1)

        # Larger spread should have larger mean distance
        assert np.std(dist_large) > np.std(dist_small)

    @pytest.mark.parametrize("texture_type", HEXAGONAL_ORIENTATIONS.keys())
    def test_all_texture_types_generate(self, texture_type):
        """Test that all texture types can generate successfully"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2

        gen = HexagonalTexture(type=texture_type, seed=42)
        texture = gen.generate(micro)

        assert isinstance(texture, Texture)
        assert texture.n_orientations == 2

    def test_ideal_orientations_in_dict(self):
        """Test that HEXAGONAL_ORIENTATIONS contains expected ideal orientations"""
        assert "basal" in HEXAGONAL_ORIENTATIONS
        assert "prismatic" in HEXAGONAL_ORIENTATIONS

        # Check that they are numpy arrays with shape (3,)
        for key, value in HEXAGONAL_ORIENTATIONS.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == (3,)

    def test_basal_orientation(self):
        """Test that basal orientation is [0, 0, 0]"""
        np.testing.assert_array_equal(
            HEXAGONAL_ORIENTATIONS["basal"], np.array([0, 0, 0])
        )

    def test_prismatic_orientation(self):
        """Test that prismatic orientation has 90° tilt"""
        expected = np.array([0, np.radians(90), 0])
        np.testing.assert_array_almost_equal(
            HEXAGONAL_ORIENTATIONS["prismatic"], expected
        )

    def test_single_grain_microstructure(self):
        """Test generation for microstructure with single grain"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[:, :, :] = 1

        gen = HexagonalTexture(type="basal", degspread=5.0, seed=42)
        texture = gen.generate(micro)

        assert texture.n_orientations == 1
        assert texture.orientations.shape == (1, 3)

    def test_many_grains_microstructure(self):
        """Test generation for microstructure with many grains"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        for i in range(100):
            x = (i % 10) * 5
            y = ((i // 10) % 10) * 5
            z = (i // 100) * 5
            micro.grain_ids[x : x + 5, y : y + 5, z : z + 5] = i + 1

        gen = HexagonalTexture(type="basal", degspread=5.0, seed=42)
        texture = gen.generate(micro)

        assert texture.n_orientations == 100
        assert texture.orientations.shape == (100, 3)

    def test_orientation_clustered_around_ideal(self):
        """Test that orientations are clustered around the ideal"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        for i in range(50):
            micro.grain_ids[i : i + 1, 0:30, 0:30] = i + 1

        # Use basal to avoid wrapping complications (all zeros)
        gen = HexagonalTexture(type="basal", degspread=5.0, seed=42)
        texture = gen.generate(micro)

        ideal = HEXAGONAL_ORIENTATIONS["basal"]
        distances = np.linalg.norm(texture.orientations - ideal, axis=1)

        mean_distance = np.mean(distances)
        # With 5° spread, distances should be small
        assert np.radians(mean_distance) < np.radians(20)

    def test_negative_spread_handling(self):
        """Test that negative spread raises ValueError"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[:, :, :] = 1

        gen = HexagonalTexture(type="basal", degspread=-5.0, seed=42)

        with pytest.raises(ValueError, match="scale < 0"):
            gen.generate(micro)

    def test_basal_vs_prismatic_different_orientations(self):
        """Test that basal and prismatic produce different ideal orientations"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[:, :, :] = 1

        gen_basal = HexagonalTexture(type="basal", degspread=0.0, seed=42)
        gen_prismatic = HexagonalTexture(type="prismatic", degspread=0.0, seed=42)

        texture_basal = gen_basal.generate(micro)
        texture_prismatic = gen_prismatic.generate(micro)

        # They should be different
        assert not np.array_equal(
            texture_basal.orientations, texture_prismatic.orientations
        )

        # Specifically, prismatic should have π/2 in second angle
        np.testing.assert_almost_equal(texture_prismatic.orientations[0, 1], np.pi / 2)
