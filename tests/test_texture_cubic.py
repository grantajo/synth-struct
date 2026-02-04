# synth_struct/tests/test_texture_cubic.py

import pytest
import numpy as np
from synth_struct.microstructure import Microstructure
from synth_struct.orientation.texture.cubic import CubicTexture, CUBIC_TEXTURES
from synth_struct.orientation.texture.texture import Texture


class TestCubicTexture:
    
    def test_initialization_default(self):
        """Test CubicTexture initialization with default parameters"""
        gen = CubicTexture(type='cube')
        
        assert gen.type == 'cube'
        assert gen.degspread == 5.0
        assert gen.seed is None
        
    def test_initialization_custom(self):
        """Test CubicTexture initialization with custom parameters"""
        gen = CubicTexture(type='goss', degspread=10.0, seed=42)
        
        assert gen.type == 'goss'
        assert gen.degspread == 10.0
        assert gen.seed == 42
        
    def test_invalid_texture_type(self):
        """Test that invalid texture type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown cubic texture type"):
            CubicTexture(type='invalid_type')
            
    @pytest.mark.parametrize("texture_type", [
        'cube', 'goss', 'brass', 'copper', 's', 'p', 'rotated_cube', 'rotated_goss'
    ])
    def test_valid_texture_types(self, texture_type):
        """Test that all valid texture types are accepted"""
        gen = CubicTexture(type=texture_type)
        assert gen.type == texture_type
        
    def test_generate_returns_texture(self):
        """Test that generate returns a Texture object"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        
        gen = CubicTexture(type='cube', seed=42)
        texture = gen.generate(micro)
        
        assert isinstance(texture, Texture)
        assert texture.representation == 'euler'
        assert texture.symmetry == 'cubic'
        
    def test_generate_correct_number_of_orientations(self):
        """Test that texture has correct number of orientations"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        micro.grain_ids[20:30, 20:30, 20:30] = 3
        
        gen = CubicTexture(type='cube', seed=42)
        texture = gen.generate(micro)
        
        assert texture.n_orientations == 3
        
    def test_zero_spread(self):
        """Test that zero spread produces exact ideal orientation"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        
        gen = CubicTexture(type='cube', degspread=0.0, seed=42)
        texture = gen.generate(micro)
        
        ideal_orientation = CUBIC_TEXTURES['cube']
        
        # All orientations should be identical to the ideal
        for i in range(texture.n_orientations):
            np.testing.assert_array_almost_equal(
                texture.orientations[i],
                ideal_orientation
            )
            
    def test_nonzero_spread_creates_variation(self):
        """Test that nonzero spread creates variation around ideal"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        for i in range(10):
            micro.grain_ids[i*2:(i+1)*2, 0:20, 0:20] = i + 1
        
        gen = CubicTexture(type='cube', degspread=5.0, seed=42)
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
        
        gen1 = CubicTexture(type='brass', degspread=5.0, seed=123)
        gen2 = CubicTexture(type='brass', degspread=5.0, seed=123)
        
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
        
        gen1 = CubicTexture(type='brass', degspread=5.0, seed=123)
        gen2 = CubicTexture(type='brass', degspread=5.0, seed=456)
        
        texture1 = gen1.generate(micro1)
        texture2 = gen2.generate(micro2)
        
        assert not np.array_equal(texture1.orientations, texture2.orientations)
        
    def test_orientations_normalized_to_2pi(self):
        """Test that orientations are normalized to [0, 2π)"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        
        gen = CubicTexture(type='copper', degspread=50.0, seed=42)  # Large spread
        texture = gen.generate(micro)
        
        # All angles should be in [0, 2π)
        assert np.all(texture.orientations >= 0)
        assert np.all(texture.orientations < 2 * np.pi)
        
    def test_spread_affects_distribution(self):
        """Test that larger spread creates wider distribution"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        for i in range(20):
            micro.grain_ids[i:i+1, 0:30, 0:30] = i + 1
        
        # Use 's' texture (no angles near 0 or 2π)
        gen_small = CubicTexture(type='s', degspread=2.0, seed=42)
        gen_large = CubicTexture(type='s', degspread=15.0, seed=42)
        
        texture_small = gen_small.generate(micro)
        texture_large = gen_large.generate(micro)
        
        std_small = np.std(texture_small.orientations, axis=0)
        std_large = np.std(texture_large.orientations, axis=0)
        
        # Larger spread should have larger standard deviation overall
        assert np.mean(std_large) > np.mean(std_small)
        
    @pytest.mark.parametrize("texture_type", CUBIC_TEXTURES.keys())
    def test_all_texture_types_generate(self, texture_type):
        """Test that all texture types can generate successfully"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20, 10:20] = 2
        
        gen = CubicTexture(type=texture_type, seed=42)
        texture = gen.generate(micro)
        
        assert isinstance(texture, Texture)
        assert texture.n_orientations == 2
        
    def test_ideal_orientations_in_dict(self):
        """Test that CUBIC_TEXTURES contains expected ideal orientations"""
        assert 'cube' in CUBIC_TEXTURES
        assert 'goss' in CUBIC_TEXTURES
        assert 'brass' in CUBIC_TEXTURES
        assert 'copper' in CUBIC_TEXTURES
        
        # Check that they are numpy arrays with shape (3,)
        for key, value in CUBIC_TEXTURES.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == (3,)
            
    def test_cube_orientation_is_zero(self):
        """Test that cube orientation is [0, 0, 0]"""
        np.testing.assert_array_equal(CUBIC_TEXTURES['cube'], np.array([0, 0, 0]))
        
    def test_single_grain_microstructure(self):
        """Test generation for microstructure with single grain"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[:, :, :] = 1
        
        gen = CubicTexture(type='cube', degspread=5.0, seed=42)
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
            micro.grain_ids[x:x+5, y:y+5, z:z+5] = i + 1
        
        gen = CubicTexture(type='s', degspread=5.0, seed=42)
        texture = gen.generate(micro)
        
        assert texture.n_orientations == 100
        assert texture.orientations.shape == (100, 3)
        
    def test_orientation_clustered_around_ideal(self):
        """Test that orientations are clustered around the ideal"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        for i in range(50):
            micro.grain_ids[i:i+1, 0:30, 0:30] = i + 1
        
        # Use 's' texture (centered away from boundaries)
        gen = CubicTexture(type='s', degspread=5.0, seed=42)
        texture = gen.generate(micro)
        
        ideal = CUBIC_TEXTURES['s']
        distances = np.linalg.norm(texture.orientations - ideal, axis=1)
        
        mean_distance = np.mean(distances)
        assert mean_distance < np.radians(15)
        
    def test_negative_spread_handling(self):
        """Test behavior with negative spread (should handle gracefully or error)"""
        # This depends on your intended behavior - either it should:
        # 1. Raise an error
        # 2. Use absolute value
        # 3. Treat as zero
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[:, :, :] = 1
        
        gen = CubicTexture(type='cube', degspread=-5.0, seed=42)
        
        with pytest.raises(ValueError, match='scale < 0'):
            gen.generate(micro)
        
        
