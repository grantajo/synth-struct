# synth_struct/tests/test_texture_base.py

import pytest
import numpy as np
from src.orientation.texture.texture_base import TextureGenerator
from src.orientation.texture.texture import Texture
from src.microstructure import Microstructure


class ConcreteTextureGenerator(TextureGenerator):
    """Concrete implementation for testing"""
    
    def generate(self, micro):
        """Generate random Euler angles for each grain"""
        n_grains = micro.num_grains if micro.num_grains > 0 else 10
        orientations = np.random.rand(n_grains, 3) * 2 * np.pi
        
        return Texture(
            orientations=orientations,
            representation='euler',
            symmetry='cubic'
        )


class TestTextureGenerator:
    
    def test_cannot_instantiate_abstract_base(self):
        """Test that TextureGenerator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TextureGenerator()
            
    def test_concrete_implementation(self):
        """Test that concrete implementation works"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.grain_ids[0:5, 0:5, 0:5] = 1
        micro.grain_ids[5:10, 5:10, 5:10] = 2
        
        gen = ConcreteTextureGenerator()
        texture = gen.generate(micro)
        
        assert isinstance(texture, Texture)
        assert texture.n_orientations >= 1
        
    def test_must_implement_generate(self):
        """Test that subclasses must implement generate"""
        class IncompleteGenerator(TextureGenerator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteGenerator()
            
    def test_generate_returns_texture(self):
        """Test that generate returns Texture object"""
        micro = Microstructure(dimensions=(20, 20), resolution=1.0)
        micro.grain_ids[0:10, 0:10] = 1
        micro.grain_ids[10:20, 10:20] = 2
        
        gen = ConcreteTextureGenerator()
        result = gen.generate(micro)
        
        assert isinstance(result, Texture)
        assert hasattr(result, 'orientations')
        assert hasattr(result, 'representation')
        assert hasattr(result, 'symmetry')
        
        
