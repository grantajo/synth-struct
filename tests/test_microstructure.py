import sys
sys.path.insert(0, '../src')

import numpy as np
import unittest
from microstructure import Microstructure
from texture import Texture

class TestGrainMasks(unittest.TestCase):

    def test_get_grains_in_box_region(self):
        """Test getting grains in a box region"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        micro.gen_voronoi(num_grains=50, seed=42)
        
        grains = micro.get_grains_in_region('box', x_min=10, x_max=40, y_min=10, y_max=40, z_min=10, z_max=40)
        
        self.assertIsInstance(grains, list)
        self.assertGreater(len(grains), 0)
        self.assertNotIn(0, grains)  # No background grain

    def test_get_grains_in_sphere_region(self):
        """Test getting grains in a spherical region"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        micro.gen_voronoi(num_grains=50, seed=42)
        
        grains = micro.get_grains_in_region('sphere', center=[25, 25, 25], radius=15)
        
        self.assertGreater(len(grains), 0)
        self.assertLess(len(grains), 50)  # Should be subset

    def test_get_grains_entire_microstructure(self):
        """Test that full box returns all grains"""
        micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        micro.gen_voronoi(num_grains=30, seed=45)
        
        grains = micro.get_grains_in_region('box')  # No limits = entire volume
        get_grains = micro.get_num_grains()
        
        self.assertEqual(len(grains), micro.get_num_grains() - 1)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
