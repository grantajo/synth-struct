import sys
sys.path.insert(0, '../src')

import numpy as np
import unittest
from microstructure import Microstructure
from texture import Texture


class TestApplyTextureToRegion(unittest.TestCase):
    
    def setUp(self):
        """Set up test microstructure"""
        self.micro = Microstructure(dimensions=(20, 20, 20), resolution=1.0)
        self.micro.gen_voronoi(num_grains=50, seed=42)
        self.micro.orientations = Texture.random_orientations(50, seed=42)
        
        # Store original orientations for comparison
        self.original_orientations = {k: v.copy() for k, v in self.micro.orientations.items()}
    
    # ============ Test Basic Functionality ============
    def test_apply_to_entire_microstructure(self):
        """Test applying texture to entire microstructure (region_grain_ids=None)"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            region_grain_ids=None,
            texture_type='cube',
            degspread=10
        )
        
        # All grains should be modified
        self.assertEqual(len(orientations), len(self.micro.orientations))
        
        # Check that orientations changed
        changed = sum(1 for gid in orientations.keys() 
                     if not np.allclose(orientations[gid], self.original_orientations[gid]))
        self.assertGreater(changed, 0)
    
    def test_apply_to_specific_region(self):
        """Test applying texture to specific grains only"""
        # Select first 10 grains
        region_grains = list(self.micro.orientations.keys())[:10]
        
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            region_grain_ids=region_grains,
            texture_type='brass',
            degspread=15
        )
        
        # Only region grains should change
        for gid in region_grains:
            self.assertFalse(np.allclose(orientations[gid], self.original_orientations[gid]))
        
        # Grains outside region should remain unchanged
        for gid in list(self.micro.orientations.keys())[10:]:
            np.testing.assert_array_almost_equal(
                orientations[gid], 
                self.original_orientations[gid]
            )
    
    def test_empty_region(self):
        """Test with empty region list"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            region_grain_ids=[],
            texture_type='cube',
            degspread=10
        )
        
        # No orientations should change
        for gid in orientations.keys():
            np.testing.assert_array_almost_equal(
                orientations[gid],
                self.original_orientations[gid]
            )
    
    # ============ Test All Texture Types ============
    def test_cube_texture(self):
        """Test cube texture {001}<100>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='cube',
            degspread=5
        )
        
        # Check that orientations are near [0, 0, 0]
        mean_orientation = np.array([0.0, 0.0, 0.0])
        for gid, angles in orientations.items():
            deviation = np.linalg.norm(angles - mean_orientation)
            self.assertLess(deviation, np.radians(20))  # Within reasonable spread
    
    def test_goss_texture(self):
        """Test Goss texture {011}<100>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='goss',
            degspread=5
        )
        
        # Check Phi is near 45 degrees
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[1], np.radians(45), delta=np.radians(20))
    
    def test_brass_texture(self):
        """Test brass texture {011}<211>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='brass',
            degspread=5
        )
        
        mean_orientation = np.array([np.radians(35.26), np.radians(45), 0.0])
        for gid, angles in orientations.items():
            deviation = np.linalg.norm(angles - mean_orientation)
            self.assertLess(deviation, np.radians(20))
    
    def test_copper_texture(self):
        """Test copper texture {112}<111>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='copper',
            degspread=5
        )
        
        # Check orientation components
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[0], np.radians(90), delta=np.radians(20))
            self.assertAlmostEqual(angles[1], np.radians(35.26), delta=np.radians(20))
    
    def test_s_texture(self):
        """Test S texture {123}<634>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='s',
            degspread=5
        )
        
        mean_orientation = np.array([np.radians(58.98), np.radians(36.70), np.radians(63.43)])
        for gid, angles in orientations.items():
            deviation = np.linalg.norm(angles - mean_orientation)
            self.assertLess(deviation, np.radians(20))
    
    def test_rotated_cube_texture(self):
        """Test rotated cube texture {001}<110>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='rotated_cube',
            degspread=5
        )
        
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[0], np.radians(45), delta=np.radians(20))
    
    def test_rotated_goss_texture(self):
        """Test rotated Goss texture {011}<011>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='rotated_goss',
            degspread=5
        )
        
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[1], np.radians(45), delta=np.radians(20))
            self.assertAlmostEqual(angles[2], np.radians(45), delta=np.radians(20))
    
    def test_p_texture(self):
        """Test P texture {011}<122>"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='p',
            degspread=5
        )
        
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[0], np.radians(70.53), delta=np.radians(20))
    
    def test_basal_texture(self):
        """Test basal texture (HCP)"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='basal',
            degspread=5
        )
        
        mean_orientation = np.array([0.0, 0.0, 0.0])
        for gid, angles in orientations.items():
            deviation = np.linalg.norm(angles - mean_orientation)
            self.assertLess(deviation, np.radians(20))
    
    def test_prismatic_texture(self):
        """Test prismatic texture (HCP)"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='prismatic',
            degspread=5
        )
        
        for gid, angles in orientations.items():
            self.assertAlmostEqual(angles[1], np.radians(90), delta=np.radians(20))
    
    def test_random_texture(self):
        """Test random texture"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='random'
        )
        
        # Check that orientations are uniformly distributed
        phi1_values = [angles[0] for angles in orientations.values()]
        
        # Should have some variety (not all clustered)
        self.assertGreater(np.std(phi1_values), 0.5)
        
        # Check bounds
        for gid, angles in orientations.items():
            self.assertGreaterEqual(angles[0], 0)
            self.assertLessEqual(angles[0], 2*np.pi)
            self.assertGreaterEqual(angles[1], 0)
            self.assertLessEqual(angles[1], np.pi)
    
    # ============ Test Degspread Parameter ============
    def test_small_degspread(self):
        """Test that small degspread creates tight clustering"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='cube',
            degspread=1  # Very tight
        )
        
        mean_orientation = np.array([0.0, 0.0, 0.0])
        deviations = [np.linalg.norm(angles - mean_orientation) 
                     for angles in orientations.values()]
        
        # All should be very close to ideal orientation
        self.assertLess(np.max(deviations), np.radians(5))
    
    def test_large_degspread(self):
        """Test that large degspread creates wider scatter"""
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='cube',
            degspread=30  # Wide spread
        )
        
        mean_orientation = np.array([0.0, 0.0, 0.0])
        deviations = [np.linalg.norm(angles - mean_orientation) 
                     for angles in orientations.values()]
        
        # Should have more scatter
        self.assertGreater(np.std(deviations), np.radians(5))
    
    def test_degspread_comparison(self):
        """Test that larger degspread gives larger scatter"""
        orientations_tight = Texture.apply_texture_to_region(
            self.micro.orientations.copy(),
            texture_type='brass',
            degspread=5
        )
        
        orientations_wide = Texture.apply_texture_to_region(
            self.micro.orientations.copy(),
            texture_type='brass',
            degspread=20
        )
        
        mean_orientation = np.array([np.radians(35.26), np.radians(45), 0.0])
        
        std_tight = np.std([np.linalg.norm(angles - mean_orientation) 
                           for angles in orientations_tight.values()])
        std_wide = np.std([np.linalg.norm(angles - mean_orientation) 
                          for angles in orientations_wide.values()])
        
        self.assertLess(std_tight, std_wide)
    
    # ============ Test Error Handling ============
    def test_invalid_texture_type(self):
        """Test that invalid texture type raises error"""
        with self.assertRaises(ValueError) as context:
            Texture.apply_texture_to_region(
                self.micro.orientations,
                texture_type='invalid_texture'
            )
        
        self.assertIn('Unknown texture type', str(context.exception))
    
    def test_nonexistent_grain_ids(self):
        """Test behavior with grain IDs that don't exist"""
        # This should not raise an error, just skip those grains
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            region_grain_ids=[999, 1000, 1001],  # Don't exist
            texture_type='cube'
        )
        
        # Original grains should be unchanged
        for gid in self.micro.orientations.keys():
            np.testing.assert_array_almost_equal(
                orientations[gid],
                self.original_orientations[gid]
            )
            
        # The nonexistent grain IDs should not be added to the dictionary
        self.assertNotIn(999, orientations)
        self.assertNotIn(1000, orientations)
        self.assertNotIn(1001, orientations)
    
    # ============ Test Multiple Regions ============
    def test_multiple_regions_different_textures(self):
        """Test applying different textures to different regions"""
        # Region 1: First 15 grains - Cube texture
        region1 = list(self.micro.orientations.keys())[:15]
        orientations = Texture.apply_texture_to_region(
            self.micro.orientations,
            region_grain_ids=region1,
            texture_type='cube',
            degspread=5
        )
        
        # Region 2: Next 15 grains - Brass texture
        region2 = list(self.micro.orientations.keys())[15:30]
        orientations = Texture.apply_texture_to_region(
            orientations,
            region_grain_ids=region2,
            texture_type='brass',
            degspread=5
        )
        
        # Check region 1 has cube texture
        cube_mean = np.array([0.0, 0.0, 0.0])
        for gid in region1:
            deviation = np.linalg.norm(orientations[gid] - cube_mean)
            self.assertLess(deviation, np.radians(20))
        
        # Check region 2 has brass texture
        brass_mean = np.array([np.radians(35.26), np.radians(45), 0.0])
        for gid in region2:
            deviation = np.linalg.norm(orientations[gid] - brass_mean)
            self.assertLess(deviation, np.radians(20))
    
    # ============ Test Return Value ============
    def test_returns_modified_dict(self):
        """Test that function returns the orientations dictionary"""
        result = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='cube'
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.micro.orientations))
    
    def test_modifies_in_place(self):
        """Test that function modifies the original dictionary"""
        original_id = id(self.micro.orientations)
        
        result = Texture.apply_texture_to_region(
            self.micro.orientations,
            texture_type='cube'
        )
        
        # Should be the same object
        self.assertEqual(id(result), original_id)


class TestTextureIntegration(unittest.TestCase):
    """Integration tests with realistic workflows"""
    
    def test_realistic_workflow_single_texture(self):
        """Test realistic workflow: create microstructure with single texture"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        micro.gen_voronoi(num_grains=100, seed=42)
        
        # Initialize with random, then apply texture to all
        micro.orientations = Texture.random_orientations(100, seed=42)
        micro.orientations = Texture.apply_texture_to_region(
            micro.orientations,
            texture_type='brass',
            degspread=10
        )
        
        self.assertEqual(len(micro.orientations), 100)
    
    def test_realistic_workflow_regional_textures(self):
        """Test realistic workflow: different textures in different regions"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        micro.gen_voronoi(num_grains=100, seed=42)
        micro.orientations = Texture.random_orientations(100, seed=42)
        
        # Bottom half: brass texture
        bottom_grains = []
        for gid in micro.orientations.keys():
            coords = np.where(micro.grain_ids == gid)
            if len(coords[0]) > 0 and np.mean(coords[0]) < 25:
                bottom_grains.append(gid)
        
        micro.orientations = Texture.apply_texture_to_region(
            micro.orientations,
            region_grain_ids=bottom_grains,
            texture_type='brass',
            degspread=10
        )
        
        # Top half: goss texture
        top_grains = []
        for gid in micro.orientations.keys():
            coords = np.where(micro.grain_ids == gid)
            if len(coords[0]) > 0 and np.mean(coords[0]) >= 25:
                top_grains.append(gid)
        
        micro.orientations = Texture.apply_texture_to_region(
            micro.orientations,
            region_grain_ids=top_grains,
            texture_type='goss',
            degspread=15
        )
        
        self.assertGreater(len(bottom_grains), 0)
        self.assertGreater(len(top_grains), 0)
        self.assertEqual(len(bottom_grains) + len(top_grains), 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
