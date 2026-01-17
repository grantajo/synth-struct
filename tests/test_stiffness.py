import sys
sys.path.insert(0, '../src')

import numpy as np
import unittest
from microstructure import Microstructure
from texture import Texture
from stiffness import Stiffness

class TestStiffness(unittest.TestCase):
    
    def setUp(self):
        """Set up test microstructure"""
        self.micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        self.micro.gen_voronoi(num_grains=5, seed=42)
        self.micro.orientations = Texture.random_orientations(5, seed=42)
        
    # Test Isotropic
    def test_isotropic_returns_dict(self):
        stiffness = Stiffness.Isotropic(self.micro)
        self.assertIsInstance(stiffness, dict)
        
    def test_isotropic_correct_num_grains(self):
        """Test that Isotropic creates stiffness for all grains"""
        stiffness = Stiffness.Isotropic(self.micro)
        num_grains = self.micro.get_num_grains()
        self.assertEqual(len(stiffness), num_grains)
    
    def test_isotropic_values(self):
        """Test that Isotropic returns correct E and nu"""
        E, nu = 200.0, 0.25
        stiffness = Stiffness.Isotropic(self.micro, E=E, nu=nu)
        for grain_id, values in stiffness.items():
            self.assertEqual(values[0], E)
            self.assertEqual(values[1], nu)
    
    def test_isotropic_default_values(self):
        """Test that Isotropic uses correct defaults"""
        stiffness = Stiffness.Isotropic(self.micro)
        for grain_id, values in stiffness.items():
            self.assertEqual(values[0], 210.0)
            self.assertEqual(values[1], 0.3)
    
    # ============ Test Cubic ============
    def test_cubic_returns_dict(self):
        """Test that Cubic returns a dictionary"""
        stiffness = Stiffness.Cubic(self.micro)
        self.assertIsInstance(stiffness, dict)
    
    def test_cubic_correct_num_grains(self):
        """Test that Cubic creates stiffness for all grains"""
        stiffness = Stiffness.Cubic(self.micro)
        self.assertEqual(len(stiffness), len(self.micro.orientations))
    
    def test_cubic_matrix_shape(self):
        """Test that Cubic returns 6x6 matrices"""
        stiffness = Stiffness.Cubic(self.micro)
        for grain_id, C in stiffness.items():
            self.assertEqual(C.shape, (6, 6))
    
    def test_cubic_symmetry(self):
        """Test that rotated stiffness matrix is symmetric"""
        stiffness = Stiffness.Cubic(self.micro)
        for grain_id, C in stiffness.items():
            np.testing.assert_array_almost_equal(C, C.T, decimal=10,
                err_msg=f"Grain {grain_id} stiffness not symmetric")
    
    def test_cubic_zero_rotation(self):
        """Test that zero rotation returns original cubic stiffness"""
        # Create microstructure with zero orientation
        micro_zero = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro_zero.gen_voronoi(num_grains=1, seed=42)
        micro_zero.orientations = {1: np.array([0.0, 0.0, 0.0])}
        
        C11, C12, C44 = 228.0, 116.5, 132.0
        stiffness = Stiffness.Cubic(micro_zero, C11=C11, C12=C12, C44=C44)
        
        C_expected = np.array([
            [C11, C12, C12,   0,   0,   0],
            [C12, C11, C12,   0,   0,   0],
            [C12, C12, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44]
        ])
        
        np.testing.assert_array_almost_equal(stiffness[1], C_expected, decimal=10)
    
    def test_cubic_positive_definite(self):
        """Test that stiffness matrices are positive definite"""
        stiffness = Stiffness.Cubic(self.micro)
        for grain_id, C in stiffness.items():
            eigenvalues = np.linalg.eigvalsh(C)
            self.assertTrue(np.all(eigenvalues > 0),
                f"Grain {grain_id} stiffness not positive definite")
    
    # ============ Test Hexagonal ============
    def test_hexagonal_returns_dict(self):
        """Test that Hexagonal returns a dictionary"""
        stiffness = Stiffness.Hexagonal(self.micro)
        self.assertIsInstance(stiffness, dict)
    
    def test_hexagonal_matrix_shape(self):
        """Test that Hexagonal returns 6x6 matrices"""
        stiffness = Stiffness.Hexagonal(self.micro)
        for grain_id, C in stiffness.items():
            self.assertEqual(C.shape, (6, 6))
    
    def test_hexagonal_symmetry(self):
        """Test that hexagonal stiffness matrix is symmetric"""
        stiffness = Stiffness.Hexagonal(self.micro)
        for grain_id, C in stiffness.items():
            np.testing.assert_array_almost_equal(C, C.T, decimal=10)
    
    def test_hexagonal_c66_relation(self):
        """Test that C66 = 0.5*(C11 - C12) for zero rotation"""
        micro_zero = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro_zero.gen_voronoi(num_grains=1, seed=42)
        micro_zero.orientations = {1: np.array([0.0, 0.0, 0.0])}
        
        C11, C12 = 162.4, 92.0
        stiffness = Stiffness.Hexagonal(micro_zero, C11=C11, C12=C12)
        
        C66_expected = 0.5 * (C11 - C12)
        C66_actual = stiffness[1][5, 5]
        
        self.assertAlmostEqual(C66_actual, C66_expected, places=10)
    
    # ============ Test Rotation Functions ============
    def test_voigt_tensor_conversion_roundtrip(self):
        """Test that Voigt <-> Tensor conversion is reversible"""
        C11, C12, C44 = 228.0, 116.5, 132.0
        C_original = np.array([
            [C11, C12, C12,   0,   0,   0],
            [C12, C11, C12,   0,   0,   0],
            [C12, C12, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44]
        ])
        
        # Convert to tensor and back
        C_tensor = Stiffness._voigt_to_tensor(C_original)
        C_recovered = Stiffness._tensor_to_voigt(C_tensor)
        
        np.testing.assert_array_almost_equal(C_original, C_recovered, decimal=10)
    
    def test_rotation_preserves_determinant_sign(self):
        """Test that rotation doesn't flip the sign of determinant"""
        stiffness = Stiffness.Cubic(self.micro)
        
        C11, C12, C44 = 228.0, 116.5, 132.0
        C_original = np.array([
            [C11, C12, C12,   0,   0,   0],
            [C12, C11, C12,   0,   0,   0],
            [C12, C12, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44]
        ])
        det_original = np.linalg.det(C_original)
        
        for grain_id, C_rotated in stiffness.items():
            det_rotated = np.linalg.det(C_rotated)
            self.assertGreater(det_rotated, 0, 
                f"Grain {grain_id}: determinant changed sign after rotation")
    
    def test_90_degree_rotation_cubic(self):
        """Test 90 degree rotation around Z-axis for cubic symmetry"""
        micro_rot = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro_rot.gen_voronoi(num_grains=1, seed=42)
        # 90 degree rotation around Z: phi1=90deg, Phi=0, phi2=0
        micro_rot.orientations = {1: np.array([np.pi/2, 0.0, 0.0])}
        
        stiffness = Stiffness.Cubic(micro_rot)
        C = stiffness[1]
        
        # For cubic, 90 degree rotation around Z should swap X and Y
        # C[0,0] and C[1,1] should remain equal (cubic symmetry)
        self.assertAlmostEqual(C[0,0], C[1,1], places=8,
            msg="C11 and C22 should be equal after 90° Z rotation (cubic symmetry)")
    
    # ============ Test Edge Cases ============
    def test_empty_orientations(self):
        """Test behavior with no orientations"""
        micro_empty = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro_empty.gen_voronoi(num_grains=1, seed=42)
        micro_empty.orientations = {}
        
        stiffness = Stiffness.Cubic(micro_empty)
        self.assertEqual(len(stiffness), 0)
    
    def test_single_grain(self):
        """Test with single grain"""
        micro_single = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro_single.gen_voronoi(num_grains=1, seed=42)
        micro_single.orientations = Texture.random_orientations(1, seed=42)
        
        stiffness = Stiffness.Cubic(micro_single)
        self.assertEqual(len(stiffness), 1)
        self.assertEqual(stiffness[1].shape, (6, 6))
    
    def test_large_number_of_grains(self):
        """Test with many grains"""
        micro_large = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        micro_large.gen_voronoi(num_grains=100, seed=42)
        micro_large.orientations = Texture.random_orientations(100, seed=42)
        
        stiffness = Stiffness.Cubic(micro_large)
        self.assertEqual(len(stiffness), 100)
    
    # ============ Test Physical Constraints ============
    def test_cubic_elastic_bounds(self):
        """Test that cubic constants satisfy stability criteria"""
        # For cubic: C11 > |C12|, C44 > 0, C11 + 2*C12 > 0
        stiffness = Stiffness.Cubic(self.micro, C11=228.0, C12=116.5, C44=132.0)
        
        for grain_id, C in stiffness.items():
            # These should hold even after rotation for positive definite matrix
            eigenvalues = np.linalg.eigvalsh(C)
            self.assertTrue(np.all(eigenvalues > 0))
    
    def test_hexagonal_elastic_bounds(self):
        """Test that hexagonal constants satisfy stability criteria"""
        stiffness = Stiffness.Hexagonal(self.micro)
        
        for grain_id, C in stiffness.items():
            eigenvalues = np.linalg.eigvalsh(C)
            self.assertTrue(np.all(eigenvalues > 0))


class TestStiffnessIntegration(unittest.TestCase):
    """Integration tests with different microstructure types"""
    
    def test_2d_microstructure(self):
        """Test stiffness calculation with 2D microstructure"""
        micro_2d = Microstructure(dimensions=(20, 20), resolution=1.0)
        micro_2d.gen_voronoi(num_grains=10, seed=42)
        micro_2d.orientations = Texture.random_orientations(10, seed=42)
        
        stiffness = Stiffness.Cubic(micro_2d)
        self.assertEqual(len(stiffness), 10)
    
    def test_different_crystal_systems(self):
        """Test that different crystal systems give different results"""
        micro = Microstructure(dimensions=(10, 10, 10), resolution=1.0)
        micro.gen_voronoi(num_grains=3, seed=42)
        micro.orientations = {
            1: np.array([0.5, 0.3, 0.2]),
            2: np.array([1.0, 0.5, 0.8]),
            3: np.array([0.2, 0.9, 0.4])
        }
        
        stiffness_cubic = Stiffness.Cubic(micro)
        stiffness_hex = Stiffness.Hexagonal(micro)
        
        # They should be different
        for grain_id in [1, 2, 3]:
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(
                    stiffness_cubic[grain_id], 
                    stiffness_hex[grain_id]
                )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
