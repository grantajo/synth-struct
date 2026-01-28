import sys
sys.path.insert(0, '../src')

import unittest

from texture import Texture
import rotation_converter as rc
import numpy as np
import numpy.testing as npt


class TestRotationConverter(unittest.TestCase):

    def test_euler_and_quats(self):
        """Test euler to quaternions and back"""
        orientations = Texture.random_orientations(5, seed=None)
        quats = rc.euler_to_quat(orientations)
        recovered = rc.quat_to_euler(quats)
        
        for key in orientations:
            npt.assert_allclose(
                orientations[key],
                recovered[key],
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"Mismatch in orientation component '{key}'"
            )
        
    def test_euler_and_rotation_matrices(self):
        """Test euler to rotation matrix and back"""
        orientations = Texture.random_orientations(5, seed=None) 
        rots = rc.euler_to_rotation_matrix(orientations)
        recovered = rc.rotation_matrix_to_euler(rots)
        
        for key in orientations:
            npt.assert_allclose(
                orientations[key],
                recovered[key],
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"Mismatch in orientation component '{key}'"
            )

    def test_quat_and_rotation_matrices(self):
        """Test quaternion to rotation matrix and back"""
        orientations = Texture.random_orientations(5, seed=None)
        quats = rc.euler_to_quat(orientations)
        rots = rc.quat_to_rotation_matrix(quats)
        recovered_quats = rc.rotation_matrix_to_quat(rots)
        
        for key in quats:
            npt.assert_allclose(
                quats[key],
                recovered_quats[key],
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"Mismatch in orientation component '{key}'"
            )
    
    def test_all_rotations(self):
        """Tests quats from Euler and quats from rotation matrices (through Euler)"""
        orientations = Texture.random_orientations(5, seed=None)
        rot_from_euler = rc.euler_to_rotation_matrix(orientations)
        quat_from_rot = rc.rotation_matrix_to_quat(rot_from_euler)
        
        quat_from_euler = rc.euler_to_quat(orientations)
        
        for key in quat_from_rot:
            npt.assert_allclose(
                quat_from_rot[key],
                quat_from_euler[key],
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"Mismatch in orientation component '{key}'"
            )
        
        
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
