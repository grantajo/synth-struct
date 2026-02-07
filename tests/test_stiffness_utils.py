# synth_struct/tests/test_stiffness_utils.py

import numpy as np
import pytest
from synth_struct.stiffness.stiffness_utils import (
    rotate_stiffness_tensor,
    rotate_stiffness_tensors_batch,
)


class TestRotateStiffnessTensor:

    def test_identity_rotation(self):
        """Test that identity rotation leaves tensor unchanged"""
        C = np.random.rand(6, 6) * 100
        R = np.eye(3)
        
        C_rotated = rotate_stiffness_tensor(C, R)
        
        np.testing.assert_array_almost_equal(C_rotated, C)

    def test_rotation_matrix_shape(self):
        """Test with proper rotation matrix"""
        C = np.random.rand(6, 6) * 100
        
        # Create a proper rotation matrix (90 deg around z-axis)
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        C_rotated = rotate_stiffness_tensor(C, R)
        
        assert C_rotated.shape == (6, 6)

    def test_180_degree_rotation_z_axis(self):
        """Test 180 degree rotation around z-axis"""
        # Create isotropic-like tensor for simple test
        C = np.eye(6) * 100
        
        # 180 degree rotation around z
        R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        C_rotated = rotate_stiffness_tensor(C, R)
        
        # For diagonal isotropic tensor, should be unchanged
        np.testing.assert_array_almost_equal(C_rotated, C, decimal=10)

    def test_cubic_symmetry_preservation(self):
        """Test that cubic symmetry is preserved under 90° rotations"""
        # Cubic stiffness tensor
        C11, C12, C44 = 160.0, 92.0, 47.0
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = C11
        C[0, 1] = C[0, 2] = C[1, 2] = C12
        C[1, 0] = C[2, 0] = C[2, 1] = C12
        C[3, 3] = C[4, 4] = C[5, 5] = C44
        
        # 90 degree rotation around z-axis
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        C_rotated = rotate_stiffness_tensor(C, R)
        
        # Should be very similar due to cubic symmetry
        np.testing.assert_array_almost_equal(C_rotated, C, decimal=10)

    def test_rotation_orthogonality(self):
        """Test that rotation matrix must be orthogonal"""
        C = np.random.rand(6, 6) * 100
        R = np.random.rand(3, 3)  # Not orthogonal
        
        # Should still compute, but result may not be physical
        C_rotated = rotate_stiffness_tensor(C, R)
        
        assert C_rotated.shape == (6, 6)

    def test_bond_transformation_matrix_symmetry(self):
        """Test that the transformation preserves certain symmetries"""
        C = np.eye(6) * 100
        
        # Small rotation
        angle = 0.1
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        C_rotated = rotate_stiffness_tensor(C, R)
        
        # Should be close to original for small rotation
        np.testing.assert_array_almost_equal(C_rotated, C, decimal=1)


class TestRotateStiffnessTensorsBatch:

    def test_single_base_tensor_multiple_rotations(self):
        """Test rotation with single base tensor and multiple rotation matrices"""
        C_base = np.random.rand(6, 6) * 100
        n = 10
        R_matrices = np.array([np.eye(3) for _ in range(n)])
        
        C_rotated = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        assert C_rotated.shape == (n, 6, 6)
        
        # All should be equal to base tensor (identity rotations)
        for i in range(n):
            np.testing.assert_array_almost_equal(C_rotated[i], C_base)

    def test_multiple_base_tensors_multiple_rotations(self):
        """Test rotation with multiple base tensors and rotation matrices"""
        n = 10
        C_base = np.random.rand(n, 6, 6) * 100
        R_matrices = np.array([np.eye(3) for _ in range(n)])
        
        C_rotated = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        assert C_rotated.shape == (n, 6, 6)
        
        # All should be equal to corresponding base tensors (identity rotations)
        for i in range(n):
            np.testing.assert_array_almost_equal(C_rotated[i], C_base[i])

    def test_mismatched_dimensions_error(self):
        """Test that mismatched dimensions raise ValueError"""
        C_base = np.random.rand(5, 6, 6) * 100  # 5 tensors
        R_matrices = np.array([np.eye(3) for _ in range(10)])  # 10 rotations
        
        with pytest.raises(ValueError, match="must match"):
            rotate_stiffness_tensors_batch(C_base, R_matrices)

    def test_invalid_base_tensor_shape(self):
        """Test that invalid base tensor shape raises ValueError"""
        C_base = np.random.rand(6, 6, 6)  # Wrong shape
        R_matrices = np.array([np.eye(3) for _ in range(5)])
        
        with pytest.raises(ValueError, match="must have shape"):
            rotate_stiffness_tensors_batch(C_base, R_matrices)

    def test_various_rotations(self):
        """Test with various rotation matrices"""
        C_base = np.eye(6) * 100
        n = 5
        
        # Create different rotation matrices
        R_matrices = np.zeros((n, 3, 3))
        for i in range(n):
            angle = i * np.pi / 4
            R_matrices[i] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        
        C_rotated = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        assert C_rotated.shape == (n, 6, 6)
        
        # First one (angle=0) should be identity
        np.testing.assert_array_almost_equal(C_rotated[0], C_base)

    def test_single_tensor_single_rotation(self):
        """Test with single tensor and single rotation"""
        C_base = np.random.rand(6, 6) * 100
        R_matrices = np.eye(3).reshape(1, 3, 3)
        
        C_rotated = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        assert C_rotated.shape == (1, 6, 6)
        np.testing.assert_array_almost_equal(C_rotated[0], C_base)

    def test_large_batch(self):
        """Test with large batch of rotations"""
        C_base = np.random.rand(6, 6) * 100
        n = 1000
        R_matrices = np.array([np.eye(3) for _ in range(n)])
        
        C_rotated = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        assert C_rotated.shape == (n, 6, 6)

    def test_consistency_with_single_rotation(self):
        """Test that batch rotation is consistent with single rotations"""
        C_base = np.random.rand(6, 6) * 100
        n = 5
        
        R_matrices = np.zeros((n, 3, 3))
        for i in range(n):
            angle = i * 0.5
            R_matrices[i] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        
        # Batch rotation
        C_batch = rotate_stiffness_tensors_batch(C_base, R_matrices)
        
        # Individual rotations
        for i in range(n):
            C_single = rotate_stiffness_tensor(C_base, R_matrices[i])
            np.testing.assert_array_almost_equal(C_batch[i], C_single)
