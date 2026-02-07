# synth_struct/tests/test_stiffness_cubic.py

import numpy as np
import pytest
from synth_struct.stiffness.cubic_stiffness import CubicStiffnessGenerator
from synth_struct.orientation.texture.texture import Texture


class TestCubicStiffnessGenerator:

    def test_initialization(self):
        """Test CubicStiffnessGenerator initialization"""
        C11, C12, C44 = 160.0, 92.0, 47.0
        gen = CubicStiffnessGenerator(C11=C11, C12=C12, C44=C44)
        
        assert gen.C11 == C11
        assert gen.C12 == C12
        assert gen.C44 == C44
        assert gen._base_tensor.shape == (6, 6)

    def test_base_tensor_structure(self):
        """Test that base tensor has correct cubic structure"""
        C11, C12, C44 = 160.0, 92.0, 47.0
        gen = CubicStiffnessGenerator(C11=C11, C12=C12, C44=C44)
        
        C = gen._base_tensor
        
        # Check diagonal elements
        assert C[0, 0] == C11
        assert C[1, 1] == C11
        assert C[2, 2] == C11
        
        # Check off-diagonal elements
        assert C[0, 1] == C12
        assert C[0, 2] == C12
        assert C[1, 2] == C12
        
        # Check shear elements
        assert C[3, 3] == C44
        assert C[4, 4] == C44
        assert C[5, 5] == C44

    def test_base_tensor_symmetry(self):
        """Test that base tensor is symmetric"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        C = gen._base_tensor
        
        np.testing.assert_array_almost_equal(C, C.T)

    def test_generate_with_identity_orientations(self):
        """Test generate method with identity rotations"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        # Create texture with identity rotation matrices
        n = 10
        orientations = np.array([np.eye(3) for _ in range(n)])
        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            symmetry="cubic"
        )
        
        # Mock microstructure (not used in basic implementation)
        micro = None
        
        stiffness = gen.generate(micro, texture)
        
        assert stiffness.n_tensors == n
        assert stiffness.crystal_structure == "cubic"
        
        # All tensors should equal base tensor (identity rotations)
        for i in range(n):
            np.testing.assert_array_almost_equal(
                stiffness.stiffness_tensors[i],
                gen._base_tensor
            )

    def test_generate_converts_euler_to_rotmat(self):
        """Test that generate converts Euler angles to rotation matrices"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        # Create texture with Euler angles
        n = 5
        orientations = np.zeros((n, 3))  # All zeros = identity
        texture = Texture(
            orientations=orientations,
            representation="euler",
            symmetry="cubic"
        )
        
        micro = None
        stiffness = gen.generate(micro, texture)
        
        assert stiffness.n_tensors == n
        
        # Should be close to base tensor (zero Euler = identity)
        for i in range(n):
            np.testing.assert_array_almost_equal(
                stiffness.stiffness_tensors[i],
                gen._base_tensor,
                decimal=10
            )

    def test_generate_with_random_orientations(self):
        """Test generate with random orientations"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        n = 10
        orientations = np.random.rand(n, 3) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            symmetry="cubic"
        )
        
        micro = None
        stiffness = gen.generate(micro, texture)
        
        assert stiffness.n_tensors == n
        assert stiffness.stiffness_tensors.shape == (n, 6, 6)

    def test_metadata_stored(self):
        """Test that elastic constants are stored in metadata"""
        C11, C12, C44 = 160.0, 92.0, 47.0
        gen = CubicStiffnessGenerator(C11=C11, C12=C12, C44=C44)
        
        n = 5
        orientations = np.array([np.eye(3) for _ in range(n)])
        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            symmetry="cubic"
        )
        
        stiffness = gen.generate(None, texture)
        
        assert stiffness.metadata["C11"] == C11
        assert stiffness.metadata["C12"] == C12
        assert stiffness.metadata["C44"] == C44

    def test_different_elastic_constants(self):
        """Test with different elastic constants"""
        # Aluminum values
        gen_al = CubicStiffnessGenerator(C11=108.0, C12=62.0, C44=28.0)
        
        # Copper values
        gen_cu = CubicStiffnessGenerator(C11=168.0, C12=121.0, C44=75.0)
        
        assert gen_al._base_tensor[0, 0] == 108.0
        assert gen_cu._base_tensor[0, 0] == 168.0
        assert gen_al._base_tensor[3, 3] == 28.0
        assert gen_cu._base_tensor[3, 3] == 75.0

    def test_single_grain(self):
        """Test with single grain"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        orientations = np.eye(3).reshape(1, 3, 3)
        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            symmetry="cubic"
        )
        
        stiffness = gen.generate(None, texture)
        
        assert stiffness.n_tensors == 1
        np.testing.assert_array_almost_equal(
            stiffness.stiffness_tensors[0],
            gen._base_tensor
        )

    def test_large_number_of_grains(self):
        """Test with large number of grains"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        n = 1000
        orientations = np.random.rand(n, 3) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            symmetry="cubic"
        )
        
        stiffness = gen.generate(None, texture)
        
        assert stiffness.n_tensors == n

    def test_90_degree_rotation_symmetry(self):
        """Test that 90° rotations respect cubic symmetry"""
        gen = CubicStiffnessGenerator(C11=160.0, C12=92.0, C44=47.0)
        
        # 90 degree rotation around z-axis
        angle = np.pi / 2
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        orientations = R.reshape(1, 3, 3)
        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            symmetry="cubic"
        )
        
        stiffness = gen.generate(None, texture)
        
        # Should be very close to base tensor due to cubic symmetry
        np.testing.assert_array_almost_equal(
            stiffness.stiffness_tensors[0],
            gen._base_tensor,
            decimal=10
        )
