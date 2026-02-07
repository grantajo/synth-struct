# synth_struct/tests/test_stiffness_isotropic.py

import numpy as np
import pytest
from synth_struct.stiffness.isotropic_stiffness import IsotropicStiffnessGenerator
from synth_struct.orientation.texture.texture import Texture


class TestIsotropicStiffnessGenerator:

    def test_initialization(self):
        """Test IsotropicStiffnessGenerator initialization"""
        E, nu = 210.0, 0.3
        gen = IsotropicStiffnessGenerator(E=E, nu=nu)
        
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        c = lam + 2 * mu
        
        assert gen.E == E
        assert gen.nu == nu
        assert gen.lam == lam
        assert gen.mu == mu
        assert gen.c == c
        assert gen._base_tensor.shape == (6, 6)

    def test_base_tensor_structure(self):
        """Test that base tensor has correct Isotropic structure"""
        E, nu = 210.0, 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        c = lam + 2 * mu
        
        gen = IsotropicStiffnessGenerator(E=E, nu=nu)
        
        C = gen._base_tensor
        
        # Check diagonal elements
        assert C[0, 0] == c
        assert C[1, 1] == c
        assert C[2, 2] == c
        
        # Check off-diagonal elements
        assert C[0, 1] == lam
        assert C[0, 2] == lam
        assert C[1, 2] == lam
        
        # Check shear elements
        assert C[3, 3] == mu
        assert C[4, 4] == mu
        assert C[5, 5] == mu

    def test_base_tensor_symmetry(self):
        """Test that base tensor is symmetric"""
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
        C = gen._base_tensor
        
        np.testing.assert_array_almost_equal(C, C.T)

    def test_generate_with_identity_orientations(self):
        """Test generate method with identity rotations"""
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
        E, nu = 210.0, 0.3
        gen = IsotropicStiffnessGenerator(E=E, nu=nu)
        
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        c = lam + 2 * mu
        
        n = 5
        orientations = np.array([np.eye(3) for _ in range(n)])
        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            symmetry="cubic"
        )
        
        stiffness = gen.generate(None, texture)
        
        assert stiffness.metadata["E"] == E
        assert stiffness.metadata["nu"] == nu
        assert stiffness.metadata["lam"] == lam
        assert stiffness.metadata["mu"] == mu
        assert stiffness.metadata["c"] == c

    def test_different_elastic_constants(self):
        """Test with different elastic constants"""
        # Aluminum values
        E_al, nu_al = 108.0, 0.33
        gen_al = IsotropicStiffnessGenerator(E=E_al, nu=nu_al)
        
        lam_al = E_al * nu_al / ((1 + nu_al) * (1 - 2 * nu_al))
        mu_al = E_al / (2 * (1 + nu_al))
        c_al = lam_al + 2 * mu_al
        
        # Copper values
        E_w, nu_w = 375.0, 0.28
        gen_w = IsotropicStiffnessGenerator(E=E_w, nu=nu_w)
        
        lam_w = E_w * nu_w / ((1 + nu_w) * (1 - 2 * nu_w))
        mu_w = E_w / (2 * (1 + nu_w))
        c_w = lam_w + 2 * mu_w
        
        assert gen_al._base_tensor[0, 0] == c_al
        assert gen_w._base_tensor[0, 0] == c_w
        assert gen_al._base_tensor[3, 3] == mu_al
        assert gen_w._base_tensor[3, 3] == mu_w

    def test_single_grain(self):
        """Test with single grain"""
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
        gen = IsotropicStiffnessGenerator(E=210.0, nu=0.3)
        
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
