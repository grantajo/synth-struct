# synth_struct/tests/test_texture.py

import numpy as np
import pytest

from synth_struct.orientation.texture.texture import Texture
from synth_struct import Phase

cubic_phase = Phase.from_preset("default_cubic")
hex_phase = Phase.from_preset("default_hexagonal")
rng = np.random.default_rng()


class TestTexture:

    def test_initialization_euler(self):
        """Test Texture initialization with Euler angles"""
        orientations = rng.random((10, 3)) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        assert texture.representation == "euler"
        assert texture.phase.crystal_system == "cubic"
        assert texture.phase.point_group == "m-3m"
        assert texture.n_orientations == 10
        np.testing.assert_array_equal(texture.orientations, orientations)

    def test_initialization_quat(self):
        """Test Texture initialization with quaternions"""
        orientations = rng.random((10, 4))
        orientations = orientations / np.linalg.norm(
            orientations, axis=1, keepdims=True
        )

        texture = Texture(
            orientations=orientations,
            representation="quat",
            phase=hex_phase,
        )

        assert texture.representation == "quat"
        assert texture.phase.crystal_system == "hexagonal"
        assert texture.phase.point_group == "6/mmm"
        assert texture.n_orientations == 10

    def test_initialization_rotmat(self):
        """Test Texture initialization with rotation matrices"""
        orientations = np.array([np.eye(3) for _ in range(10)])

        texture = Texture(
            orientations=orientations,
            representation="rotmat",
            phase=cubic_phase,
        )

        assert texture.representation == "rotmat"
        assert texture.n_orientations == 10

    def test_initialization_with_metadata(self):
        """Test Texture initialization with metadata"""
        orientations = rng.random((5, 3))
        metadata = {"source": "EBSD", "sample_id": "S001"}

        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
            metadata=metadata,
        )

        assert texture.metadata == metadata
        assert texture.metadata["source"] == "EBSD"

    def test_invalid_orientations_type(self):
        """Test that non-array orientations raise TypeError"""
        with pytest.raises(TypeError, match="must be a NumPy array"):
            Texture(
                orientations=[[1, 2, 3], [4, 5, 6]],  # List, not array
                representation="euler",
                phase=cubic_phase,
            )

    def test_invalid_orientations_shape(self):
        """Test that 1D orientations raise ValueError"""
        with pytest.raises(ValueError, match="must have shape"):
            Texture(
                orientations=np.array([1, 2, 3]),  # 1D array
                representation="euler",
                phase=cubic_phase,
            )

    def test_invalid_representation(self):
        """Test that invalid representation raises ValueError"""
        orientations = rng.random((5, 3))

        with pytest.raises(ValueError, match="Unknown representation"):
            Texture(
                orientations=orientations,
                representation="invalid",
                phase=cubic_phase,
            )

    def test_copy(self):
        """Test that copy creates independent copy"""
        orientations = rng.random((5, 3))
        metadata = {"test": "data"}

        texture1 = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
            metadata=metadata,
        )

        texture2 = texture1.copy()

        # Modify original
        texture1.orientations[0, 0] = 999
        texture1.metadata["new_key"] = "new_value"

        # Copy should be unchanged
        assert texture2.orientations[0, 0] != 999
        assert "new_key" not in texture2.metadata

    def test_subset(self):
        """Test subset extraction"""
        orientations = rng.random((10, 3))
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        indices = np.array([0, 2, 4, 6])
        subset_texture = texture.subset(indices)

        assert subset_texture.n_orientations == 4
        np.testing.assert_array_equal(
            subset_texture.orientations, orientations[indices]
        )
        assert subset_texture.representation == "euler"
        assert subset_texture.phase.crystal_system == "cubic"

    def test_subset_preserves_metadata(self):
        """Test that subset preserves metadata"""
        orientations = np.random.rand(10, 3)
        metadata = {"source": "test"}

        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
            metadata=metadata,
        )

        subset_texture = texture.subset(np.array([0, 1, 2]))

        assert subset_texture.metadata == metadata

    def test_euler_to_quat_conversion(self):
        """Test conversion from Euler to quaternion"""
        orientations = rng.random((5, 3)) * 2 * np.pi
        texture_euler = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        texture_quat = texture_euler.to_representation("quat")

        assert texture_quat.representation == "quat"
        assert texture_quat.orientations.shape == (5, 4)

        # Quaternions should be normalized
        norms = np.linalg.norm(texture_quat.orientations, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5))

    def test_euler_to_rotmat_conversion(self):
        """Test conversion from Euler to rotation matrix"""
        orientations = rng.random((5, 3)) * 2 * np.pi
        texture_euler = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        texture_rotmat = texture_euler.to_representation("rotmat")

        assert texture_rotmat.representation == "rotmat"
        assert texture_rotmat.orientations.shape == (5, 3, 3)

        # Rotation matrices should be orthogonal
        for R in texture_rotmat.orientations:
            identity = np.dot(R.T, R)
            np.testing.assert_array_almost_equal(identity, np.eye(3))

    def test_quat_to_euler_conversion(self):
        """Test conversion from quaternion to Euler"""
        from synth_struct.orientation.rotation_converter import euler_to_quat

        original_euler = rng.random((5, 3)) * 2 * np.pi
        orientations = euler_to_quat(original_euler)

        texture_quat = Texture(
            orientations=orientations,
            representation="quat",
            phase=cubic_phase,
        )

        texture_euler = texture_quat.to_representation("euler")

        assert texture_euler.representation == "euler"
        assert texture_euler.orientations.shape == (5, 3)

    def test_quat_to_rotmat_conversion(self):
        """Test conversion from quaternion to rotation matrix"""
        orientations = rng.random((5, 4))
        orientations = orientations / np.linalg.norm(
            orientations, axis=1, keepdims=True
        )

        texture_quat = Texture(
            orientations=orientations,
            representation="quat",
            phase=cubic_phase,
        )

        texture_rotmat = texture_quat.to_representation("rotmat")

        assert texture_rotmat.representation == "rotmat"
        assert texture_rotmat.orientations.shape == (5, 3, 3)

    def test_rotmat_to_euler_conversion(self):
        """Test conversion from rotation matrix to Euler"""
        orientations = np.array([np.eye(3) for _ in range(5)])

        texture_rotmat = Texture(
            orientations=orientations,
            representation="rotmat",
            phase=cubic_phase,
        )

        texture_euler = texture_rotmat.to_representation("euler")

        assert texture_euler.representation == "euler"
        assert texture_euler.orientations.shape == (5, 3)

    def test_rotmat_to_quat_conversion(self):
        """Test conversion from rotation matrix to quaternion"""
        orientations = np.array([np.eye(3) for _ in range(5)])

        texture_rotmat = Texture(
            orientations=orientations,
            representation="rotmat",
            phase=cubic_phase,
        )

        texture_quat = texture_rotmat.to_representation("quat")

        assert texture_quat.representation == "quat"
        assert texture_quat.orientations.shape == (5, 4)

    def test_conversion_preserves_symmetry(self):
        """Test that conversions preserve symmetry"""
        orientations = rng.random((5, 3)) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=hex_phase,
        )

        texture_quat = texture.to_representation("quat")
        texture_rotmat = texture.to_representation("rotmat")

        assert texture_quat.phase.crystal_system == "hexagonal"
        assert texture_rotmat.phase.crystal_system == "hexagonal"

    def test_conversion_preserves_metadata(self):
        """Test that conversions preserve metadata"""
        orientations = rng.random((5, 3)) * 2 * np.pi
        metadata = {"source": "test", "value": 42}

        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
            metadata=metadata,
        )

        texture_quat = texture.to_representation("quat")

        assert texture_quat.metadata == metadata

        # Original metadata should be independent
        texture_quat.metadata["new_key"] = "new_value"
        assert "new_key" not in texture.metadata

    def test_same_representation_conversion(self):
        """Test conversion to same representation returns copy"""
        orientations = rng.random((5, 3)) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        texture2 = texture.to_representation("euler")

        assert texture2.representation == "euler"
        np.testing.assert_array_equal(texture2.orientations, texture.orientations)

        # Should be a copy, not the same object
        texture2.orientations[0, 0] = 999
        assert texture.orientations[0, 0] != 999

    def test_round_trip_conversion_euler_quat(self):
        """Test round-trip conversion Euler -> Quat -> Euler"""
        from synth_struct.orientation.rotation_converter import (
            euler_to_rotation_matrix,
        )

        original = rng.random((5, 3)) * 2 * np.pi
        texture = Texture(
            orientations=original,
            representation="euler",
            phase=cubic_phase,
        )

        texture_quat = texture.to_representation("quat")
        texture_back = texture_quat.to_representation("euler")

        # Should be approximately equal (within numerical precision)
        # Note: Euler angles are modulo 2π, so normalize
        R_original = euler_to_rotation_matrix(original)
        R_back = euler_to_rotation_matrix(texture_back.orientations)

        np.testing.assert_array_almost_equal(R_original, R_back, decimal=5)

    def test_round_trip_conversion_euler_rotmat(self):
        """Test round-trip conversion Euler -> RotMat -> Euler"""
        from synth_struct.orientation.rotation_converter import (
            euler_to_rotation_matrix,
        )

        original = rng.random((5, 3)) * 2 * np.pi
        texture = Texture(
            orientations=original,
            representation="euler",
            phase=cubic_phase,
        )

        texture_rotmat = texture.to_representation("rotmat")
        texture_back = texture_rotmat.to_representation("euler")

        R_original = euler_to_rotation_matrix(original)
        R_back = euler_to_rotation_matrix(texture_back.orientations)

        np.testing.assert_array_almost_equal(R_original, R_back, decimal=5)

    @pytest.mark.parametrize(
        "phase_list",
        [
            Phase.from_preset("default_cubic"),
            Phase.from_preset("default_hexagonal"),
            Phase.from_preset("Fe-fcc"),
            Phase.from_preset("Ti-alpha"),
        ],
    )
    def test_various_phases(self, phase_list):
        """Test that various symmetry strings are accepted"""
        orientations = rng.random((5, 3))
        texture = Texture(
            orientations=orientations, representation="euler", phase=phase_list
        )

        assert texture.phase == phase_list

    def test_large_number_of_orientations(self):
        """Test with large number of orientations"""
        orientations = rng.random((1000, 3)) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        assert texture.n_orientations == 1000

        # Test conversion still works
        texture_quat = texture.to_representation("quat")
        assert texture_quat.n_orientations == 1000

    def test_single_orientation(self):
        """Test with single orientation"""
        orientations = rng.random((1, 3)) * 2 * np.pi
        texture = Texture(
            orientations=orientations,
            representation="euler",
            phase=cubic_phase,
        )

        assert texture.n_orientations == 1
