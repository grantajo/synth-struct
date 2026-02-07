# synth_struct/tests/test_stiffness.py

import numpy as np
import pytest
from synth_struct.stiffness.stiffness import Stiffness

class TestStiffness:

    def test_initialization_basic(self):
        """Test Stiffness initialization with basic parameters"""
        n_grains = 10
        stiffness_tensors = np.random.rand(n_grains, 6, 6) * 100
        
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        assert stiffness.crystal_structure == "cubic"
        assert stiffness.n_tensors == n_grains
        np.testing.assert_array_equal(stiffness.stiffness_tensors, stiffness_tensors)

    def test_initialization_with_metadata(self):
        """Test Stiffness initialization with metadata"""
        stiffness_tensors = np.random.rand(5, 6, 6) * 100
        metadata = {"C11": 160.0, "C12": 92.0, "C44": 47.0}

        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic",
            metadata=metadata,
        )

        assert stiffness.metadata == metadata
        assert stiffness.metadata["C11"] == 160.0

    def test_invalid_stiffness_type(self):
        """Test that non-array stiffness tensors raise TypeError"""
        with pytest.raises(TypeError, match="must be a NumPy array"):
            Stiffness(
                stiffness_tensors=[[[1, 2], [3, 4]]],  # List, not array
                crystal_structure="cubic",
            )

    def test_invalid_stiffness_shape_ndim(self):
        """Test that wrong dimensionality raises ValueError"""
        with pytest.raises(ValueError, match="must have shape"):
            Stiffness(
                stiffness_tensors=np.random.rand(6, 6),  # 2D instead of 3D
                crystal_structure="cubic",
            )

    def test_invalid_stiffness_shape_size(self):
        """Test that wrong tensor size raises ValueError"""
        with pytest.raises(ValueError, match="must be 6x6"):
            Stiffness(
                stiffness_tensors=np.random.rand(5, 3, 3),  # 3x3 instead of 6x6
                crystal_structure="cubic",
            )

    def test_invalid_crystal_structure_type(self):
        """Test that non-string crystal_structure raises TypeError"""
        stiffness_tensors = np.random.rand(5, 6, 6)

        with pytest.raises(TypeError, match="crystal_structure must be a string"):
            Stiffness(
                stiffness_tensors=stiffness_tensors,
                crystal_structure=123,  # Not a string
            )

    def test_copy(self):
        """Test that copy creates independent copy"""
        stiffness_tensors = np.random.rand(5, 6, 6) * 100
        metadata = {"test": "data"}

        stiffness1 = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic",
            metadata=metadata,
        )

        stiffness2 = stiffness1.copy()

        # Modify original
        stiffness1.stiffness_tensors[0, 0, 0] = 999
        stiffness1.metadata["new_key"] = "new_value"

        # Copy should be unchanged
        assert stiffness2.stiffness_tensors[0, 0, 0] != 999
        assert "new_key" not in stiffness2.metadata

    def test_subset(self):
        """Test subset extraction"""
        stiffness_tensors = np.random.rand(10, 6, 6) * 100
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        indices = np.array([0, 2, 4, 6])
        subset_stiffness = stiffness.subset(indices)

        assert subset_stiffness.n_tensors == 4
        np.testing.assert_array_equal(
            subset_stiffness.stiffness_tensors, stiffness_tensors[indices]
        )
        assert subset_stiffness.crystal_structure == "cubic"

    def test_subset_preserves_metadata(self):
        """Test that subset preserves metadata"""
        stiffness_tensors = np.random.rand(10, 6, 6) * 100
        metadata = {"source": "test"}

        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic",
            metadata=metadata,
        )

        subset_stiffness = stiffness.subset(np.array([0, 1, 2]))

        assert subset_stiffness.metadata == metadata

    def test_n_tensors_property(self):
        """Test n_tensors property"""
        for n in [1, 5, 100, 1000]:
            stiffness_tensors = np.random.rand(n, 6, 6)
            stiffness = Stiffness(
                stiffness_tensors=stiffness_tensors,
                crystal_structure="cubic"
            )
            assert stiffness.n_tensors == n

    @pytest.mark.parametrize(
        "crystal_structure", ["cubic", "hexagonal", "isotropic", "tetragonal", "orthorhombic"]
    )
    def test_various_crystal_structures(self, crystal_structure):
        """Test that various crystal structure strings are accepted"""
        stiffness_tensors = np.random.rand(5, 6, 6)
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure=crystal_structure
        )

        assert stiffness.crystal_structure == crystal_structure

    def test_single_tensor(self):
        """Test with single stiffness tensor"""
        stiffness_tensors = np.random.rand(1, 6, 6) * 100
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        assert stiffness.n_tensors == 1

    def test_large_number_of_tensors(self):
        """Test with large number of stiffness tensors"""
        stiffness_tensors = np.random.rand(1000, 6, 6) * 100
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        assert stiffness.n_tensors == 1000

    def test_symmetric_stiffness_tensor(self):
        """Test that stiffness tensors can be symmetric"""
        n = 5
        stiffness_tensors = np.zeros((n, 6, 6))
        
        # Create symmetric tensors
        for i in range(n):
            C = np.random.rand(6, 6) * 100
            stiffness_tensors[i] = (C + C.T) / 2  # Make symmetric

        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        # Check symmetry is preserved
        for i in range(n):
            np.testing.assert_array_almost_equal(
                stiffness.stiffness_tensors[i],
                stiffness.stiffness_tensors[i].T
            )

    def test_empty_metadata(self):
        """Test that empty metadata works correctly"""
        stiffness_tensors = np.random.rand(5, 6, 6)
        stiffness = Stiffness(
            stiffness_tensors=stiffness_tensors,
            crystal_structure="cubic"
        )

        assert stiffness.metadata == {}
        assert len(stiffness.metadata) == 0
