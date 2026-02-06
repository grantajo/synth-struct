# synth_struct/tests/test_microstructure.py

import pytest
import numpy as np

from synth_struct.microstructure import Microstructure


class TestMicrostructure:
    def test_initialization_2d(self):
        """Test 2D microstructure initialization"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)

        assert micro.dimensions == (50, 50)
        assert micro.resolution == (1.0)
        assert micro.units == "micron"

        assert micro.grain_ids.shape == (50, 50)
        assert micro.grain_ids.dtype == np.int32
        assert np.all(micro.grain_ids == 0)  # background initially

    def test_initialization_3d(self):
        """Test 3D microstructure initialization"""
        micro = Microstructure(dimensions=(25, 25, 25), resolution=1.0, units="mm")

        assert micro.dimensions == (25, 25, 25)
        assert micro.resolution == 1.0
        assert micro.units == "mm"

        assert micro.grain_ids.shape == (25, 25, 25)
        assert micro.grain_ids.dtype == np.int32

    def test_attach_and_get_field(self):
        """Test attaching and retrieving fields"""
        micro = Microstructure(dimensions=(25, 25), resolution=1.0)

        # Create simple field
        orientations = np.random.rand(10, 10, 3)

        # Attach field
        micro.attach_field("orientations", orientations)

        # Retrieve field
        retrieved_field = micro.get_field("orientations")

        # Check field retrieval
        np.testing.assert_array_equal(orientations, retrieved_field)

    def test_multiple_fields(self):
        """Test attaching multiple fields"""
        micro = Microstructure(dimensions=(10, 10), resolution=1.0)

        # Attach multiple fields
        orientations = np.random.rand(10, 10, 3)
        stiffness = np.random.rand(10, 10, 6, 6)

        micro.attach_field("orientations", orientations)
        micro.attach_field("stiffness", stiffness)

        # Check both fields can be retrieved
        assert "orientations" in micro.fields
        assert "stiffness" in micro.fields

        np.testing.assert_array_equal(micro.get_field("orientations"), orientations)
        np.testing.assert_array_equal(micro.get_field("stiffness"), stiffness)

    def test_invalid_field_retrieval(self):
        """Test retrieving a non-existent field"""
        micro = Microstructure(dimensions=(10, 10), resolution=1.0)

        with pytest.raises(KeyError):
            micro.get_field("non_existent_field")

    @pytest.mark.parametrize(
        "dims",
        [
            (10, 10),  # 2D
            (10, 10, 10),  # 3D
            (5, 5),  # Small 2D
            (20, 20, 20),  # Large 3D
        ],
    )
    def test_dimension_variations(self, dims):
        """Test microstructure initialization with different dimension sizes"""
        micro = Microstructure(dimensions=dims, resolution=0.1)

        assert micro.dimensions == dims
        assert micro.grain_ids.shape == dims

    def test_metadata(self):
        """Test metadata functionality"""
        micro = Microstructure(dimensions=(10, 10), resolution=1.0)

        # Add metadata
        micro.metadata["sample_id"] = "test_sample"
        micro.metadata["creation_date"] = "2024-02-02"

        # Check metadata
        assert micro.metadata["sample_id"] == "test_sample"
        assert micro.metadata["creation_date"] == "2024-02-02"
