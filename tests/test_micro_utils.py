# synth_struct/tests/test_micro_utils.py

import pytest
import numpy as np

from src.microstructure import Microstructure
from src.micro_utils import (
    get_grains_in_region,
    _create_box_mask,
    _create_sphere_mask,
    _create_cylinder_mask,
)

class TestGetGrains:
    
    @pytest.fixture
    def micro_2d_simple(self):
        """Create a simple 2D microstructure with known grain positions"""
        micro = Microstructure(dimensions=(100,100), resolution=1.0)
        
        # Create a simple grain structure
        micro.grain_ids[10:30, 10:30] = 1  # Grain 1 in corner
        micro.grain_ids[40:60, 40:60] = 2  # Grain 2 in center
        micro.grain_ids[70:90, 70:90] = 3  # Grain 3 in opposite corner
        
        return micro
        
    @pytest.fixture
    def micro_3d_simple(self):
        """Create a simple 3D microstructure with known grain positions"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        # Create grains in different regions
        micro.grain_ids[10:30, 10:30, 10:30] = 1
        micro.grain_ids[40:60, 40:60, 40:60] = 2
        micro.grain_ids[70:90, 70:90, 70:90] = 3
        return micro
        
    def test_box_region_2d_single_grain(self, micro_2d_simple):
        """Test box region that captures a single grain in 2D"""
        grains = get_grains_in_region(
            micro_2d_simple, 'box',
            x_min=10, x_max=30, y_min=10, y_max=30
        )
        assert len(grains) == 1
        assert 1 in grains
        
    def test_box_region_2d_multiple_grains(self, micro_2d_simple):
        """Test box region that captures multiple grains in 2D"""
        grains = get_grains_in_region(
            micro_2d_simple, 'box',
            x_min=0, x_max=70, y_min=0, y_max=70
        )
        assert len(grains) == 2
        assert 1 in grains and 2 in grains
        
    def test_box_region_2d_all_grains(self, micro_2d_simple):
        """Test box region that captures all grains in 2D"""
        grains = get_grains_in_region(micro_2d_simple, 'box')  # Default to full domain
        assert len(grains) == 3
        assert set(grains) == {1, 2, 3}
    
    def test_box_region_3d_single_grain(self, micro_3d_simple):
        """Test box region that captures a single grain in 3D"""
        grains = get_grains_in_region(
            micro_3d_simple, 'box',
            x_min=10, x_max=30, y_min=10, y_max=30, z_min=10, z_max=30
        )
        assert len(grains) == 1
        assert 1 in grains
    
    def test_box_region_empty(self, micro_2d_simple):
        """Test box region that contains no grains (only background)"""
        grains = get_grains_in_region(
            micro_2d_simple, 'box',
            x_min=0, x_max=5, y_min=0, y_max=5
        )
        assert len(grains) == 0
    
    # Sphere/Circle region tests
    def test_sphere_region_2d_center(self, micro_2d_simple):
        """Test circular region in 2D centered on a grain"""
        grains = get_grains_in_region(
            micro_2d_simple, 'sphere',
            center=[50, 50], radius=15
        )
        assert len(grains) == 1
        assert 2 in grains
    
    def test_sphere_region_2d_large(self, micro_2d_simple):
        """Test large circular region that captures multiple grains"""
        grains = get_grains_in_region(
            micro_2d_simple, 'sphere',
            center=[50, 50], radius=50
        )
        assert len(grains) >= 2  # Should capture center and at least one other
    
    def test_sphere_region_3d_center(self, micro_3d_simple):
        """Test spherical region in 3D centered on a grain"""
        grains = get_grains_in_region(
            micro_3d_simple, 'sphere',
            center=[50, 50, 50], radius=15
        )
        assert len(grains) == 1
        assert 2 in grains
    
    def test_sphere_region_default_center_2d(self, micro_2d_simple):
        """Test sphere with default center (microstructure center) in 2D"""
        grains = get_grains_in_region(
            micro_2d_simple, 'sphere',
            radius=15
        )
        assert 2 in grains  # Center grain should be captured
    
    def test_sphere_region_default_center_3d(self, micro_3d_simple):
        """Test sphere with default center (microstructure center) in 3D"""
        grains = get_grains_in_region(
            micro_3d_simple, 'sphere',
            radius=15
        )
        assert 2 in grains  # Center grain should be captured
    
    def test_sphere_region_no_radius_error(self, micro_2d_simple):
        """Test that missing radius parameter raises error"""
        with pytest.raises(ValueError, match="radius.*required"):
            get_grains_in_region(micro_2d_simple, 'sphere', center=[50, 50])
    
    def test_sphere_region_wrong_center_dimension_2d(self, micro_2d_simple):
        """Test that wrong center dimensions raise error in 2D"""
        with pytest.raises(ValueError, match="2 coordinates"):
            get_grains_in_region(
                micro_2d_simple, 'sphere',
                center=[50, 50, 50], radius=10
            )
    
    def test_sphere_region_wrong_center_dimension_3d(self, micro_3d_simple):
        """Test that wrong center dimensions raise error in 3D"""
        with pytest.raises(ValueError, match="3 coordinates"):
            get_grains_in_region(
                micro_3d_simple, 'sphere',
                center=[50, 50], radius=10
            )
    
    # Cylinder region tests
    def test_cylinder_region_z_axis(self, micro_3d_simple):
        """Test cylinder along z-axis"""
        grains = get_grains_in_region(
            micro_3d_simple, 'cylinder',
            center=[50, 50], radius=15, axis='z'
        )
        assert 2 in grains
    
    def test_cylinder_region_x_axis(self, micro_3d_simple):
        """Test cylinder along x-axis"""
        grains = get_grains_in_region(
            micro_3d_simple, 'cylinder',
            center=[50, 50], radius=15, axis='x'
        )
        assert len(grains) >= 1
    
    def test_cylinder_region_y_axis(self, micro_3d_simple):
        """Test cylinder along y-axis"""
        grains = get_grains_in_region(
            micro_3d_simple, 'cylinder',
            center=[50, 50], radius=15, axis='y'
        )
        assert len(grains) >= 1
    
    def test_cylinder_region_with_bounds(self, micro_3d_simple):
        """Test cylinder with specified z bounds"""
        grains = get_grains_in_region(
            micro_3d_simple, 'cylinder',
            center=[50, 50], radius=15, z_min=40, z_max=60
        )
        assert 2 in grains
    
    def test_cylinder_region_default_center(self, micro_3d_simple):
        """Test cylinder with default center"""
        grains = get_grains_in_region(
            micro_3d_simple, 'cylinder',
            radius=15
        )
        assert 2 in grains
    
    def test_cylinder_region_no_radius_error(self, micro_3d_simple):
        """Test that missing radius parameter raises error"""
        with pytest.raises(ValueError, match="radius.*required"):
            get_grains_in_region(micro_3d_simple, 'cylinder', center=[50, 50])
    
    def test_cylinder_region_2d_error(self, micro_2d_simple):
        """Test that cylinder on 2D microstructure raises error"""
        with pytest.raises(ValueError, match="only supported for 3D"):
            get_grains_in_region(
                micro_2d_simple, 'cylinder',
                center=[50, 50], radius=10
            )
    
    def test_cylinder_region_invalid_axis(self, micro_3d_simple):
        """Test that invalid axis raises error"""
        with pytest.raises(ValueError, match="Axis must be"):
            get_grains_in_region(
                micro_3d_simple, 'cylinder',
                center=[50, 50], radius=10, axis='w'
            )
    
    def test_cylinder_wrong_center_dimension(self, micro_3d_simple):
        """Test that wrong center dimensions raise error for cylinder"""
        with pytest.raises(ValueError, match="2 coordinates"):
            get_grains_in_region(
                micro_3d_simple, 'cylinder',
                center=[50, 50, 50], radius=10, axis='z'
            )
    
    # Custom mask tests
    def test_custom_mask_2d(self, micro_2d_simple):
        """Test custom mask in 2D"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # Mask over grain 2
        
        grains = get_grains_in_region(micro_2d_simple, 'custom_mask', mask=mask)
        assert len(grains) == 1
        assert 2 in grains
    
    def test_custom_mask_3d(self, micro_3d_simple):
        """Test custom mask in 3D"""
        mask = np.zeros((100, 100, 100), dtype=bool)
        mask[40:60, 40:60, 40:60] = True  # Mask over grain 2
        
        grains = get_grains_in_region(micro_3d_simple, 'custom_mask', mask=mask)
        assert len(grains) == 1
        assert 2 in grains
    
    def test_custom_mask_no_mask_error(self, micro_2d_simple):
        """Test that missing mask parameter raises error"""
        with pytest.raises(ValueError, match="mask.*required"):
            get_grains_in_region(micro_2d_simple, 'custom_mask')
    
    def test_custom_mask_wrong_shape_error(self, micro_2d_simple):
        """Test that wrong mask shape raises error"""
        mask = np.zeros((50, 50), dtype=bool)  # Wrong shape
        with pytest.raises(ValueError, match="shape.*doesn't match"):
            get_grains_in_region(micro_2d_simple, 'custom_mask', mask=mask)
    
    # Error handling tests
    def test_invalid_region_type(self, micro_2d_simple):
        """Test that invalid region type raises error"""
        with pytest.raises(ValueError, match="Unknown region_type"):
            get_grains_in_region(micro_2d_simple, 'invalid_type')
    
    def test_case_insensitive_region_type(self, micro_2d_simple):
        """Test that region type is case insensitive"""
        grains1 = get_grains_in_region(micro_2d_simple, 'BOX')
        grains2 = get_grains_in_region(micro_2d_simple, 'box')
        assert np.array_equal(grains1, grains2)


class TestCreateBoxMask:
    """Test suite for _create_box_mask function"""
    
    def test_box_mask_2d_full_domain(self):
        """Test box mask covering full 2D domain"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        mask = _create_box_mask(micro)
        assert mask.shape == (50, 50)
        assert np.all(mask)  # All True
    
    def test_box_mask_2d_partial(self):
        """Test box mask covering partial 2D domain"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        mask = _create_box_mask(micro, x_min=10, x_max=20, y_min=15, y_max=25)
        assert mask.shape == (50, 50)
        assert np.sum(mask) == 10 * 10  # 10x10 box
    
    def test_box_mask_3d_full_domain(self):
        """Test box mask covering full 3D domain"""
        micro = Microstructure(dimensions=(30, 30, 30), resolution=1.0)
        mask = _create_box_mask(micro)
        assert mask.shape == (30, 30, 30)
        assert np.all(mask)
    
    def test_box_mask_3d_partial(self):
        """Test box mask covering partial 3D domain"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        mask = _create_box_mask(
            micro,
            x_min=10, x_max=20,
            y_min=10, y_max=20,
            z_min=10, z_max=20
        )
        assert mask.shape == (50, 50, 50)
        assert np.sum(mask) == 10 * 10 * 10  # 10x10x10 box
    
    def test_box_mask_defaults(self):
        """Test that default bounds work correctly"""
        micro = Microstructure(dimensions=(40, 40), resolution=1.0)
        mask = _create_box_mask(micro, x_min=10)  # Only specify x_min
        assert mask.shape == (40, 40)
        assert np.sum(mask) == 30 * 40  # (40-10) * 40


class TestCreateSphereMask:
    """Test suite for _create_sphere_mask function"""
    
    def test_sphere_mask_2d_center(self):
        """Test circular mask in 2D at center"""
        micro = Microstructure(dimensions=(100, 100), resolution=1.0)
        mask = _create_sphere_mask(micro, center=[50, 50], radius=10)
        assert mask.shape == (100, 100)
        # Check approximate area (π * r²)
        assert 250 < np.sum(mask) < 350  # ~314
    
    def test_sphere_mask_2d_default_center(self):
        """Test circular mask with default center"""
        micro = Microstructure(dimensions=(100, 100), resolution=1.0)
        mask = _create_sphere_mask(micro, radius=10)
        assert mask.shape == (100, 100)
        assert mask[50, 50]  # Center should be True
    
    def test_sphere_mask_3d_center(self):
        """Test spherical mask in 3D at center"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_sphere_mask(micro, center=[50, 50, 50], radius=10)
        assert mask.shape == (100, 100, 100)
        # Check approximate volume (4/3 * π * r³)
        assert 3500 < np.sum(mask) < 4500  # ~4189
    
    def test_sphere_mask_no_radius_error(self):
        """Test that missing radius raises error"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="radius.*required"):
            _create_sphere_mask(micro, center=[25, 25])
    
    def test_sphere_mask_wrong_center_2d(self):
        """Test that wrong center dimension raises error in 2D"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="2 coordinates"):
            _create_sphere_mask(micro, center=[25, 25, 25], radius=10)
    
    def test_sphere_mask_wrong_center_3d(self):
        """Test that wrong center dimension raises error in 3D"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="3 coordinates"):
            _create_sphere_mask(micro, center=[25, 25], radius=10)


class TestCreateCylinderMask:
    """Test suite for _create_cylinder_mask function"""
    
    def test_cylinder_mask_z_axis(self):
        """Test cylinder mask along z-axis"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_cylinder_mask(micro, center=[50, 50], radius=10, axis='z')
        assert mask.shape == (100, 100, 100)
        # Check that mask extends through z
        assert mask[50, 50, 0]
        assert mask[50, 50, 99]
    
    def test_cylinder_mask_x_axis(self):
        """Test cylinder mask along x-axis"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_cylinder_mask(micro, center=[50, 50], radius=10, axis='x')
        assert mask.shape == (100, 100, 100)
        # Check that mask extends through x
        assert mask[0, 50, 50]
        assert mask[99, 50, 50]
    
    def test_cylinder_mask_y_axis(self):
        """Test cylinder mask along y-axis"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_cylinder_mask(micro, center=[50, 50], radius=10, axis='y')
        assert mask.shape == (100, 100, 100)
        # Check that mask extends through y
        assert mask[50, 0, 50]
        assert mask[50, 99, 50]
    
    def test_cylinder_mask_with_bounds(self):
        """Test cylinder mask with z bounds"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_cylinder_mask(
            micro,
            center=[50, 50], radius=10,
            z_min=30, z_max=70, axis='z'
        )
        assert mask.shape == (100, 100, 100)
        assert mask[50, 50, 50]  # Inside bounds
        assert not mask[50, 50, 20]  # Outside bounds
        assert not mask[50, 50, 80]  # Outside bounds
    
    def test_cylinder_mask_default_center(self):
        """Test cylinder with default center"""
        micro = Microstructure(dimensions=(100, 100, 100), resolution=1.0)
        mask = _create_cylinder_mask(micro, radius=10)
        assert mask[50, 50, 50]  # Center should be True
    
    def test_cylinder_mask_no_radius_error(self):
        """Test that missing radius raises error"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="radius.*required"):
            _create_cylinder_mask(micro, center=[25, 25])
    
    def test_cylinder_mask_2d_error(self):
        """Test that cylinder on 2D raises error"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="only supported for 3D"):
            _create_cylinder_mask(micro, center=[25, 25], radius=10)
    
    def test_cylinder_mask_invalid_axis(self):
        """Test that invalid axis raises error"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="Axis must be"):
            _create_cylinder_mask(micro, center=[25, 25], radius=10, axis='w')
    
    def test_cylinder_mask_wrong_center_dimension(self):
        """Test that wrong center dimension raises error"""
        micro = Microstructure(dimensions=(50, 50, 50), resolution=1.0)
        with pytest.raises(ValueError, match="2 coordinates"):
            _create_cylinder_mask(
                micro,
                center=[25, 25, 25], radius=10, axis='z'
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_radius_sphere(self):
        """Test sphere with zero radius"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        mask = _create_sphere_mask(micro, center=[25, 25], radius=0)
        # Should contain at most the center point
        assert np.sum(mask) <= 1
    
    def test_very_large_radius(self):
        """Test sphere with radius larger than domain"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        mask = _create_sphere_mask(micro, center=[25, 25], radius=1000)
        # Should cover entire domain
        assert np.all(mask)
    
    def test_box_with_inverted_bounds(self):
        """Test that inverted bounds create empty mask"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        mask = _create_box_mask(micro, x_min=30, x_max=20)  # x_max < x_min
        assert np.sum(mask) == 0
    
    def test_boundary_grain(self):
        """Test grains at domain boundaries"""
        micro = Microstructure(dimensions=(50, 50), resolution=1.0)
        micro.grain_ids[0:5, 0:5] = 1  # Corner grain
        grains = get_grains_in_region(
            micro, 'box',
            x_min=0, x_max=10, y_min=0, y_max=10
        )
        assert 1 in grains
