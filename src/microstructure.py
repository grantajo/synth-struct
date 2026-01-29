# synth_struct/src/microstructure.py

import numpy as np

class Microstructure:
    def __init__(self, dimensions, resolution, units='micron'):
        """
        dimensions: tuple (nx, ny) or (nx, ny, nz) for 2D or 3D
        resolution: physical size per voxel
        """
        
        self.dimensions = tuple(dimensions)
        self.resolution = resolution
        self.units = units
        
        self.grain_ids = np.zeros(self.dimensions, dtype=np.int32) # 0 = background (e.g. unindexed EBSD)
        self.num_grains = 0
        
        self.fields = {} 
        self.metadata = {}
        
    def attach_field(self, name, array):
        """
        Attach per-grain or per-voxel data (orientations, stiffnesses, etc.)
        """
        self.fields[name] = array
        
    def get_field(self, name):
        return self.fields[name]
        
    def get_num_grains(self):
        return self.num_grains # exclude background (0)
        
        
    def get_grains_in_region(self, region_type='box', **kwargs):
        """
        Get grain IDs of grains that are in a specific region
        
        Args:
        - region_type: Type of region ('box', 'sphere', 'cylinder', 'custom_mask')
        - **kwargs: Region-specific parameters
        
        Region types and parameters:
        
        'box':
            - x_min, x_max: X bounds (default: 0, dimensions[0])
            - y_min, y_max: Y bounds (default: 0, dimensions[1])
            - z_min, z_max: Z bounds (default: 0, dimensions[2])
          
        'sphere':
            - center: [x, y, z] center coordinates (default: center of microstructure)
            - radius: Radius of sphere (required)
            
        'cylinder':
            - center: [x, y] center in XY plane (default: center of microstructure)
            - radius: Radius of cylinder (required)
            - z_min, z_max: Z bounds (default: 0, dimensions[0])
            - axis: Cylinder axis ('x', 'y', or 'z', default: 'z')
            
        'custom_mask':
            - mask: Boolean array same shape as microstructure (required)
            
        Returns:
            List of grain IDs in the region (excluding backgroudn grain 0)
            
        Examples:
            # Box region
            grains = micro.get_grains_in_region('box', x_min=10, x_max=50, y_min=20, y_max=80)
        
            # Sphere in center
            grains = micro.get_grains_in_region('sphere', center=[50, 50, 50], radius=30)
        
            # Cylinder along Z-axis
            grains = micro.get_grains_in_region('cylinder', center=[50, 50], radius=20)
        
            # Custom mask
            mask = my_custom_function(micro.grain_ids)
            grains = micro.get_grains_in_region('custom_mask', mask=mask)
        """
            
        if region_type == 'box':
            mask = self._create_box_mask(**kwargs)
            
        elif region_type == 'sphere':
            mask = self._create_sphere_mask(**kwargs)
            
        elif region_type == 'cylinder':
            mask = self._create_cylinder_mask(**kwargs)
            
        elif region_type == 'custom_mask':
            mask = kwargs.get('mask')
            if mask is None:
                raise ValueError("'mask' parameter is required for custom_mask region type")
            if mask.shape != self.grain_ids.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match microstructure shape {self.grain_ids.shape}")
                
        else:
            raise ValueError(f"Unknown region_type: {region_type}. "
                            f"Available types: 'box', 'sphere', 'cylinder', 'custom_mask'")
                            
        # Get unique grain IDs in the masked region
        grains_in_region = np.unique(self.grain_ids[mask])
        
        # Remove background (grain 0)
        grains_in_region = grains_in_region[grains_in_region > 0]
        
        return grains_in_region.tolist()  
        
    def _create_box_mask(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        """
        Create a box-shaped mask
        """
        
        # Handle 2D and 3D cases
        if len(self.dimensions) == 3:
            nx, ny, nz = self.dimensions
            x_min = x_min if x_min is not None else 0
            x_max = x_max if x_max is not None else nx
            y_min = y_min if y_min is not None else 0
            y_max = y_max if y_max is not None else ny
            z_min = z_min if z_min is not None else 0
            z_max = z_max if z_max is not None else nz
            
            x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
            mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
            
        
        else: # 2D Case
            nx, ny = self.dimensions
            x_min = x_min if x_min is not None else 0
            x_max = x_max if x_max is not None else nx
            y_min = y_min if y_min is not None else 0
            y_max = y_max if y_max is not None else ny
            
            x, y = np.mgrid[0:nx, 0:ny]
            mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            
        return mask
        
    def _create_sphere_mask(self, center=None, radius=None):
        """
        Create a spherical mask
        """
        
        if radius is None:
            raise ValueError("'radius' parameters is required for sphere region type")
            
        if len(self.dimensions) == 3:
            nx, ny, nz = self.dimensions
            if center is None:
                center = [nx/2, ny/2, nz/2]
        
            if len(center) != 3:
                raise ValueError(f"Center for 3D sphere must have 3 coordinates, got {len(center)}")
            
            x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
            distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            mask = distances < radius
            
        else: # 2D - circle
            nx, ny = self.dimensions
            if center is None:
                center = [nx/2, ny/2]
                
            if len(center) != 2:
                raise ValueError(f"Center for 2D circle must have 2 coordinates, got {len(center)}")
                
            x, y = np.mgrid[0:nx, 0:ny]
            distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            mask = distances < radius
            
        return mask
        
    def _create_cylinder_mask(self, center=None, radius=None, z_min=None, z_max=None, axis='z'):
        """
        Create a cylindrical mask
        """
        
        if radius is None:
            raise ValueError("'radius' parameters is required for cylinder region type")
            
        if len(self.dimensions) != 3:
            raise ValueError("Cylinder region only supported for 3D microstructures")
            
        nx, ny, nz = self.dimensions
        
        if axis == 'z':
            if center is None:
                center = [nx/2, ny/2]
            z_min = z_min if z_min is not None else 0
            z_max = z_max if z_max is not None else nz
        
            z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
            radial_dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            mask = (radial_dist < radius) & (z >= z_min) & (z < z_max)
        
        elif axis == 'y':
            if center is None:
                center = [nx/2, nz/2]
            y_min = z_min if z_min is not None else 0
            y_max = z_max if z_max is not None else ny
            
            z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
            radial_dist = np.sqrt((x - center[0])**2 + (z - center[1])**2)
            mask = (radial_dist < radius) & (y >= y_min) & (y < y_max)
        
        elif axis == 'x':
            if center is None:
                center = [ny/2, nz/2]
            x_min = z_min if z_min is not None else 0
            x_max = z_max if z_max is not None else nx
            
            z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
            radial_dist = np.sqrt((y - center[0])**2 + (z - center[1])**2)
            mask = (radial_dist < radius) & (x >= x_min) & (x < x_max)
            
        else:
            raise ValueError(f"Axis must be 'x', 'y', or 'z', got '{axis}'")
            
        return mask 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
