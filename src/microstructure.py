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
            
    def gen_voronoi(self, num_grains, seed=None, chunk_size=1_000_000):
        """
        Generate grains with a standard Voronoi tesselation
        
        Args:
        - num_grains: Number of grains to generate
        - seed: Random seed for reproducibility
        - chunk_size: For memory efficient generation
        """
        if seed:
            np.random.seed(seed)
        
        # Number of dimensions
        ndim = len(self.dimensions)
        self.num_grains = num_grains
        
        # Generate random seed points
        seeds = np.random.rand(num_grains, ndim) * np.array(self.dimensions)
        tree = cKDTree(seeds)
        
        # Total number of voxels
        total_voxels = int(np.prod(self.dimensions))
        grain_ids_flat = np.empty(total_voxels, dtype=np.int32)
        
        # Process in chunks
        for start in range(0, total_voxels, chunk_size):
            end = min(start+chunk_size, total_voxels)
            
            # Convert flat indices to coordinates
            flat_indices = np.arange(start, end)
            coords = np.column_stack(
                np.unravel_index(flat_indices, self.dimensions)
            )
            
            # Find nearest seed for each coordinate
            distances, indices = tree.query(chunk_coords)
            grain_ids_flat[start:end] = indices + 1
            
        # Reshape back to original dimensions
        self.grain_ids = grain_ids_flat.reshape(self.dimensions)
        
    def gen_voronoi_w(self, num_grains, grain_shapes='spherical', shape_params=None, seed=None):
        """
        Generate grains with controlled shapes using weighted Voronoi tesselation
        
        Args:
        - num_grains: Number of grains to generate
        - grain_shapes: Shape type ('ellipsoidal', 'columnar', 'equiaxed', 'mixed', 'custom')
        - shape_params: Dictionary of shape parameters (dependent on grain_shapes)
        - seed: Random seed for reproducibility
        
        Shape types and parameters:
        
        'ellipsoidal': Elongated grains
            - aspect_ratio_mean: Mean aspect ratio (default: 2.0)
            - aspect_ratio_std: Std dev of aspect ratio (default: 0.5)
            - orientation: Preferred elongation direction ('x', 'y', 'z', 'random')
            - base_size: size of the short axis (default: 10.0)
        'columnar': Column-like grains along one axis, only works for 3D
            - axis: Growth direction ('x', 'y', or 'z', default: 'z')
            - aspect_ratio: Length/weidth ratio (default: 3.0)
            - base_size: size of the short axis (default: 10.0)
        'mixed': Mixture of shapes
            - ellipsoid_fraction: Fraction of ellipsoidal grains (default: 0.5)
            - aspect_ratio_mean: For ellipsoidal grains (default: 5.0)
            - base_size: size of the short axis (default: 10.0)
        
        Maybe figure out a way to add in a custom
        'custom': User-defined weights **NOT IMPLEMENTED**
            - weights: Array of weights for each grain (required)
        """
        if seed:
            np.random.seed(seed)
            
        # Set default parameters
        if shape_params is None:
            shape_params = {}
            
        ndim = len(self.dimensions)
        
        seeds = np.random.rand(num_grains, ndim) * np.array(self.dimensions)
        
        # Generate weights based on shape type
            
        if grain_shapes == 'ellipsoidal':
            scale_factors, rotations = self._generate_ellipsoidal_params(num_grains, shape_params)
        
        elif grain_shapes == 'columnar':
            scale_factors, rotations = self._generate_columnar_params(num_grains, shape_params)
            
        elif grain_shapes == 'mixed':
            scale_factors, rotations = self._generate_mixed_params(num_grains, shape_params)
                
        else:
            raise ValueError(f"Unknown grain_shapes: {grain_shapes}")
            
        # Perform weighted Voronoi tessellation
        self._anisotropic_voronoi_assignment(seeds, scale_factors, rotations)
        
        print(f"Generated {num_grains} grains with {grain_shapes} morphology")
        
        
    def _generate_ellipsoidal_params(self, num_grains, params):
        """Generate weights and directions for ellipsoidal grains"""
        aspect_ratio_mean = params.get('aspect_ratio_mean', 5.0)
        aspect_ratio_std = params.get('aspect_ratio_std', 0.5)
        orientation = params.get('orientation', 'random')
        base_size = params.get('base_size', 10.0)
        
        ndim = len(self.dimensions)
        
        # Generate aspect ratios for each grain
        aspect_ratios = np.random.normal(aspect_ratio_mean, aspect_ratio_std, num_grains)
        aspect_ratios = np.clip(aspect_ratios, 1.5, 10.0)  
        
        scale_factors = np.zeros((num_grains, ndim))
        rotations = []
        
        for i in range(num_grains):
            if ndim == 3:
                long_axis = base_size * aspect_ratios[i]
                short_axis = base_size
                scale_factors[i] = [short_axis, short_axis, long_axis]
                
                if orientation == 'random':
                    angles = np.random.uniform(0, 2*np.pi, 3)
                    R = self._euler_to_rotation_matrix_3d(angles)
                elif orientation == 'x':
                    R = self._rotation_z_to_x()
                elif orientation == 'y':
                    R = self._rotation_z_to_y()
                elif orientation == 'z':
                    R = np.eye(3)
                else:
                    R = np.eye(3)
                    
                rotations.append(R)
            
            else: # 2D
                long_axis = base_size * aspect_ratios[i]
                short_axis = base_size
                scale_factors[i] = [short_axis, long_axis]
                
                if orientation == 'random':
                    angle = np.random.uniform(0, 2*np.pi)
                    R = self._rotation_matrix_2d(angle)
                elif orientation == 'x':
                    R = self._rotation_matrix_2d(0)
                elif orientation == 'y':
                    R = self._rotation_matrix_2d(np.pi/2)
                else:
                    R = np.eye(2)
                    
                rotations.append(R)
                
        return scale_factors, rotations
        
        
    def _generate_columnar_params(self, num_grains, params):
        """Generate weights for columnar grains"""
        axis = params.get('axis', 'z')
        aspect_ratio = params.get('aspect_ratio', 5.0)
        base_size = params.get('base_size', 8.0)
        
        ndim = len(self.dimensions)
        
        if ndim != 3:
            raise ValueError("Columnar grains only supported for 3D microstructures")
            
        scale_factors = np.zeros((num_grains, 3))
        rotations = []
        
        long_axis = base_size * aspect_ratio
        short_axis = base_size
        
        for i in range(num_grains):
            this_long = long_axis * np.random.uniform(0.8, 1.2)
            this_short = short_axis * np.random.uniform(0.8, 1.2)
            
            scale_factors[i] = [this_short, this_short, this_long]
            
            if axis == 'x':
                R = self._rotation_z_to_x()
            elif axis == 'y':
                R = self._rotation_z_to_y()
            else:
                R = np.eye(3)
                
            rotations.append(R)
            
        return scale_factors, rotations
        
        
    def _generate_mixed_params(self, num_grains, params):
        """Generate weights for mixed grain morphologies"""
        ellipsoid_fraction = params.get('ellipsoid_fraction', 0.5)
        aspect_ratio_mean = params.get('aspect_ratio_mean', 5.0)
        base_size = params.get('base_size', 10.0)
        
        ndim = len(self.dimensions)
        num_ellipsoidal = int(num_grains * ellipsoid_fraction)
        
        scale_factors = np.zeros((num_grains, ndim))
        rotations = []
        
        for i in range(num_grains):
            if i < num_ellipsoidal: # Ellipsoidal
                aspect_ratio = np.random.normal(aspect_ratio_mean, 0.5)
                aspect_ratio = np.clip(aspect_ratio, 1.5, 8.0)
                
                if ndim == 3:
                    scale_factors[i] = [base_size, base_size, base_size * aspect_ratio]
                    angles = np.random.uniform(0, 2*np.pi, 3)
                    R = self._euler_to_rotation_matrix_3d(angles)
                else: # 2D
                    scale_factors[i] = [base_size, base_size * aspect_ratio]
                    angle = np.random.uniform(0, 2*np.pi)
                    R = self._rotation_matrix_2d(angle)
                    
            else: # Spherical
                scale_factors[i] = base_size
                R = np.eye(ndim)
                
            rotations.append(R)
            
        return scale_factors, rotations
    
    
    
    def _anisotropic_voronoi_assignment(self, seeds, scale_factors, rotations):
        """
        Assign voxels using anisotropic distance metric
        
        For each point p and seed s with scaling S and rotation R:
        d_aniso = ||S^-1 * R^T * (p - s)||^2
        """
        
        ndim = len(self.dimensions)
        total_voxels = np.prod(self.dimensions)
        grain_ids_flat = np.zeros(total_voxels, dtype=np.int32)
        
        chunk_size = 100_000
        
        print(f"Performing anisotropic Voronoi tessellation...")
        
        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)
            
            chunk_indices = np.arange(start, end)
            chunk_coords = np.unravel_index(chunk_indices, self.dimensions)
            chunk_coords = np.column_stack(chunk_coords).astype(float)
            
            distances = np.zeros((len(chunk_coords), len(seeds)))
            
            for i, (seed, scale, R) in enumerate(zip(seeds, scale_factors, rotations)):
                diff = chunk_coords - seed
                diff_rotated = diff @ R.T # R^T * (p-s)
                diff_scaled = diff_rotated / scale
                
                distances[:, i] = np.sum(diff_scaled**2, axis=1)
                
            grain_assignment = np.argmin(distances, axis=1) + 1
            grain_ids_flat[start:end] = grain_assignment
            
            if (start // chunk_size) % 10 == 0:
                progress = 100 * end / total_voxels
                print(f"  Progress: {progress:.1f}%")
                
        self.grain_ids = grain_ids_flat.reshape(self.dimensions)
        print("Done!")
        
        
    def _rotation_matrix_2d(self, angle):
        """2D rotation matrix"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
        
    def _euler_to_rotation_matrix_3d(self, angles):
        """3D rotation matrix from Euler angles (ZXZ convention)"""
        alpha, beta, gamma = angles
        
        # Rotation around Z
        Rz1 = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha),  np.cos(alpha), 0],
            [            0,              0, 1]
        ])
        
        # Rotation around X
        Rx = np.array([
            [1,            0,             0],
            [0, np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta),  np.cos(beta)]
        ])
        
        # Rotation around Z again
        Rz2 = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [            0,              0, 1]
        ])
        
        return Rz2 @ Rx @ Rz1
        
        
    def _rotation_z_to_x(self):
        return np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [-1, 0, 0]
        ])
        
    def _rotation_z_to_y(self):
        return np.array([
            [ 1, 0, 0],
            [ 0, 0, 1],
            [ 0,-1, 0]
        ])
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
