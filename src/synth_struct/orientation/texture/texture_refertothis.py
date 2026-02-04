# synth_struct/src/orientation/texture.py

import numpy as np

"""
Texture generation for microstructure
Orientations are stored as Numpy arrays of shape (num_grains, 3)
"""

class Texture:
    @staticmethod
    def random_orientations(num_grains, seed=None):
        """
        Generate random crystallographic orientations (Euler angles)
        
        Args:
        - num_grains: int - Number of grains to generate
        - seed: int or None - Random seed for reproducibility
        
        Returns:
        - np.ndarray of shape (num_grains, 3) - Euler angles [phi1, Phi, phi2] in radians
        
        Note:
            Returns array indexed from 0 to num_grains - 1.
            Grain ID mapping: grain_id = array_index + 1
        """
        if seed:
            np.random.seed(seed)
            
        # Euler angles: phi1, Phi, phi2
        # Ranges: phi1 [0, 2π], Phi [0, π], phi2 [0, 2π]
        oorientations = np.zeros((num_grains, 3))
        
        orientations[:, 0] = np.random.uniform(0, 2*np.pi, num_grains)
        orientations[:, 1] = np.arccos(np.random.uniform(-1, 1, num_grains))
        orientations[:, 2] = np.random.uniform(0, 2*np.pi, num_grains))
            
        return orientations
        
        
    @staticmethod
    def apply_texture_to_region(orientations, region_grain_ids=None, texture_type='cube', 
                                degspread=15, custom_euler=None, seed=None):
        """
        Apply specific texture to grains in a region.
        
        Args:
        - orientations: np.ndarray of shape (num_grains, 3) - Existing Euler angles
        - region_grain_ids: array-like or None - Grain IDs (1-indexed) to apply texture to.
                            If None, applies to all grains.
        - texture_type: str - Type of texture component to apply
        - degspread: float - Spread around ideal orientation in degrees
        - custom_euler: list or np.ndarray - Custom Euler angles [phi1, Phi, phi2] in degrees
             (only used when texture_type='custom')
        - seed: int or None - Random seed for reproducibility
            
        Returns:
        - np.ndarray of shape (num_grains, 3) - Updated orientations
            
        Texture components (Miller indices {hkl}<uvw>):
        - cube: {001}<100> - Recrystallization texture
        - goss: {011}<100> - Recrystallization texture (strong in Fe-Si)
        - brass: {011}<211> - Rolling texture (FCC metals)
        - copper: {112}<111> - Rolling texture (FCC metals)
        - s: {123}<634> - Rolling texture (FCC metals)
        - rotated_cube: {001}<110> - Recrystallization texture
        - rotated_goss: {011}<011> - Recrystallization texture
        - p: {011}<122> - Rolling texture
        - random: Random orientations
        - basal: Basal texture (HCP)
        - prismatic: Prismatic texture (HCP)
        - custom: User-provided Euler angles (requires custom_euler)
            
        Note:
            Grain IDs are 1-indexed (grain 1 is stored at orientations[0, :])
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Make a copy to avoid modifying the input array
        orientation = orientations.copy()
        num_grains = orientations.shape[0]
        
        # If no region specified, apply to all grains
        if region_grain_ids is None:
            region_indices = np.arange(num_grains)
        else:
            # Convert grain IDs (1-indexed) to array indices (0-indexed)
            region_grain_ids = np.asarray(region_grain_ids)
            region_indices = region_grain_ids - 1
            
            # Filter out invalid indices
            valid_mask = (region_indices >= 0) & (region_indices < num_grains)
            if not np.all(valid_mask):
                print(f"Warning: Some grain IDs are out of range and will be ignored.")
                region_indices = region_indices[valid_mask]
                

        
        # Define ideal orientations for each texture type
        if texture_type == 'cube':
            # Cube texture: {001}<100>
            mean_orientation = np.array([0.0, 0.0, 0.0])
        
        elif texture_type == 'goss':
            # Goss texture: {011}<100>
            mean_orientation = np.array([0.0, np.radians(45.0), np.radians(45.0)])
        
        elif texture_type == 'brass':
            # Brass texture: {110}<112>
            mean_orientation = np.array([np.radians(35.26), np.radians(45.0), 0.0])
            
        elif texture_type == 'copper':
            # Copper texture: {112}<111>
            mean_orientation = np.array([np.radians(90.0), np.radians(35.26), np.radians(45.0)])
            
        elif texture_type == 's':
            # S texture: {123}<634>
            mean_orientation = np.array([np.radians(58.98), np.radians(36.70), np.radians(63.43)])
        
        elif texture_type == 'rotated_cube':
            # Rotated Cube texture: {001}<011>
            mean_orientation = np.array([np.radians(45.0), 0.0, 0.0])
        
        elif texture_type == 'rotated_goss':
            # Rotated Goss texture: {011}<011>
            mean_orientation = np.array([0.0, np.radians(45.0), np.radians(45.0)])
        
        elif texture_type == 'p':
            # Goss texture: {011}<122>
            mean_orientation = np.array([np.radians(70.53), np.radians(45.0), 0.0])
        
        elif texture_type == 'basal':
            # Basal texture (HCP): c-axis parallel to normal
            mean_orientation = np.array([0.0, 0.0, 0.0])
            
        elif texture_type == 'prismatic':
            # Prismatic texture (HCP): c-axis perpendicular to normal 
            mean_orientation = np.array([0.0, np.radians(90.0), 0.0])
            
        elif texture_type == 'custom':
            # Custom user-provided Euler angles
            if custom_euler is None:
                raise ValueError("custom_euler must be provided when texture_type='custom'. "
                               "Provide as [phi1, Phi, phi2] in degrees.")
            
            custom_euler = np.asarray(custom_euler)
            if custom_euler.shape != (3,):
                raise ValueError(f"custom_euler must have shape (3,) [phi1, Phi, phi2], "
                               f"got shape {custom_euler.shape}")
            
            # Convert from degrees to radians
            mean_orientation = np.radians(custom_euler)
        
        elif texture_type == 'random':
            # Random texture: Generate new random orientations for specified grains
            num_region_grains = len(region_indices)
            orientations[region_indices, 0] = np.random.uniform(0.0, 2*np.pi, num_region_grains)
            orientations[region_indices, 1] = np.arccos(np.random.uniform(-1.0, 1.0, num_region_grains))
            orientations[region_indices, 2] = np.random.uniform(0.0, 2*np.pi, num_region_grains)
            return orientations
            
        else:
            raise ValueError(f"Unknown texture type: {texture_type}. "
                             f"Available types: cube, goss, brass, copper, s, rotated_cube, "
                             f"rotated_goss, p, basal, prismatic, random, custom")
                             
        # APply texture with scatter to grains in region
        num_region_grains = len(region_indices)
        spread = np.radians(degspread)
        noise = np.random.normal(0, spread, (num_region_grains, 3))
        orientations[region_indices] = mean_orientation + noise
        
        return orientations
        
    @staticmethod
    def miller_to_euler(hkl, uvw):
        """
        Convert Miller indices to Euler angles (Bunge convention).
        Allows users to define textures using crystallographic notation.
        
        Args:
            hkl: array-like - Plane normal as [h, k, l]
            uvw: array-like - Direction as [u, v, w]
            
        Returns:
            np.ndarray of shape (3,) - Euler angles [phi1, Phi, phi2] in degrees
            
        Example:
            # Goss texture {011}<100>
            euler = Texture.miller_to_euler([0, 1, 1], [1, 0, 0])
            orientations = Texture.apply_texture_to_region(
                orientations,
                texture_type='custom',
                custom_euler=euler
            )
        """
        # Normalize vectors
        hkl = np.array(hkl, dtype=float)
        uvw = np.array(uvw, dtype=float)
        
        hkl = hkl / np.linalg.norm(hkl)
        uvw = uvw / np.linalg.norm(uvw)
        
        # hkl should be parallel to sample normal (ND)
        # uvw should be parallel to rolling direction (RD)
        
        # Create rotation matrix from crystal to sample frame
        # RD = uvw, TD = hkl x uvw, ND = hkl
        RD = uvw
        ND = hkl
        TD = np.cross(ND, RD)
        TD = TD / np.linalg.norm(TD)
        
        # Rotation matrix (crystal to sample)
        g = np.column_stack([RD, TD, ND])
        
        # Convert rotation matrix to Euler angles (Bunge ZXZ)
        # Phi (second angle)
        Phi = np.arccos(g[2, 2])
        
        # Handle gimbal lock cases
        if np.abs(Phi) < 1e-10:  # Phi ~ 0
            phi1 = np.arctan2(g[0, 1], g[0, 0])
            phi2 = 0.0
        elif np.abs(Phi - np.pi) < 1e-10:  # Phi ~ 180
            phi1 = np.arctan2(-g[0, 1], g[0, 0])
            phi2 = 0.0
        else:
            phi1 = np.arctan2(g[2, 0], -g[2, 1])
            phi2 = np.arctan2(g[0, 2], g[1, 2])
        
        # Convert to degrees and ensure positive angles
        phi1_deg = np.degrees(phi1) % 360
        Phi_deg = np.degrees(Phi)
        phi2_deg = np.degrees(phi2) % 360
        
        return np.array([phi1_deg, Phi_deg, phi2_deg])
        
    @staticmethod
    def attach_to_microstructure(micro, orientations, field_name='orientations'):
        """
        Attach orientation data to a Microstructure object.
        
        Args:
        - micro: Microstructure object
        - orientations: np.ndarray of shape (num_grains, 3) - Euler angles
        - field_name: str - Name of the field to attach (default: 'orientations')
            
        Returns:
        - None (modifies microstructure in place)
            
        Example:
            micro = Microstructure(dimensions=(100, 100), resolution=1.0)
            # ... generate microstructure ...
            orientations = Texture.random_orientations(micro.num_grains)
            Texture.attach_to_microstructure(micro, orientations)
        """
        if orientations.shape[0] != micro.num_grains:
            raise ValueError(f"Number of orientations ({orientations.shape[0]}) does not match "
                             f"number of grains ({micro.num_grains})")
        
        microstructure.attach_field(field_name, orientations)
        
    @staticmethod
    def from_microstructure(micro, field_name='orientations'):
        """
        Retrieve orientation data from a Microstructure object.
        
        Args:
        - micro: Microstructure object
        - field_name: str - Name of the field to retrieve (default: 'orientations')
            
        Returns:
        - np.ndarray of shape (num_grains, 3) - Euler angles
            
        Example:
            orientations = Texture.from_microstructure(ms)
        """
        return micro.get_field(field_name)
        
    @staticmethod
    def set_grain_orientation(orientations, grain_id, euler_angles):
        """
        Set the orientation of a specific grain.
        
        Args:
        - orientations: np.ndarray of shape (num_grains, 3) - Euler angles
        - grain_id: int - Grain ID (1-indexed)
        - euler_angles: array-like of shape (3,) - Euler angles [phi1, Phi, phi2] in radians
            
        Returns:
        - np.ndarray of shape (num_grains, 3) - Updated orientations
            
        Example:
            orientations = Texture.set_grain_orientation(orientations, grain_id=5, 
                                                       euler_angles=[0, 0, 0])
        """
        orientations = orientations.copy()
        
        if grain_id < 1 or grain_id > orientations.shape[0]:
            raise ValueError(f"Grain ID {grain_id} is out of range [1, {orientations.shape[0]}]")
        
        euler_angles = np.asarray(euler_angles)
        if euler_angles.shape != (3,):
            raise ValueError(f"euler_angles must have shape (3,), got shape {euler_angles.shape}")
        
        orientations[grain_id - 1] = euler_angles
        
        return orientations
        
    @staticmethod
    def orientation_statistics(orientations, degrees=False):
        """
        Compute basic statistics on orientation distribution.
        
        Args:
        - orientations: np.ndarray of shape (num_grains, 3) - Euler angles in radians
        - degrees: bool - If True, return statistics in degrees instead of radians
            
        Returns:
        - dict with keys 'mean', 'std', 'min', 'max' for each Euler angle
            
        Example:
            stats = Texture.orientation_statistics(orientations, degrees=True)
            print(f"Mean phi1: {stats['phi1']['mean']:.2f} degrees")
        """
        if degrees:
            orientations = np.degrees(orientations)
        
        stats = {
            'phi1': {
                'mean': np.mean(orientations[:, 0]),
                'std': np.std(orientations[:, 0]),
                'min': np.min(orientations[:, 0]),
                'max': np.max(orientations[:, 0])
            },
            'Phi': {
                'mean': np.mean(orientations[:, 1]),
                'std': np.std(orientations[:, 1]),
                'min': np.min(orientations[:, 1]),
                'max': np.max(orientations[:, 1])
            },
            'phi2': {
                'mean': np.mean(orientations[:, 2]),
                'std': np.std(orientations[:, 2]),
                'min': np.min(orientations[:, 2]),
                'max': np.max(orientations[:, 2])
            }
        }
        
        return stats
        
