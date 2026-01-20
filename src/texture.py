import numpy as np

class Texture:
    @staticmethod
    def random_orientations(num_grains, seed=None):
        """
        Generate random crystallographic orientations (Euler angles)
        """
        if seed:
            np.random.seed(seed)
            
        # Euler angles: phi1, Phi, phi2
        # Ranges: phi1 [0, 2π], Phi [0, π], phi2 [0, 2π]
        orientations = {}
        for grain_id in range(1, num_grains + 1):
            phi1 = np.random.uniform(0, 2*np.pi)
            Phi = np.arccos(np.random.uniform(-1, 1)) # Uniform on a sphere
            phi2 = np.random.uniform(0, 2*np.pi)
            orientations[grain_id] = np.array([phi1, Phi, phi2])
            
        return orientations
        
    @staticmethod
    def apply_texture_to_region(orientations, region_grain_ids=None, texture_type='cube', degspread=15):
        """
        Apply specific texture to grains in a region
        
        Args:
        - orientations: Dictionary of grain orientations
        - region_grain_ids: List of grain IDs in the region. If 'None', applies to all grains
        - texture_type: Type of texture component to apply
        - degspread: Spread around ideal orientation in degrees
        
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
        """
        
        # If no region specified, apply to all grains
        if region_grain_ids is None:
            region_grain_ids = list(orientations.keys())
        
        spread = np.radians(degspread)
        
        if texture_type == 'cube':
            # Cube texture: {001}<100>
            mean_orientation = np.array([0.0, 0.0, 0.0])
        
        elif texture_type == 'goss':
            # Goss texture: {011}<100>
            mean_orientation = np.array([0.0, 0.0, 0.0])
        
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
                raise ValueError("custom_euler must be provided when texture_type='custom'."
                                "Provide as [phi1, Phi, phi2] in degrees.")
                                
            if len(custom_euler) != 3:
                raise ValueError(f"custom_euler must have 3 values [phi1, Phi, phi2], got {len(custom_euler)}")
            
            # Convert from degrees to radians
            mean_orientation = np.radians(custom_euler)
            
        elif texture_type == 'random':
            # Random texture: Random orientations
            for grain_id in region_grain_ids:
                phi1 = np.random.uniform(0.0, 2*np.pi)
                Phi = np.arccos(np.random.uniform(-1.0, 1.0))
                phi2 = np.random.uniform(0.0, 2*np.pi)
                orientations[grain_id] = np.array([phi1, Phi, phi2])
            return orientations
            
        else:
            raise ValueError(f"Unknown texture type: {texture_type}. "
                            f"Available types: cube, goss, brass, copper, s, rotated_cube, rotated_goss, p, basal, prismatic, random")
        
        # Apply texture with scatter to all grains in region
        for grain_id in region_grain_ids:
            # Add noise around ideal orientation
            orientations[grain_id] = mean_orientation + np.random.normal(0, spread, 3)
        
        return orientations
        
    @staticmethod
def miller_to_euler(hkl, uvw):
    """
    Convert Miller indices to Euler angles (Bunge convention)
    Allows users to define textures using crystallographic notation
    
    Args:
    - hkl: Plane normal as [h, k, l]
    - uvw: Direction as [u, v, w]
    
    Returns: Euler angles [phi1, Phi, phi2] in degrees
    
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
    
    return [phi1_deg, Phi_deg, phi2_deg]
        
