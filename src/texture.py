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
    def apply_texture_to_region(orientations, region_grain_ids, texture_type='cube', degspread=15):
        """
        Apply specific texture to grains in a region
        """
        
        # Placeholder - expand with more texture models
        if texture_type == 'cube':
            # Cube texture: orientations near {001}<100>
            mean_orientation = np.array([0, 0, 0])
            spread = np.radians(degspread) # 15 degree
            
            for grain_id in region_grain_ids:
                # Add noise around ideal orientation
                orientations[grain_id] = mean_orientation + np.random.normal(0, spread, 3)
                
        return orientations
        
        
