import numpy as np

def normalize_angle(angle):
    """ Normalize angle to be within [0, 2π) """
    return angle % (2 * np.pi)

def euler_to_quat(orientations, convention='ZXZ'):
    quat = {}
    if convention == 'ZXZ':
        for grain_id, angles in orientations.items():
            phi1, Phi, phi2 = angles
            
            # Half angle calculations
            c1 = np.cos(phi1 / 2)
            c2 = np.cos(Phi / 2)
            c3 = np.cos(phi2 / 2)
            s1 = np.sin(phi1 / 2)
            s2 = np.sin(Phi / 2)
            s3 = np.sin(phi2 / 2)
            
            w = c1 * c2 * c3 - s1 * c2 * s3
            x = c1 * s2 * c3 + s1 * s2 * s3
            y = -c1 * s2 * s3 + s1 * s2 * c3
            z = c1 * c2 * s3 + s1 * c2 * c3
            
            quat[grain_id] = np.array([w, x, y, z])

    return quat

def quat_to_euler(orientations, convention='ZXZ'):
    euler_angles = {}
    
    for grain_id, quat in orientations.items():
        w, x, y, z = quat
        
        # Normalize quaternion
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        
        # ZXZ Euler angles from quaternion
        phi1 = np.arctan2(x*z + w*y, w*x - y*z)
        Phi = np.arccos(2*(w**2 + z**2) - 1)
        phi2 = np.arctan2(x*z - w*y, w*x + y*z)
        
        euler_angles[grain_id] = np.array([normalize_angle(phi1), 
                                         normalize_angle(Phi), 
                                         normalize_angle(phi2)])
        
    return euler_angles
