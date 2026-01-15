import numpy as np
import math

def euler_to_quat(orientations, convention='ZYX')
	"""
	Convert Euler angles to quaternion
	
	Args:
	- orientations: List of orientations [phi1 Phi phi2] in radians
	- convention: rotation order (default ZYX)
	
	Returns list of quaternions [w, x, y, z]
	"""
	
	num_grains = len(orientations)
	
	if convention == 'ZYX':
		
	    for grain_id in range(1, num_grains+1):
	        e1, e2, e3 = orientations[grain_id]
	        
	        # Half angle calculations
	        c1 = np.cos(e1 / 2)
	        c2 = np.cos(e2 / 2)
	        c3 = np.cos(e3 / 2)
	        
	        s1 = np.sin(e1 / 2)
	        s2 = np.sin(e2 / 2)
	        s3 = np.sin(e3 / 2)
	        
	        # ZYX convention quaternion
	        w = c1 * c2 * c3 + s1 * s2 * s3
	        x = s1 * c2 * c3 - c1 * s2 * s3
	        y = c1 * s2 * c3 + s1 * c2 * s3
	        z = c1 * c2 * s3 - s1 * s2 * c3
	        
	        quat[grain_id] = np.array([w, x, y, z])
	        
    return quat
    
def quat_to_euler(orientations, convention='ZYX')
    """
    Convert list of quaternions to list of Euler angles
    
    Args: 
    - orientations: list of orientations [w, x, y, z]
    - 
    
    Returns list of Euler angles [phi1, Phi, phi2] in radians
    """
    
    num_grains = len(orientations)
    
    if convention == 'ZYX':
        for grain_id in range(1, num_grains+1):
            w, x, y, z = orientations[grain_id]
            
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x**2 + y**2)
            e1 = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (w * y - z * x)
            e2 = np.arcsin(sinp)
            
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y**2 + z**2)
            e3 = np.arctan2(siny_cosp, cosy_cosp)
            
            euler_angle[grain_id] = np.array([e1, e2, e3])
            
    return euler_angle
            
