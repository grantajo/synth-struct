import numpy as np

"""
This file holds a set of conversion functions between Euler angles, 
quaternions, and orientation matrices

These are based on Rowenhorst et al. 2015. Modelling Simul. Materi. Sci. Eng.
doi: 10.1088/0965-0393/23/8/083501

Bunge Euler convention is always ZXZ
"""
# eu2quat, eu2om, quat2eu, om2eu, quat2om, om2quat
def normalize_angle(angle):
    """ Normalize angle to be within [0, 2π) """
    return angle % (2 * np.pi)

def euler_to_quat(orientations):
    quat = {}

    for grain_id, angles in orientations.items():
        phi1, Phi, phi2 = angles
        
        # Half angle calculations
        sigma = 0.5 * (phi1 + phi2)
        delta = 0.5 * (phi1 - phi2)
        c = np.cos(Phi / 2)
        s = np.sin(Phi / 2)
        
        quat[grain_id] = np.array([
            c*np.cos(sigma), -s*np.cos(delta), -s*np.sin(delta), -c*np.sin(sigma)
        ])
        
        if quat[grain_id][0] < 0:
            quat[grain_id] = -quat[grain_id]
        
    return quat

def euler_to_rotation_matrix(orientations):
    """
    Convert Bunge Euler angles to rotation matrix
    """
    R = {}
    
    for grain_id, euler_angle in orientations.items():
        phi1, Phi, phi2 = euler_angle
        
        R_new = np.array([
            [ np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(Phi),
              np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(Phi),
              np.sin(phi2)*np.sin(Phi)],
            [-np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(Phi),
             -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(Phi),
              np.cos(phi2)*np.sin(Phi)],
            [ np.sin(phi1)*np.sin(Phi),
             -np.cos(phi1)*np.sin(Phi),
              np.cos(Phi)]
        ])
        
        R[grain_id] = R_new
    
    return R
    
def quat_to_euler(orientations):
    euler_angles = {}
    
    for grain_id, quat in orientations.items():
        q0, q1, q2, q3 = quat
        
        # Normalize quaternion
        norm = np.sqrt(q1**2 + q2**2 + q3**2 + q0**2)
        q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
        
        q03 = q0**2 + q3**2
        q12 = q1**2 + q2**2
        chi = np.sqrt(q03*q12)
        
        if chi == 0 and q12 == 0:
            phi1 = np.arctan2(-2*q0*q3, q0**2 - q3**2)
            Phi = 0.
            phi2 = 0.
        if chi == 0 and q03 == 0:
            phi1 = np.arctan2(2*q1*q2, q1**2 - q2**2)
            Phi = np.pi
            phi2 = 0.
        else:
            phi1 = np.arctan2((q1*q3 - q0*q2) / chi, (-q0*q1 - q2*q3) / chi)
            Phi = np.arctan2(2*chi, q03 - q12)
            phi2 = np.arctan2((q0*q2 + q1*q3) / chi, (q2*q3 - q0*q1) / chi)
            
        euler_angles[grain_id] = np.array([normalize_angle(phi1),
                                          normalize_angle(Phi),
                                          normalize_angle(phi2)])
        
    return euler_angles
    
def quat_to_rotation_matrix(orientations):
    """
    Convert quaternion to rotation matrix
    """ 
    R = {}
    
    for grain_id, quat in orientations.items():
        q0, q1, q2, q3 = quat
        
        qbar = q0**2 - (q1**2 + q2**2 + q3**2)
        
        R[grain_id] = np.array([
            [   qbar + 2*q1**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),    qbar + 2*q2**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1),    qbar + 2*q3**2]
        ])
    return R
    
def rotation_matrix_to_euler(orientations):
    """
    Convert rotation matrix to Euler angles
    """
    euler_angles = {}
    
    for grain_id, R in orientations.items():
        if abs(R[2,2]) == 1:
            phi1 = np.arctan2(R[0,1], R[0,0])
            Phi = (np.pi/2)*(1 - R[2,2])
            phi2 = 0.
        else:
            xi = 1 / np.sqrt(1 - R[2,2]**2)
            phi1 = np.arctan2(R[2,0]*xi, -R[2,1]*xi)
            Phi = np.arccos(R[2,2])
            phi2 = np.arctan2(R[0,2]*xi, R[1,2]*xi)
            
        euler_angles[grain_id] = np.array([normalize_angle(phi1),
                                           normalize_angle(Phi),
                                           normalize_angle(phi2)])
                                           
    return euler_angles

def rotation_matrix_to_quat(orientations):
    """
    Convert rotation matrix to quaternions
    """
    quat = {}
    
    for grain_id, R in orientations.items():     
        q0 = 0.5 * np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
        q1 = 0.5 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        q2 = 0.5 * np.sqrt(1 - R[0,0] + R[1,1] - R[2,2])
        q3 = 0.5 * np.sqrt(1 - R[0,0] - R[1,1] + R[2,2])
        
        if R[2,1] < R[1,2]:
            q1 = -q1
        if R[0,2] < R[2,0]:
            q2 = -q2
        if R[1,0] < R[0,1]:
            q3 = -q3
            
        norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
        
        quat[grain_id] = np.array([q0, q1, q2, q3])
        
    return quat
    
    
