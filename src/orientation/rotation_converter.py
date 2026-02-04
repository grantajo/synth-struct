# synth_struct/src/orientation/rotation_converter.py
"""
To import from a src file, use
from ..orientations.rotation_converter import (functions here)
"""

import numpy as np

"""
This file holds a set of conversion functions between Euler angles, 
quaternions, and orientation matrices

These are based on Rowenhorst et al. 2015. Modelling Simul. Materi. Sci. Eng.
doi: 10.1088/0965-0393/23/8/083501

Bunge Euler convention is always ZXZ
"""

# eu2quat, eu2om, quat2eu, om2eu, quat2om, om2quat

def normalize_angle(angles):
    """ Normalize Euelr angles to be within [0, 2π) """
    return angles % (2 * np.pi)

def euler_to_quat(euler_angles):
    """
    Convert Bunge Euler angles to quaternions.
    
    Args:
    - euler_angles: np.ndarray of shape (3,) or (N, 3) - [phi1, Phi, phi2]
    
    Returns:
    - quats: np.ndarray of shape (4,) or (N, 4) - [q0, q1, q2, q3]
    """
    
    single_input = euler_angles.ndim == 1
    if single_input:
        euler_angles = euler_angles[np.newaxis, :]
        
    N = len(euler_angles)
    quat = np.zeros((N, 4))
        
    phi1 = euler_angles[:, 0]
    Phi =  euler_angles[:, 1]
    phi2 = euler_angles[:, 2]
    
    # Half angle calculations
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c = np.cos(Phi / 2)
    s = np.sin(Phi / 2)
    
    quat = np.column_stack([
         c*np.cos(sigma), 
        -s*np.cos(delta), 
        -s*np.sin(delta), 
        -c*np.sin(sigma)
    ])
    
    # Ensure q0 >= 0
    mask = quat[:, 0] < 0
    quat[mask] = -quat[mask]
        
    return quat[0] if single_input else quat

def euler_to_rotation_matrix(euler_angles):
    """
    Convert Bunge Euler angles to rotation matrix.
    
    Args:
    - euler_angles: np.ndarray of shape (3,) or (N, 3) - [phi1, Phi, phi2]
    
    Returns:
    - rotation_matrices: np.ndarray of shape (3, 3) or (N, 3, 3)
    """
    single_input = euler_angles.ndim == 1
    
    if single_input:
        euler_angles = euler_angles[np.newaxis, :]
        
    phi1 = euler_angles[:, 0]
    Phi =  euler_angles[:, 1]
    phi2 = euler_angles[:, 2]
    
    N = len(euler_angles)
    
    R = np.zeros((N, 3, 3))
    
    # First row
    R[:, 0, 0] = np.cos(phi1)*np.cos(phi2) - np.sin(phi1)*np.sin(phi2)*np.cos(Phi)
    R[:, 0, 1] = np.sin(phi1)*np.cos(phi2) + np.cos(phi1)*np.sin(phi2)*np.cos(Phi)
    R[:, 0, 2] = np.sin(phi2)*np.sin(Phi)
    
    # Second row
    R[:, 1, 0] = -np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(Phi)
    R[:, 1, 1] = -np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(Phi)
    R[:, 1, 2] =  np.cos(phi2)*np.sin(Phi)
    
    # Third row
    R[:, 2, 0] =  np.sin(phi1)*np.sin(Phi)
    R[:, 2, 1] = -np.cos(phi1)*np.sin(Phi)
    R[:, 2, 2] =  np.cos(Phi)
    
    return R[0] if single_input else R
    
    
def quat_to_euler(quats):
    """
    Convert quaternions to Bunge Euler angles.
    
    Args:
    - quats: np.ndarray of shape (4,) or (N, 4) - [q0, q1, q2, q3]
    
    Returns:
    - euler_angles: np.ndarray of shape (3,) or (N, 3) - [phi1, Phi, phi2]
    """
    
    single_input = quats.ndim == 1
    if single_input:
        quats = quats[np.newaxis, :]
        
    N = len(quats)
        
    # Normalize quaternions
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norm
    
    
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]
    
    q03 = q0**2 + q3**2
    q12 = q1**2 + q2**2
    chi = np.sqrt(q03*q12)
    
    euler_angles = np.zeros((N, 3))    
        
    # Case 1: chi == 0 and q12 == 0:
    mask1 = (chi == 0) & (q12 == 0)
    euler_angles[mask1, 0] = np.arctan2(-2*q0[mask1]*q3[mask1], q0[mask1]**2 - q3[mask1]**2)
    euler_angles[mask1, 1] = 0.
    euler_angles[mask1, 2] = 0.
    
    # Case 2: chi == 0 and q03 == 0:
    mask2 = (chi == 0) & (q03 == 0)
    euler_angles[mask2, 0] = np.arctan2(-2*q0[mask2]*q3[mask2], q0[mask2]**2 - q3[mask2]**2)
    euler_angles[mask2, 1] = 0.
    euler_angles[mask2, 2] = 0.
    
    # Case 3: General case
    mask3 = ~mask1 & ~mask2
    euler_angles[mask3, 0] = np.arctan2(( q1[mask3]*q3[mask3] - q0[mask3]*q2[mask3]) / chi[mask3], 
                                        (-q0[mask3]*q1[mask3] - q2[mask3]*q3[mask3]) / chi[mask3])
    euler_angles[mask3, 1] = np.arctan2(2*chi[mask3], q03[mask3] - q12[mask3])
    euler_angles[mask3, 2] = np.arctan2((q0[mask3]*q2[mask3] + q1[mask3]*q3[mask3]) / chi[mask3], 
                                        (q2[mask3]*q3[mask3] - q0[mask3]*q1[mask3]) / chi[mask3])
    
    # Normalize Euler angles
    euler_angles = normalize_angle(euler_angles)
    
    return euler_angles[0] if single_input else euler_angles
    
    
def quat_to_rotation_matrix(quats):
    """
    Convert quaternion to rotation matrix
    
    Args:
    - quat: np.ndarray of shape (4,) or (N, 4) - [q0, q1, q2, q3]
        
    Returns:
    - rotation_matrix: np.ndarray of shape (3, 3) or (N, 3, 3)
    """ 
    
    single_input = quats.ndim == 1
    if single_input:
        quats = quats[np.newaxis, :]
    
    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]

    qbar = q0**2 - (q1**2 + q2**2 + q3**2)
    
    N = len(quats)
    R = np.zeros((N, 3, 3))
    
    # First row
    R[:, 0, 0] = qbar + 2*q1**2
    R[:, 0, 1] = 2*(q1*q2 - q0*q3)
    R[:, 0, 2] = 2*(q1*q3 + q0*q2)
    
    # Second row
    R[:, 1, 0] = 2*(q1*q2 + q0*q3)
    R[:, 1, 1] = qbar + 2*q2**2
    R[:, 1, 2] = 2*(q2*q3 - q0*q1)
    
    # Third row
    R[:, 2, 0] = 2*(q1*q3 - q0*q2)
    R[:, 2, 1] = 2*(q2*q3 + q0*q1)
    R[:, 2, 2] = qbar + 2*q3**2
        
        
    return R[0] if single_input else R
    

def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles
    
    Args:
    - R: np.ndarray of shape (3, 3) or (N, 3, 3)
    
    Returns:
    - euler_angles: np.ndarray of shape (3,) or (N, 3) - [phi1, Phi, phi2]
    """
    single_input = R.ndim == 2
    
    if single_input:
        R = R[np.newaxis, :, :]
        
    N = len(R)
    euler_angles = np.zeros((N, 3))
    
    # Case 1: R[2,2] == ±1 (gimbal lock)
    mask1 = np.abs(R[:, 2, 2]) == 1
    euler_angles[mask1, 0] = np.arctan2(R[mask1,0,1], R[mask1,0,0])
    euler_angles[mask1, 1] = (np.pi/2)*(1 - R[mask1,2,2])
    euler_angles[mask1, 2] = 0.
    
    # Case 2: General case
    mask2 = ~mask1
    xi = 1 / np.sqrt(1 - R[mask2,2,2]**2)
    euler_angles[mask2, 0] = np.arctan2(R[mask2,2,0]*xi, -R[mask2,2,1]*xi)
    euler_angles[mask2, 1] = np.arccos(R[mask2,2,2])
    euler_angles[mask2, 2] = np.arctan2(R[mask2,0,2]*xi, R[mask2,1,2]*xi)
    
    euler_angles = normalize_angle(euler_angles)
                                           
    return euler_angles[0] if single_input else euler_angles

def rotation_matrix_to_quat(R):
    """
    Convert rotation matrix to quaternions
    
    Args:
    - R: np.ndarray of shape (3, 3) or (N, 3, 3)
    
    Returns:
    - quats: np.ndarray of shape (4,) or (N, 4) - [q0, q1, q2, q3]
    """
    single_input = R.ndim == 2
    if single_input:
        R = R[np.newaxis, :, :]
        
    N = len(R)
    quats = np.zeros((N, 4))
    
    # Convert R to quaternions
    quats[:,0] = 0.5 * np.sqrt(1 + R[:,0,0] + R[:,1,1] + R[:,2,2])
    quats[:,1] = 0.5 * np.sqrt(1 + R[:,0,0] - R[:,1,1] - R[:,2,2])
    quats[:,2] = 0.5 * np.sqrt(1 - R[:,0,0] + R[:,1,1] - R[:,2,2])
    quats[:,3] = 0.5 * np.sqrt(1 - R[:,0,0] - R[:,1,1] + R[:,2,2])
    
    # Sign corrections
    quats[R[:,2,1] < R[:,1,2], 1] *= -1
    quats[R[:,0,2] < R[:,2,0], 2] *= -1
    quats[R[:,1,0] < R[:,0,1], 2] *= -1
            
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norm
    
    return quats[0] if single_input else quats
    
"""
Additional utility functions for microstructure generation
"""

def create_rotation_matrix_2d(angle):
    """
    Create a 2D rotation matrix.
    
    Args:
    - angle: float or np.ndarray - Rotation angle(s) in radians
    
    Returns:
    - R: np.ndarray of shape (2, 2) or (N, 2, 2) - Rotation matrix
    """
    
    single_input = np.isscalar(angle)
    if single_input:
        angle = np.array([angle])
        
    c, s = np.cos(angle), np.sin(angle)
    N = len(angle)
    R = np.zeros((N, 2, 2))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    
    return R[0] if single_input else R
    
def rotation_z_to_x():
    """
    Rotation matrix that aligns z-axis to x-axis.
    Created for weighted microstructure generation (ellipsoidal, columnar, etc.)
    Maps: z -> x, x -> -z, y -> y
    
    Returns:
    - np.ndarray of shape (3, 3)
    """
    return np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0]
    ])
    
def rotation_z_to_y():
    """
    Rotation matrix that aligns z-axis to x-axis.
    Created for weighted microstructure generation (ellipsoidal, columnar, etc.)
    Maps: z -> y, y -> -z, x -> x
    
    Returns:
    - np.ndarray of shape (3, 3)
    """
    return np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0,-1, 0]
    ])
    

