import numpy as np
import sys
sys.path.insert(0, '../src')
from euler_quat_converter import euler_to_quat

class Stiffness:
    @staticmethod
    def Isotropic(microstructure, E=210.0, nu=0.3):
        """
        Give each grain the modulus and Poisson ratio.
        
        Args:
        - microstructure: A generated microstructure, does not need a texture.
        - E: Elastic modulus
        - nu: Poisson ratio
        
        Returns stiffness matrix for each grain
        """
        
        stiffness = {}
        
        num_grains = microstructure.get_num_grains()
        
        for grain_id in range(1, num_grains + 1):
            stiffness[grain_id] = np.array([E, nu])
        
        return stiffness
        
    @staticmethod
    def Cubic(microstructure, C11=228.0, C12=116.5, C44=132.0):
        """
        Give each grain the stiffness matrix for a Cubic (FCC or BCC) material with the rotations
        
        Args:
        - microstructure: A generated microstructure with a texture
        - C11: C11 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        - C12: C12 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        - C44: C44 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        
        Returns stiffness matrix rotated to the global reference frame for each grain
        """
        # Create stiffness list
        stiffness = {}
        
        # Create base cubic stiffness tensor in Voigt notation
        C_crystal = np.array([
            [C11, C12, C12,   0,   0,   0],
            [C12, C11, C12,   0,   0,   0],
            [C12, C12, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44]
        ])
        
        # Convert Euler angles to quaternions
        quaternions = euler_to_quat(microstructure.orientations, convention='ZXZ')
        
        # For each grain, rotate the stiffness tensor using quaternions
        for grain_id, quat in quaternions.items():
            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)
            
            # Rotate stiffness tensor with quaternions
            C_rotated = Stiffness._rotate_with_quat(C_crystal, quat)
            
            stiffness[grain_id] = C_rotated
            
        return stiffness
        
    @staticmethod
    def Hexagonal(microstructure, C11=162.4, C12=92.0, C13=69.0, C33=180.7, C44=46.7):
        """
        Give each grain the stiffness matrix for a hexagonal (HCP) material with the rotations
        
        Args:
        - microstructure: A generated microstructure with a texture
        - C11: C11 value in GPa (default is value for Ti from Meyers and Chawla T2.3)
        - C12: C12 value in GPa (default is value for Ti from Meyers and Chawla T2.3)
        - C33: C33 value in GPa (default is value for Ti from Meyers and Chawla T2.3)
        - C44: C44 value in GPa (default is value for Ti from Meyers and Chawla T2.3)
        - C66: C66 value in GPa (default is value for Ti from Meyers and Chawla T2.3)
        
        Returns stiffness matrix rotated to the global reference frame for each grain
        """
        stiffness = {}
        C66 = 0.5 * (C11 - C12)
        
        C_crystal = np.array([
            [C11, C12, C13,   0,   0,   0],
            [C12, C11, C13,   0,   0,   0],
            [C13, C13, C33,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C66]
        ])
        
        # Convert Euler angles to quaternions
        quaternions = euler_to_quat(microstructure.orientations, convention='ZXZ')
        
        # For each grain, rotate the stiffness tensor using quaternions
        for grain_id, quat in quaternions.items():
            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)
            
            # Rotate stiffness tensor with quaternions
            C_rotated = Stiffness._rotate_with_quat(C_crystal, quat)
            
            stiffness[grain_id] = C_rotated
        
        return stiffness
        
    @staticmethod
    def _rotate_with_quat(C, q):
        """
        Rotate stiffness tensor using a quaternion
        
        Args:
        - C: 6x6 stiffness matrix in Voigt notation
        - q: normalized quaternion [w, x, y, z]
        
        Returns a rotated 6x6 stiffness matrix
        """
        
        w, x, y, z = q
        
        # Build rotation matrix from quaternion elements
        # R_ij terms needed for rotation
        R = np.array([
            [1 - 2*(y**2 + z**2),       2*(x*y - w*z),       2*(x*z + w*y)],
            [      2*(x*y + w*z), 1 - 2*(x**2 + z**2),       2*(y*z - w*x)],
            [      2*(x*z - w*y),       2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Convert stiffness to 4th-order tensor
        C_tensor = Stiffness._voigt_to_tensor(C)
        
        # Rotate using Einstein summation with the rotation matrix from the quaternion
        C_rotated_tensor = np.einsum('im,jn,kp,lq,mnpq->ijkl', R, R, R, R, C_tensor)
        
        # Convert back to voigt notation
        C_rotated = Stiffness._tensor_to_voigt(C_rotated_tensor)
        
        return C_rotated
        
    @staticmethod
    def _voigt_to_tensor(C):
        """Convert 6x6 Voigt notation to 3x3x3x3 tensor"""
        # Voigt notation mapping
        voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
        
        C_tensor = np.zeros((3, 3, 3, 3))
        
        for i in range(6):
            for j in range(6):
                m, n = voigt_map[i]
                p, q = voigt_map[j]
                C_tensor[m, n, p, q] = C[i, j]
                C_tensor[n, m, p, q] = C[i, j]
                C_tensor[m, n, q, p] = C[i, j]
                C_tensor[n, m, q, p] = C[i, j]
        
        return C_tensor
        
    @staticmethod
    def _tensor_to_voigt(C_tensor):
        """Convert 3x3x3x3 tensor to 6x6 Voigt notation"""
        voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
        
        C = np.zeros((6, 6))
        
        for i in range(6):
            for j in range(6):
                m, n = voigt_map[i]
                p, q = voigt_map[j]
                C[i, j] = C_tensor[m, n, p, q]
        
        return C
        
    
