import sys
sys.path.insert(0, '../src')

from euler_quat_converter import euler_to_quat

import numpy as np

class Stiffness:
    @staticmethod
    def Isotropic(microstructure, E=210.0, nu=0.3, noise=None):
        """
        Give each grain the modulus and Poisson ratio.
        
        Args:
        - microstructure: A generated microstructure, does not need a texture.
        - E: Elastic modulus
        - nu: Poisson ratio
        
        Returns stiffness matrix for each grain
        """
        
        stiffness = {}
        
        lam = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2*(1 + nu))
        c = lam + 2*mu
        
        num_grains = microstructure.get_num_grains()
        
        for grain_id in range(1, num_grains + 1):
            if noise == None:
                C_isotropic = np.array([
                    [  c, lam, lam,  0,  0,  0],
                    [lam,   c, lam,  0,  0,  0],
                    [lam, lam,   c,  0,  0,  0],
                    [  0,   0,   0, mu,  0,  0],
                    [  0,   0,   0,  0, mu,  0],
                    [  0,   0,   0,  0,  0, mu]
                ])
                
            else:
                E_noise, nu_noise = E, nu
                lam_noise = E_noise * nu_noise / ((1 + nu_noise) * (1 - 2*nu_noise))
                mu_noise = E_noise / (2*(1 + nu_noise))
                c_noise = lam_noise + 2*mu_noise
                        
                C_isotropic = np.array([
                    [  c_noise, lam_noise, lam_noise,        0,        0,        0],
                    [lam_noise,   c_noise, lam_noise,        0,        0,        0],
                    [lam_noise, lam_noise,   c_noise,        0,        0,        0],
                    [        0,         0,         0, mu_noise,        0,        0],
                    [        0,         0,         0,        0, mu_noise,        0],
                    [        0,         0,         0,        0,        0, mu_noise]
                ])
        
            stiffness[grain_id] = C_isotropic
        
        return stiffness
        
    @staticmethod
    def Cubic(microstructure, C11=228.0, C12=116.5, C44=132.0, noise=None):
        """
        Give each grain the stiffness matrix for a Cubic (FCC or BCC) material with the rotations
        
        Args:
        - microstructure: A generated microstructure with a texture
        - C11: C11 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        - C12: C12 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        - C44: C44 value in GPa (default is value for Fe (BCC) from Meyers and Chawla T2.3)
        - noise: Adding in some noise to each value of the stiffness tensor (default is no noise)
        
        Returns stiffness matrix rotated to the global reference frame for each grain
        """
        # Create stiffness list
        stiffness = {}
        
        # Convert Euler angles to quaternions
        quaternions = euler_to_quat(microstructure.orientations, convention='ZXZ')
        
        # For each grain, rotate the stiffness tensor using quaternions
        for grain_id, quat in quaternions.items():
            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)
            
            if noise == None:
                C_crystal = np.array([
                    [C11, C12, C12,   0,   0,   0],
                    [C12, C11, C12,   0,   0,   0],
                    [C12, C12, C11,   0,   0,   0],
                    [  0,   0,   0, C44,   0,   0],
                    [  0,   0,   0,   0, C44,   0],
                    [  0,   0,   0,   0,   0, C44]
                ])
                
            else:
                C11_noise, C12_noise, C44_noise = C11, C12, C44
                C11_noise *= 1.0 + noise * np.random.normal()
                C12_noise *= 1.0 + noise * np.random.normal()
                C44_noise *= 1.0 + noise * np.random.normal()
            
                C_crystal = np.array([
                    [C11_noise, C12_noise, C12_noise,         0,         0,         0],
                    [C12_noise, C11_noise, C12_noise,         0,         0,         0],
                    [C12_noise, C12_noise, C11_noise,         0,         0,         0],
                    [        0,         0,         0, C44_noise,         0,         0],
                    [        0,         0,         0,         0, C44_noise,         0],
                    [        0,         0,         0,         0,         0, C44_noise]
                ])
            
            C_rotated = Stiffness._rotate_with_quat(C_crystal, quat)
            
            stiffness[grain_id] = C_rotated
                
            
        return stiffness
        
    @staticmethod
    def Hexagonal(microstructure, C11=162.4, C12=92.0, C13=69.0, C33=180.7, C44=46.7, noise=None):
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
        
        # Convert Euler angles to quaternions
        quaternions = euler_to_quat(microstructure.orientations, convention='ZXZ')
        
        # For each grain, rotate the stiffness tensor using quaternions
        for grain_id, quat in quaternions.items():
            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)
            
            if noise == None:
                C_crystal = np.array([
                    [C11, C12, C13,   0,   0,   0],
                    [C12, C11, C13,   0,   0,   0],
                    [C13, C13, C33,   0,   0,   0],
                    [  0,   0,   0, C44,   0,   0],
                    [  0,   0,   0,   0, C44,   0],
                    [  0,   0,   0,   0,   0, C66]
                ])
                
            else:
                C11_noise, C12_noise, C13_noise, C33_noise, C44_noise = C11, C12, C13, C33, C44
                
                C11_noise *= 1.0 + noise * np.random.normal()
                C12_noise *= 1.0 + noise * np.random.normal()
                C13_noise *= 1.0 + noise * np.random.normal()
                C33_noise *= 1.0 + noise * np.random.normal()
                C44_noise *= 1.0 + noise * np.random.normal()
                C66_noise = 0.5 * (C11_noise - C12_noise)
            
                C_crystal = np.array([
                    [C11_noise, C12_noise, C13_noise,         0,         0,         0],
                    [C12_noise, C11_noise, C13_noise,         0,         0,         0],
                    [C13_noise, C13_noise, C33_noise,         0,         0,         0],
                    [        0,         0,         0, C44_noise,         0,         0],
                    [        0,         0,         0,         0, C44_noise,         0],
                    [        0,         0,         0,         0,         0, C66_noise]
                ])
            
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
        
    
