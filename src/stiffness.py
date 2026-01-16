import numpy as np

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
        
        num_grains = len(microstructure.get_num_grains())
        
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
        stiffness = {}
        
        num_grains = len(microstructure.get_num_grains())
        
        for grain_id in range(1, num_grains + 1):
            stiffness[grain_id]
            
        return stiffness
        
    @staticmethod
    def Hexagonal(microstructure, C11=162.4, C12=92.0, C33=180.7, C44=46.7, C66=69.0):
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
        
        num_grains = len(microstructure.get_num_grains())
        for grain_id in range(1, num_grains + 1):
            stiffness[grain_id]
        
        return stiffness
