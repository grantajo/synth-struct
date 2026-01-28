# synth_struct/src/generators/gen_base.py

import numpy as np

class MicrostructureGenerator:
    """
    Base class for microstructure generators.
    
    All generator classes should inherit from this and implement:
        generate(micro: Microstructure) -> None
        
    Provides shared utilities like allocating per-grain arrays
    """
    
    def generate(self, micro):
        """
        Generate grains in the given Microstructure
        Calse _generate_internal() then allocates arrays.
        """
        self._generate_internal(micro)
        self._allocate_grain_arrays(micro)
        
        
    def _generate_internal(self, micro):
        """Internal generation logic. Must be implemented by subclasses."""
        
        raise NotImplementedError("Subclasses must implement the generate() method.")
        
    def _allocate_grain_arrays(self, micro):
        """
        Allocate arrays for per-grain properties in the Microstructure.
        
        Creates:
            micro.orientations: np.ndarray of shape (num_grains+1, 3) - Euler angles (radians)
            micro.stiffness: np.ndarray of shape (num_grains+1, 6, 6) - Stiffness tensors (GPa)
            micro.phase: np.ndarray of shape (num_grains+1,) - integer phase IDs
            
        Note: Index 0 is reserved for background; graind Ids 1..num_grains map to indices 1..num_grians
        """
        
        n = micro.num_grains+1 # +1 to include background (ID=0)
        micro.orientations = np.zeros((n, 3), dtype=np.float64)
        micro.stiffness = np.zeros((n, 6, 6), dtype=np.float32)
        micro.phase = np.zeros(n, dtype=np.int8)
        
    def get_seed_coordinates(self):
        """
        Return the coordinates of the Voronoi seed points
        """
        return self.seeds
    
