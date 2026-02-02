# synth_struct/src/orientation/texture/texture.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..rotation_converter import (
    euler_to_quat,
    euler_to_rotation_matrix,
    quat_to_euler,
    quat_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quat
)

import numpy as np

@dataclass
class Texture:
    """
    Data container for crystallographic textures.
    
    A Texture represents a mapping from grains (or voxels)
    to orientations, together with symmetry information.
    """
    
    orientations: np.ndarray
    representation: str
    symmetry: str
    metadata: Dict = field(default_factor=dict)
    
    def __post_init__(self):
        self._validate()
        
    def _validate(self):
        if not isinstance(self.orientations, np.ndarray):
            raise TypeError("Orientations must be a NumPy array")
            
        if self.orientations.ndim < 2:
            raise ValueError("Orienations must have shape (n, ...) where n is number of grains")
            
        if self.representation not in {"euler", "quat", "rotmat"}:
            raise ValueError(f"Unknown representation '{self.representation}'. "
                             f"Expected 'euler', 'quat', or 'rotmat'.")
                             
        if not isinstance(self.symmetry, str):
            raise TypeError(f"symmetry must be a string (e.g. 'cubic' or 'hexagonal')")
            
            
    @property
    def n_orientations(self) -> int:
        return self.orienations.shape[0]
        
    
    def to_representation(self, representation: str) -> "Texture":
        """
        Convert from one texture type to another.
        
        Args:
        - representation: {'euler', 'quat', or 'rotmat'}
        """
        
        if representation == self.representation:
            return self.copy()
            
        if self.representation == 'euler' and representation == 'quat':
            new_orientations = euler_to_quat(self.orientations)
            
        elif self.representation == 'euler' and representation == 'rotmat':
            new_orientations = euler_to_rotation_matrix(self.orientations)
            
        elif self.representation == 'quat' and representation == 'euler':
            new_orientations = quat_to_euler(self.orientations)
            
        elif self.representation == 'quat' and representation == 'rotmat':
            new_orientations = quat_to_rotation_matrix(self.orientations)
            
        elif self.representation == 'rotmat' and representation == 'euler':
            new_orientations = rotation_matrix_to_euler(self.orientations)
            
        elif self.representation == 'rotmat' and representation == 'quat':
            new_orientations = rotation_matrix_to_quat(self.orientations)
            
        
        else:
            raise ValueError(f"Conversion from '{self.representation}' to "
                             f"'{representation}' not supported."
                             
        return Texture(
            orientations=new_orientations,
            representation=representation,
            symmetry=self.symmetry,
            metadata=self.metadata.copy()
        )
        
    def copy(self) -> "Texture":
        return Texture(
            orientations=self.orientations.copy(),
            representation=self.representation,
            symmetry=self.symmetry,
            metadata=self.metadata.copy()
        )
        
    def subset(self, indices: np.ndarray) -> "Texture":
        """
        Return a texture corresponding to a subset of grains.
        """
        
        return Texture(
            orientations=self.orienations[indices],
            representation=self.representation,
            symmetry=self.symmetry,
            metadata=self.metadata.copy()
        )
            
            
            
    
