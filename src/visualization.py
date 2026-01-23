import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import rotation_converter

class IPFVisualizer:
    """
    Class for creating inverse pole figure (IPF) maps and pole figures
    """
    
    @staticmethod
    def euler_to_ipf_color(euler_angles, direction='z', crystal_structure='cubic'):
        """
        Convert Euler angles to IPF color for a given sample direction
        
        Args:
        - euler_angles: [phi1, Phi, phi2] in radians
        - direction: Sample direction ('x', 'y', or 'z')
        - crystal_structure: 'cubic' or 'hexagonal'
        
        Returns: RGB color [r, g, b]
        """
        
        # Get rotation matrix from Euler angles
        R = euler_quat_r_converter.euler_to_rotation_matrix(euler_angles)
        
        if direction == 'x':
            sample_dir = np.array([1, 0, 0])
        elif direction == 'y':
            sample_dir = np.array([0, 1, 0])
        else:
            sample_dir = np.array([0, 0, 1])
            
        crystal_dir = R.T @ sample_dir
        
        if crystal_structure == 'cubic':
            color = IPFVisualizer._cubic_ipf_color(crystal_dir)
        elif crystal_structure == 'hexagonal':
            color = IPFVisualizer._hexagonal_ipf_color(crystal_dir)
        else:
            raise ValueError(f"Unknown crystal structure: {crystal_structure}")
            
        return color
        
    @staticmethod
    def _cubic_ipf_color(direction):
        """
        Get IPF color for cubic crystal system
        Standard triangle: [001] - [101] - [111]
        """
        # Normalize and take absolute values (cubic symmetry)
        d = np.abs(direction)
        d = d / np.linalg.norm(d)
        
        # Sort components: d[0] >= d[1] >= d[2]
        d = np.sort(d)[::-1]
        
        # Map to fundamental zone
        # Vertices: [001] = (0,0,1), [101] = (1,0,1)/sqrt(2), [111] = (1,1,1)/sqrt(3)
        
        # Barycentric coordinates in standard triangle
        # [001] corner: d[2] is dominant
        # [101] corner: d[0] and d[2] similar, d[1] small
        # [111] corner: all components similar
        
        # Normalize to [111] direction
        max_val = d[0]
        if max_val < 1e-10:
            return np.array([0,0,0])
            
        d_norm = d / max_val
        
        # Calculate position in fundamental zone
        # Red increases toward [001]
        # Green increases toward [101]
        # Blue increases toward [111]
        
        # Simple mapping
        red = 1.0 - d_norm[1]
        green = d_norm[1] - d_norm[2]
        blue = d_norm[2]
        
        # Normalize to [0, 1]
        total = red+green+blue
        if total < 1e-10:
            return np.array([0,0,0])
            
        color = np.array([red, green, blue]) / total
        
        return color
        
    @staticmethod
    def _hexagonal_ipf_color(direction):
        """
        Get IPF color for hexagonal crystal system
        Standard triangle: [0001] - [2 -1 -1 0] - [1 0 -1 0]
        """
        
        d = np.abs(direction)
        d = d / np.linalg.norm(d)
        
        # For hexagonal: c-axis is [0,0,1]
        # Red for basal (c-axis aligned with direction)
        # Blue for prismatic (c-axis perpendicular)
        
        c_component = d[2]
        ab_component = np.sqrt(d[0]**2 + d[1]**2)
        
        red = c_component
        blue = ab_component
        green = 0.5 * (red + blue)
        
        total = red+green+blue
        if total < 1e-10:
            return np.array([0,0,0])
        
        return np.array([red, green, blue]) / total
        
    @staticmethod
    def create_ipf_map(microstructure, direction='z', crystal_structure='cubic',
                       slice_idx=None, slice_direction='z'):
        """
        Create an IPF map for a microstructure
        
        Args:
        - microstructure: Microstructure object with orientations
        - direction: IPF direction in sample frame ('x', 'y', or 'z')
        - crystal_structure: 'cubic' or 'hexagonal'
        - slice_idx: Slice index, default: middle slice for 3D (None for 2D)
        - slice_direction: Direction of slice('x', 'y', or 'z')
        
        Returns: RGB image array [r, g, b]
        """
        
        if len(microstructure.dimensions) == 3:
            if slice_idx is None:
                slice_idx = microstructure.dimensions[0] // 2
            
            if slice_direction == 'z':
                grain_slice = microstructure.grain_ids[:, :, slice_idx]
            if slice_direction == 'y':
                grain_slice = microstructure.grain_ids[:, slice_idx, :]
            if slice_direction == 'x':
                grain_slice = microstructure.grain_ids[slice_idx, :, :]
                
        else:
            grain_slice = microstructure.grain_ids
            
        # Create RGB image
        ipf_map = np.zeros((*grain_slice.shape, 3))
        
        # Color each grain
        for grain_id, euler in microstructure.orientations.items():
            mask = grain_slice == grain_id
            color = IPFVisualizer.euler_to_ipf_color(euler, direction, crystal_structure)
            ipf_map[mask] = color
            
        return ipf_map
        
    @staticmethod
    def plot_ipf_map(microstructure, direction='z', crystal_structure='cubic',
                     slice_idx=None, slice_direction='z', filename=None,
                     show_legend=True):
        """
        Plot an IPF map with legend
        
        Args:
        - microstructure: Microstructure object
        - direction: IPF direction ('x', 'y', or 'z')
        - crystal_structure: 'cubic' or 'hexagonal'
        - slice_idx: Slice index for 3D microstructures
        - slice_direction: Slice direction
        - filename: Save to file if provided
        - show_legend: Shoe IPF color key
        """
        
        ipf_map = IPFVisualizer.create_ipf_map(
            microstructure, direction, crystal_structure, 
            slice_idx, slice_direction
        )
        
        # Create figure
        if show_legend:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            
        # Plot IPF map
        ax1.imshow(ipf_map, interpolation='nearest')
        ax1.set_title(f"IPF-{direction.upper()} Map ({crystal_structure.capitalize()})")
        
        if show_legend:
            IPFVisualizer._plot_ipf_legend(ax2, crystal_structure)
            
        plt.tight_layout()
        
        if filename:
            plt.savefig('../output/'+filename, dpi=150, bbox_inches='tight')
            print(f"Saved IPF map to {'../output/'+filename}")
            
        #plt.show()
        
    @staticmethod
    def _plot_ipf_legend(ax, crystal_structure):
        """Plot IPF color key/legend"""
        
        if crystal_structure == 'cubic':
            # Create grid of directions in standard trinagle
            n_points = 250
            colors = np.zeros((n_points, n_points, 3))
            
            for i in range(n_points):
                for j in range(n_points):
                    # Barycentric coordinates
                    u = i / n_points
                    v = j / n_points
                    
                    if u+v <= 1.0:
                        # Map to crystal directions
                        d001 = np.array([0, 0, 1])
                        d101 = np.array([1, 0, 1]) / np.sqrt(2)
                        d111 = np.array([1, 1, 1]) / np.sqrt(3)
                        
                        w = 1.0 - u - v
                        
                        direction = w*d001 + u*d101 + v*d111
                        direction = direction / np.linalg.norm(direction)
                        
                        colors[i, j] = IPFVisualizer._cubic_ipf_color(direction)
                        
            ax.imshow(colors, origin='lower', interpolation='bilinear')
            ax.set_title('IPF Color Key (Cubic)')
            
            ax.text(0, 0, '[001]', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(n_points-1, 0, '[101]', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0, n_points-1, '[111]', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))    
        
        else:
            ax.text(0.5, 0.5, 'Hexagonal IPF Color Key (Simplified)',
                    ha='center', va='center', fontsize=14)
            ax.text(0.5, 0.3, 'Red: Basal [0001]', ha='center', color='red', fontsize=10)
            ax.text(0.5, 0.2, 'Blue: Prismatic', ha='center', color='blue', fontsize=10)
            
        ax.set_xlim(0, n_points if crystal_structure == 'cubic' else 1)
        ax.set_ylim(0, n_points if crystal_structure == 'cubic' else 1)
        ax.axis('off')
        
    @staticmethod
    def plot_multiple_ipf_maps(microstructure, directions=['x', 'y', 'z'],
                               crystal_structure='cubic', slice_idx=None,
                               slice_direction='z', filename=None):
        """
        Plot IPF maps for multiple directions
        
        Args:
        - microstructure: Microstructure object
        - directions: List of IPF directions
        - crystal_structure: 'cubic' or 'hexagonal'
        - slice_idx: Slice index
        - slice_direction: Slice direction
        - filename: Save filename
        """
        
        n_dirs = len(directions)
        fig, axes = plt.subplots(1, n_dirs, figsize=(6*n_dirs, 6))
        
        if n_dirs == 1:
            axes = [axes]
        
        for ax, direction in zip(axes, directions):
            ipf_map = IPFVisualizer.create_ipf_map(
                microstructure, direction, crystal_structure,
                slice_idx, slice_direction
            )
            
            ax.imshow(ipf_map, interpolation='nearest')
            ax.set_title(f'IPF-{direction.upper()}')
                    
        plt.tight_layout()
        
        if filename:
            plt.savefig('../output/'+filename, dpi=300, bbox_inches='tight')
            print(f"Saved IPF maps to {'../output/'+filename}")
        
        #plt.show()
        
        
        
