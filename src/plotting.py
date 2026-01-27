import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from orix.quaternion import Orientation, Symmetry
from orix.vector import Vector3d
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.plot import IPFColorKeyTSL
from orix import plot

from rotation_converter import euler_to_quat

class OrixVisualizer:
    """
    Microstructure visualization using orix
    Mostly EBSD related visualizations
    
    - IPF maps
    - Pole figures
    - ODFs
    """
    
    @staticmethod
    def create_crystal_map_from_microstructure(microstructure, crystal_structure='cubic'):
        """
        Convert microstructure to orix CrystalMap
        
        Args:
        - microstructure: Microstructure object with orientations
        - crystal_structure: 'cubic', 'hexagonal', etc.
        
        Returns: orix CrystalMap        
        """
                
        if crystal_structure.lower() in ['cubic', 'fcc', 'bcc']:
            phase = Phase(name='Cubic', point_group='m-3m')
            symmetry = phase.point_group
        elif crystal_structure.lower() in ['hexagonal', 'hcp']:
            phase = Phase(name='Hexagonal', point_group='6/mmm')
            symmetry = phase.point_group
        else:
            raise ValueError(f"Unknown crystal structure: {crystal_structure}")
            
        phase_list = PhaseList(phase)
            
        quaternions = euler_to_quat(microstructure.orientations)
        
        grain_ids_flat = microstructure.grain_ids.flatten()
        
        num_points = len(grain_ids_flat)
        quaternion_array = np.zeros((num_points, 4))
        
        # Map grain IDs to quaternions
        for i, grain_id in enumerate(grain_ids_flat):
            if grain_id > 0 and grain_id in quaternions:
                quaternion_array[i] = quaternions[grain_id]
            else:
                quaternion_array[i] = [1,0,0,0]
                
        orientations = Orientation(quaternion_array, symmetry=symmetry)
        
        if len(microstructure.dimensions) == 3:
            nx, ny, nz = microstructure.dimensions
            
            z_coords = np.repeat(np.arange(nz), ny * nx)
            y_coords = np.tile(np.repeat(np.arange(ny), nx), nz)
            x_coords = np.tile(np.arange(nx), nz * ny)
            
            crystal_map = CrystalMap(
                rotations=orientations,
                phase_id=np.zeros(num_points, dtype=int),
                x=x_coords,
                y=y_coords,
                phase_list=phase_list,
                scan_unit='px'
            )
            
            # Manually add z coordinates as a property
            crystal_map.prop['z'] = z_coords
        
        else:  # 2D
            ny, nx = microstructure.dimensions
            
            # Create flattened coordinate arrays
            y_coords = np.repeat(np.arange(ny), nx)
            x_coords = np.tile(np.arange(nx), ny)
            
            # Create CrystalMap
            crystal_map = CrystalMap(
                rotations=orientations,
                phase_id=np.zeros(num_points, dtype=int),
                x=x_coords,
                y=y_coords,
                phase_list=phase_list,
                scan_unit='px'
            )
            
        return crystal_map
            
            
    @staticmethod
    def plot_ipf_map(microstructure, direction='z', crystal_structure='cubic',
                     slice_idx=None, slice_direction='z', filename=None, 
                     show_scalebar=True, show_title=False):
                     
        """
        Create IPF map using orix
        
        Args:
        - microstructure: Microstructure object
        - direction: IPF direction ('x', 'y', or 'z')
        - crystal_structure: 'cubic' or 'hexagonal'
        - slice_idx: Slice index for 3D
        - slice_direction: Slice direction for 3D
        - filename: Save filename
        - show_colorkey: Show IPF color key
        """
        
        crystal_map = OrixVisualizer.create_crystal_map_from_microstructure(
            microstructure, crystal_structure
        )
        
        if len(microstructure.dimensions) == 3:
            if slice_idx is None:
                slice_idx = microstructure.dimensions[2] // 2
                
            if slice_direction == 'z':
                crystal_map_slice = crystal_map[crystal_map.prop['z'] == slice_idx]
            elif slice_direction == 'y':
                crystal_map_slice = crystal_map[crystal_map.y == slice_idx]
            elif slice_direction == 'x':
                crystal_map_slice = crystal_map[crystal_map.x == slice_idx]
        else:
            crystal_map_slice = crystal_map
            
        if direction.lower() == 'x':
            ipf_direction = Vector3d.xvector()
        elif direction.lower() == 'y':
            ipf_direction = Vector3d.yvector()
        else:
            ipf_direction = Vector3d.zvector()
            
        symmetry = crystal_map.phases[0].point_group
            
        ipf_key = IPFColorKeyTSL(symmetry, direction=ipf_direction)
        rgb_pixels = ipf_key.orientation2color(crystal_map_slice.rotations)
        
        if len(microstructure.dimensions) == 3:
            if slice_direction == 'z':
                shape = (microstructure.dimensions[0], microstructure.dimensions[1])
            elif slice_direction == 'y':
                shape = (microstructure.dimensions[0], microstructure.dimensions[2])
            else:  # x
                shape = (microstructure.dimensions[1], microstructure.dimensions[2])
        else:
            shape = microstructure.dimensions
            
        fig, ax = plt.subplots(figsize=(8,8))
            
        ipf_image = rgb_pixels.reshape(*shape, 3)
        ax.imshow(ipf_image)
        
        if show_title:
            ax.set_title(f'IPF-{direction.upper()} Map')
        
        ax.axis('off')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        if show_scalebar:    
            scalebar = ScaleBar(1.0,
                                units='µm',
                                location='lower right',
                                box_color='white',
                                box_alpha=0.75,
                                color='black')
            ax.add_artist(scalebar)
            
        plt.tight_layout()
        
        if filename:
            plt.savefig(f'../output/{filename}', dpi=150, bbox_inches='tight')
            print(f"Saved IPF map to ../output/{filename}")
            
        # plt.show()
        
    @staticmethod
    def plot_pole_figure(ax, miller_index, microstructure,
                         crystal_structure='cubic', show_labels=True,
                         subset=None):
        """
        Plot a pole figure
        
        Args:
        - ax: axis for pole figure to be plotted on
        - miller_index: 
        - microstructure: Microstructure object
        - crystal_structure: 'cubic' or 'hexagonal'
        - show_labels: Show labels on the 
        """
        
        from orix import plot
        from orix.vector import Miller
        
        crystal_map = OrixVisualizer.create_crystal_map_from_microstructure(
            microstructure, crystal_structure
        )
        
        symmetry = crystal_map.phases[0].point_group
            
        phase = crystal_map.phases[0]
        
        miller = Miller(uvw=miller_index, phase=phase)
        orientations = crystal_map.orientations
        print(orientations.size)
        
        if subset:
            n = orientations.size
            
            idx = np.random.choice(n, size=int(subset*n), replace=False)
            orientations_sub = orientations[idx]
            print(orientations_sub.size)
            pf = ax.scatter(orientations_sub.inv().outer(miller))
        else:        
            pf = ax.scatter(orientations.inv().outer(miller))
        
        if show_labels:
            ax.set_labels('X', 'Y', 'Z')
        
        ax.set_title(fr"$\left<{miller_index[0]} {miller_index[1]} {miller_index[2]}\right>$")
        
        return pf
    
    @staticmethod
    def plot_all_pole_figures(axes, miller_indices, microstructure, 
                              crystal_structure='cubic', show_labels=True, subset=None,
                              **plot_kwargs):
    
        from orix.vector import Miller
        
        crystal_map = OrixVisualizer.create_crystal_map_from_microstructure(
            microstructure, crystal_structure
        )
        
        if len(axes) != len(miller_indices):
            raise ValueError(f"Number of axes {len(axes)} must match number of Miller indices {len(miller_indices)}")
        
        artists = []
        
        for ax, hkl in zip(axes, miller_indices):
            artist = OrixVisualizer.plot_pole_figure(ax, hkl, microstructure, 
                crystal_structure=crystal_structure, show_labels=show_labels, subset=subset,
                **plot_kwargs)
            artists.append(artist)
            
        return artists
        
    @staticmethod
    def create_pole_figure_axes(fig, n, projection='stereographic'):
        """
        Create n stereographic subplots arranged in a single row.
        """
        axes = []
        for i in range(n):
            ax = fig.add_subplot(1, n, i+1, projection=projection)
            axes.append(ax)
            
        return axes
        
        
        
    @staticmethod
    def plot_odf(microstructure, crystal_structure='cubic', filename=None):
        """
        Plot Orientation Distribution Function (ODF)
        
        Args:
        - microstructure: Microstructure object
        - crystal_structure: 'cubic' or 'hexagonal'
        - filename: Save filename
        """
        
        crystal_map = OrixVisualizer.create_crystal_map_from_microstructure(
            microstructure, crystal_structure
        )
        
        fig = crystal_map.rotations.scatter(
            projection="axangle",
            figure_kwargs=dict(figsize=(10, 10))
        )
        
        plt.title('Orientation Distribution')
        
        if filename:
            plt.savefig(f"../output/{filename}", dpi=300, bbox_inches='tight')
            print(f"Saved ODF to ../output/{filename}")
        
        # plt.show()
        
        
        
        
        
        
        
        
        
        
        
