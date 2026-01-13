import h5py
import numpy as np

def write_struct_hdf5(microstructure, filename):
    """
    Write microstructure to HDF5 file
    """
    
    with h5py.File(filename, 'w') as f:
        # Create groups
        geometry = f.create_group('Geometry')
        orientations = f.create_group('Orientations')
        
        # Write grain IDs
        geometry.create_dataset('GrainIDs', data=microstructure.grain_ids)
        
        # Write metadata
        geometry.attrs['dimensions'] = microstructure.dimensions
        geometry.attrs['resolution'] = microstructure.resolution
        
        # Write orientations
        num_grains = microstructure.get_num_grains()
        euler_angles = np.zeros((num_grains, 3))
        
        for grain_id, angles in microstructure.orientations.items():
            if grain_id > 0:
                euler_angles[grain_id - 1] = angles
        
        orientations.create_dataset('EulerAngles', data=euler_angles)
        orientations.attrs['convention'] = 'Bunge (ZXZ)'
        orientations.attrs['units'] = 'radians'
