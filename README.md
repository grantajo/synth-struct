This is a microstructure generator that outputs an HDF5 file with the grain structure and the orientations.

Things to be added:
- Add grain structures with grain size better distribution
- Fix HDF5 writer so that a software can read it
- Fix IPF Key visualization so that it is standard to other software
- Add in ability to have porosity
- Make sure there is the ability to have a second phase with a different stiffness tensor
- Add in ability to have precipitates
 
Long term additions:
- Add in ability to have orientation gradients within grains, subgrains and GNDs
- Add in ability to create a mesh with the microstructure
 
Dependencies
- Numpy
- Scipy
- h5py
