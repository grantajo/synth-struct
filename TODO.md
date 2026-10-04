
# Todo Lists

## Just finished

## Currently working on

- Generate better seeds for grain generators
  - Generate different seeds for the ellipsoidal and equiaxed grains
  - Generate seeds only in plane for columnar
  - Weighted laguerre tesselation for equiaxed to get better size distribution

## Short-term todo

- New lath generator (CA see generators/lath_updated.py)

## Things to be added

- Test CPFE abilities with some example microstructures
- Add in second phase generation
  - Add in ability to have precipitates
  - Add in porosity
- Maybe make a power user import subspace such as rotation conversion or symmetry operations?

## Long term additions

- See what adding in ODF textures would be like and what would be gained from it
- Continue orix integration
  - Add in ability to have the colorkey shown in an IPF map
  - Add in ability to plot a single given colorkey with ipfcolorkeys.py
- Add in GPU CUDA/ROCm implementations for anisotropic voronoi assignment and new lath generator when done
  - Won't be able to use Eigen implementation
- Add in ability to have orientation gradients within grains, subgrains and GNDs
