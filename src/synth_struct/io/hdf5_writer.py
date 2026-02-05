import h5py
import numpy as np


def write_hdf5_test(microstructure, filename):
    """
    File format:
    /Microstructure
    ├── GrainData
    │   ├── GrainIDs
    │   ├── GrainOrientations
    │   └── Metadata (
    ├── SpatialInformation
    │   ├── Dimensions
    │   ├── Resolution
    │   └── SpatialUnits
    └── ProcessingInformation
        └── Date
    """
    with h5py.File(filename, "w") as f:
        # Create main groups
        grain_group = f.create_group("GrainData")
        spatial_group = f.create_group("SpatialInformation")
        processing_group = f.create_group("ProcessingInformation")

        # Store spatial information
        spatial_group.attrs["Dimensions"] = microstructure.dimensions
        spatial_group.attrs["Resolution"] = microstructure.resolution
        spatial_group.attrs["SpatialUnits"] = microstructure.units

        # Store grain data
        grain_group.create_dataset("GrainIDs", data=microstructure.grain_ids)
        grain_group.create_dataset(
            "GrainOrientations", data=microstructure.orientations
        )

        # Add metadata
        num_grains = len(microstructure.grain_ids)
        grain_group.attrs["GrainCount"] = num_grains


def write_struct_hdf5(microstructure, filename):
    """
    Write microstructure to HDF5 file
    """

    with h5py.File(filename, "w") as f:
        # Create groups
        geometry = f.create_group("Geometry")
        orientations = f.create_group("Orientations")

        # Write grain IDs
        geometry.create_dataset("GrainIDs", data=microstructure.grain_ids)

        # Write metadata
        geometry.attrs["dimensions"] = microstructure.dimensions
        geometry.attrs["resolution"] = microstructure.resolution

        # Write orientations
        num_grains = len(microstructure.orientations)
        if num_grains > 0:
            max_grain_id = max(microstructure.orientations.keys())
            euler_angles = np.zeros((max_grain_id, 3))

        for grain_id, angles in microstructure.orientations.items():
            if grain_id > 0:
                euler_angles[grain_id - 1] = angles

        orientations.create_dataset("EulerAngles", data=euler_angles)
        orientations.attrs["convention"] = "Bunge (ZXZ)"
        orientations.attrs["units"] = "radians"
        orientations.attrs["num_grains"] = num_grains


def write_d3d(microstructure, filename):
    """
    Write microstructure in DREAM.3D compatible format
    """
    with h5py.File(filename, "w") as f:
        # CRITICAL: Add file version at root level
        f.attrs.create("FileVersion", "8.0", dtype="S3")
        f.attrs.create(
            "DREAM3D Version", "6.5.0", dtype="S10"
        )  # Or your installed version

        # Pipeline (empty but required)
        pipeline = f.create_group("Pipeline")
        pipeline.attrs["Number_Filters"] = np.int32(0)

        # Create the DataContainerBundle
        f.create_group("DataContainerBundles")
        f.create_group("DataStructure")
        data_structure = f["DataStructure"]

        # Create the DataContainer
        data_container = data_structure.create_group("DataContainer")

        # AttributeMatrices metadata
        data_container.attrs["AttributeMatrixNames"] = [
            "CellData",
            "CellFeatureData",
            "CellEnsembleData",
        ]
        data_container.attrs["DataContainerType"] = "DataContainer"

        # Geometry information
        geometry = data_container.create_group("_SIMPL_GEOMETRY")
        geometry.attrs["GeometryType"] = np.uint32(0)  # ImageGeometry
        geometry.attrs["GeometryTypeName"] = "ImageGeometry"
        geometry.attrs["SpatialDimensionality"] = np.uint32(3)
        geometry.attrs["UnitDimensionality"] = np.uint32(3)

        # Dimensions (XYZ order for DREAM.3D)
        if len(microstructure.dimensions) == 3:
            dims = np.array(
                [
                    microstructure.dimensions[2],
                    microstructure.dimensions[1],
                    microstructure.dimensions[0],
                ],
                dtype=np.uint64,
            )
        else:
            # Handle 2D case
            dims = np.array(
                [microstructure.dimensions[1], microstructure.dimensions[0], 1],
                dtype=np.uint64,
            )

        geometry.create_dataset("DIMENSIONS", data=dims)

        # Origin and spacing
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        spacing = np.array(
            [
                microstructure.resolution,
                microstructure.resolution,
                microstructure.resolution,
            ],
            dtype=np.float32,
        )
        geometry.create_dataset("ORIGIN", data=origin)
        geometry.create_dataset("SPACING", data=spacing)

        # Cell Data (voxel-level data)
        cell_data = data_container.create_group("CellData")
        cell_data.attrs["AttributeMatrixType"] = np.uint32(3)  # Cell
        cell_data.attrs["TupleDimensions"] = dims

        # Feature IDs (grain IDs) - Flatten in C order, then reshape for DREAM.3D
        feature_ids = microstructure.grain_ids.flatten(order="C").astype(np.int32)
        feature_ids_dataset = cell_data.create_dataset("FeatureIds", data=feature_ids)
        feature_ids_dataset.attrs["ComponentDimensions"] = np.uint64(1)
        feature_ids_dataset.attrs["DataArrayVersion"] = np.int32(2)
        feature_ids_dataset.attrs["ObjectType"] = "DataArray<int32_t>"
        feature_ids_dataset.attrs["TupleDimensions"] = dims

        # Phases (all grains are phase 1)
        phases = np.ones_like(feature_ids, dtype=np.int32)
        phases_dataset = cell_data.create_dataset("Phases", data=phases)
        phases_dataset.attrs["ComponentDimensions"] = np.uint64(1)
        phases_dataset.attrs["DataArrayVersion"] = np.int32(2)
        phases_dataset.attrs["ObjectType"] = "DataArray<int32_t>"
        phases_dataset.attrs["TupleDimensions"] = dims

        # Feature Data (grain-level data)
        num_features = (
            max(microstructure.orientations.keys())
            if microstructure.orientations
            else 0
        )
        feature_data = data_container.create_group("CellFeatureData")
        feature_data.attrs["AttributeMatrixType"] = np.uint32(7)  # CellFeature
        feature_data.attrs["TupleDimensions"] = np.uint64(num_features + 1)

        # Euler angles - DREAM.3D expects all features including feature 0
        euler_angles = np.zeros((num_features + 1, 3), dtype=np.float32)
        for grain_id, angles in microstructure.orientations.items():
            if grain_id <= num_features:
                euler_angles[grain_id] = angles

        euler_dataset = feature_data.create_dataset("EulerAngles", data=euler_angles)
        euler_dataset.attrs["ComponentDimensions"] = np.uint64(3)
        euler_dataset.attrs["DataArrayVersion"] = np.int32(2)
        euler_dataset.attrs["ObjectType"] = "DataArray<float>"
        euler_dataset.attrs["TupleDimensions"] = np.uint64(num_features + 1)

        # Phases for features
        feature_phases = np.ones(num_features + 1, dtype=np.int32)
        feature_phases[0] = 0  # Background is phase 0
        phases_feature_dataset = feature_data.create_dataset(
            "Phases", data=feature_phases
        )
        phases_feature_dataset.attrs["ComponentDimensions"] = np.uint64(1)
        phases_feature_dataset.attrs["DataArrayVersion"] = np.int32(2)
        phases_feature_dataset.attrs["ObjectType"] = "DataArray<int32_t>"
        phases_feature_dataset.attrs["TupleDimensions"] = np.uint64(num_features + 1)

        # Ensemble Data (phase-level data)
        ensemble_data = data_container.create_group("CellEnsembleData")
        ensemble_data.attrs["AttributeMatrixType"] = np.uint32(11)  # CellEnsemble
        ensemble_data.attrs["TupleDimensions"] = np.uint64(2)

        # Crystal structure (1 = Cubic-High, 999 = Unknown)
        crystal_structures = np.array([999, 1], dtype=np.uint32)
        crystal_dataset = ensemble_data.create_dataset(
            "CrystalStructures", data=crystal_structures
        )
        crystal_dataset.attrs["ComponentDimensions"] = np.uint64(1)
        crystal_dataset.attrs["DataArrayVersion"] = np.int32(2)
        crystal_dataset.attrs["ObjectType"] = "DataArray<uint32_t>"
        crystal_dataset.attrs["TupleDimensions"] = np.uint64(2)

        # Material names
        material_names[0] = "Invalid Phase"
        material_names[1] = "Primary Phase"
        material_names.attrs["ComponentDimensions"] = np.uint64(1)
        material_names.attrs["DataArrayVersion"] = np.int32(2)
        material_names.attrs["ObjectType"] = "StringDataArray"
        material_names.attrs["TupleDimensions"] = np.uint64(2)

        print(f"Saved DREAM.3D v8.0 format to {filename}")
        print(f"  Dimensions: {microstructure.dimensions}")
        print(f"  Features: {num_features}")


def write_raw_binary(microstructure, filename_prefix):
    """
    Write raw binary files that DREAM.3D can import
    Creates: .raw (grain IDs) and .txt (metadata)
    """
    # Write grain IDs as raw binary
    raw_filename = f"{filename_prefix}_grains.raw"
    microstructure.grain_ids.astype(np.int32).tofile(raw_filename)

    # Write metadata file
    meta_filename = f"{filename_prefix}_meta.txt"
    with open(meta_filename, "w") as f:
        f.write(
            f"Dimensions: {microstructure.dimensions[0]} {microstructure.dimensions[1]} {microstructure.dimensions[2]}\n"
        )
        f.write(f"Resolution: {microstructure.resolution}\n")
        f.write("Data Type: int32\n")
        f.write("Byte Order: lttle-endian\n")

    print(f"Saved raw binary to {raw_filename}")
    print(f"Metadata saved to {meta_filename}")


def write_hdf5(microstructure, filename):
    """
    Write a simple HDF5 file that can be imported into DREAM.3D
    Uses DREAM.3D's "Import HDF5 Dataset" filter
    """
    with h5py.File(filename, "w") as f:
        # Create a simple, flat structure

        # Grain IDs as 3D array
        grain_ids_dataset = f.create_dataset(
            "GrainIDs", data=microstructure.grain_ids, compression="gzip"
        )

        # Store dimensions as attributes
        grain_ids_dataset.attrs["dimensions"] = microstructure.dimensions
        grain_ids_dataset.attrs["resolution"] = microstructure.resolution
        grain_ids_dataset.attrs["description"] = "Grain/Feature IDs for each voxel"

        # Euler angles - one per grain
        num_grains = (
            max(microstructure.orientations.keys())
            if microstructure.orientations
            else 0
        )
        euler_angles = np.zeros((num_grains + 1, 3), dtype=np.float32)

        for grain_id, angles in microstructure.orientations.items():
            if grain_id <= num_grains:
                euler_angles[grain_id] = angles

        euler_dataset = f.create_dataset("EulerAngles", data=euler_angles)
        euler_dataset.attrs["description"] = (
            "Euler angles (Bunge ZXZ convention) in radians"
        )
        euler_dataset.attrs["convention"] = "Bunge (ZXZ)"
        euler_dataset.attrs["units"] = "radians"
        euler_dataset.attrs["num_grains"] = num_grains

        # Metadata group
        meta = f.create_group("Metadata")
        meta.attrs["generator"] = "Synthetic Microstructure Generator"
        meta.attrs["dimensions"] = microstructure.dimensions
        meta.attrs["resolution"] = microstructure.resolution
        meta.attrs["num_grains"] = num_grains
        meta.attrs["total_voxels"] = np.prod(microstructure.dimensions)

        print(f"Saved HDF5 file to {filename}")
        print(f"  Dimensions: {microstructure.dimensions}")
        print(f"  Resolution: {microstructure.resolution}")
        print(f"  Number of grains: {num_grains}")
        print("\nTo import in DREAM.3D:")
        print("  1. Use filter: 'Import HDF5 Dataset'")
        print("  2. Select dataset: '/GrainIDs'")
        print(f"  3. Set dimensions: {microstructure.dimensions}")
