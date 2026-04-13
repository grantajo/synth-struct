# synth_struct/tests/test_struct_mesh.py

# synth_struct/tests/test_file_output.py

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import pyvista as pv

from synth_struct import Microstructure, Phase
from synth_struct.io.structured_mesh import (
    SolverFormat,
    ElementType,
    _FORMAT_EXTENSIONS,
    _build_regular_grid,
    _validate_element_size,
    _validate_against_voxels,
    _write_format,
    _write_damask_hdf5,
    _write_abaqus_inp,
    _write_custom_hdf5,
    export_microstructure,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _micro_3d(dims=(4, 5, 6), resolution=10.0, num_grains=3):
    """Return a simple 3-D Microstructure with grains filling the volume."""
    micro = Microstructure(dimensions=dims, resolution=resolution)
    cells_per_grain = max(1, int(np.prod(dims) // num_grains))
    flat = micro.grain_ids.ravel()
    for g in range(num_grains):
        start = g * cells_per_grain
        end = start + cells_per_grain if g < num_grains - 1 else len(flat)
        flat[start:end] = g + 1
    micro.grain_ids = flat.reshape(dims)
    return micro


# ---------------------------------------------------------------------------
# SolverFormat / ElementType enums
# ---------------------------------------------------------------------------


class TestSolverFormat:

    def test_damask_vti_value(self):
        """SolverFormat.DAMASK_VTI should have the expected string value."""
        assert SolverFormat.DAMASK_VTI.value == "damask_vti"

    def test_damask_hdf5_value(self):
        """SolverFormat.DAMASK_HDF5 should have the expected string value."""
        assert SolverFormat.DAMASK_HDF5.value == "damask_hdf5"

    def test_abaqus_explicit_value(self):
        """SolverFormat.ABAQUS_EXPLICIT should have the expected string value."""
        assert SolverFormat.ABAQUS_EXPLICIT.value == "abaqus_explicit"

    def test_custom_hdf5_value(self):
        """SolverFormat.CUSTOM_HDF5 should have the expected string value."""
        assert SolverFormat.CUSTOM_HDF5.value == "custom_hdf5"

    @pytest.mark.parametrize(
        "fmt",
        [
            SolverFormat.DAMASK_VTI,
            SolverFormat.DAMASK_HDF5,
            SolverFormat.ABAQUS_EXPLICIT,
            SolverFormat.CUSTOM_HDF5,
        ],
    )
    def test_format_extensions_start_with_dot(self, fmt):
        """Every active SolverFormat should have a dot-prefixed extension."""
        assert fmt in _FORMAT_EXTENSIONS
        assert _FORMAT_EXTENSIONS[fmt].startswith(".")

    def test_format_extensions_correct_values(self):
        """Extension map should contain the expected file suffixes."""
        assert _FORMAT_EXTENSIONS[SolverFormat.DAMASK_VTI] == ".vti"
        assert _FORMAT_EXTENSIONS[SolverFormat.DAMASK_HDF5] == ".hdf5"
        assert _FORMAT_EXTENSIONS[SolverFormat.ABAQUS_EXPLICIT] == ".inp"
        assert _FORMAT_EXTENSIONS[SolverFormat.CUSTOM_HDF5] == ".h5"


class TestElementType:

    def test_hex8_value(self):
        """ElementType.HEX8 should have the expected string value."""
        assert ElementType.HEX8.value == "hex8"


# ---------------------------------------------------------------------------
# _build_regular_grid
# ---------------------------------------------------------------------------


class TestBuildRegularGrid:

    def test_returns_image_data(self):
        """_build_regular_grid should return a PyVista ImageData object."""
        micro = _micro_3d(dims=(4, 5, 6))
        grid = _build_regular_grid(micro)
        assert isinstance(grid, pv.ImageData)

    def test_dimensions_are_cells_plus_one(self):
        """ImageData dimensions should be one greater than cell counts on each axis."""
        micro = _micro_3d(dims=(4, 5, 6))
        grid = _build_regular_grid(micro)
        assert tuple(grid.dimensions) == (5, 6, 7)

    def test_spacing_matches_resolution(self):
        """Grid spacing should equal the microstructure resolution on every axis."""
        micro = _micro_3d(resolution=15.0)
        grid = _build_regular_grid(micro)
        assert grid.spacing == (15.0, 15.0, 15.0)

    def test_origin_is_zero(self):
        """Grid origin should be at (0, 0, 0)."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        assert tuple(grid.origin) == (0.0, 0.0, 0.0)

    def test_grain_id_stored_in_cell_data(self):
        """grain_id field should be present in cell data."""
        micro = _micro_3d(dims=(3, 3, 3))
        grid = _build_regular_grid(micro)
        assert "grain_id" in grid.cell_data

    def test_grain_id_cell_count_matches_volume(self):
        """grain_id length should equal the total number of voxels."""
        dims = (3, 4, 5)
        micro = _micro_3d(dims=dims)
        grid = _build_regular_grid(micro)
        assert grid.cell_data["grain_id"].shape[0] == int(np.prod(dims))

    def test_phase_id_stored_when_present(self):
        micro = _micro_3d()

        # --- Properly define a phase ---
        phase = Phase.from_preset(
            "default_cubic"
        )  # construct however your API requires
        micro.add_phase(0, phase)

        # --- Now assign valid phase IDs ---
        phase_ids = np.zeros_like(micro.grain_ids, dtype=np.int32)
        micro.phase_ids = phase_ids

        grid = _build_regular_grid(micro)

        assert "phase_id" in grid.cell_data

    def test_phase_id_absent_when_none(self):
        """phase_id should not appear in cell data when micro.phase_ids is None."""
        micro = _micro_3d()
        micro.phase_ids = None
        grid = _build_regular_grid(micro)
        assert "phase_id" not in grid.cell_data

    def test_orientation_euler_stored_when_present(self):
        """orientation_euler should appear in cell data when orientations are set."""
        micro = _micro_3d(num_grains=3)
        micro.orientations = np.zeros((micro.num_grains + 1, 3))
        grid = _build_regular_grid(micro)
        assert "orientation_euler" in grid.cell_data

    def test_orientation_euler_absent_when_none(self):
        """orientation_euler should not appear in cell data when orientations are None."""
        micro = _micro_3d()
        micro.orientations = None
        grid = _build_regular_grid(micro)
        assert "orientation_euler" not in grid.cell_data

    def test_stiffness_stored_as_36_components(self):
        """stiffness should be stored as (N_cells, 36) when micro.stiffness is set."""
        micro = _micro_3d(dims=(2, 2, 2), num_grains=2)
        micro.stiffness = np.eye(6)[np.newaxis].repeat(micro.num_grains + 1, axis=0)
        grid = _build_regular_grid(micro)
        assert "stiffness" in grid.cell_data
        n_cells = int(np.prod(micro.dimensions))
        assert grid.cell_data["stiffness"].shape == (n_cells, 36)

    def test_stiffness_absent_when_none(self):
        """stiffness should not appear in cell data when micro.stiffness is None."""
        micro = _micro_3d()
        micro.stiffness = None
        grid = _build_regular_grid(micro)
        assert "stiffness" not in grid.cell_data

    def test_2d_microstructure_does_not_crash(self):
        """A 2-D microstructure should produce a valid ImageData without error."""
        micro = Microstructure(dimensions=(8, 8), resolution=5.0)
        micro.grain_ids[:4, :] = 1
        micro.grain_ids[4:, :] = 2
        grid = _build_regular_grid(micro)
        assert isinstance(grid, pv.ImageData)

    def test_single_grain_microstructure(self):
        """A microstructure with one grain should produce a valid grid."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.grain_ids[:, :, :] = 1
        grid = _build_regular_grid(micro)
        assert np.all(grid.cell_data["grain_id"] == 1)


# ---------------------------------------------------------------------------
# _validate_element_size
# ---------------------------------------------------------------------------


class TestValidateElementSize:

    def test_fine_resolution_prints_sufficient(self, capsys):
        """Resolution of 5 µm should pass the λ/10 check for 1 MHz steel."""
        micro = _micro_3d(resolution=5.0)
        _validate_element_size(micro, frequency_hz=1e6, wave_velocity_ms=6000.0)
        out = capsys.readouterr().out
        assert "sufficient" in out

    def test_coarse_resolution_raises_warning(self):
        """Resolution of 1000 µm should trigger a UserWarning for 10 MHz steel."""
        micro = _micro_3d(resolution=1000.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_element_size(micro, frequency_hz=10e6, wave_velocity_ms=6000.0)
        assert any("too coarse" in str(warning.message) for warning in w)

    def test_resolution_exactly_at_limit_does_not_warn(self):
        """Resolution exactly at the λ/10 threshold should not trigger a warning."""
        # λ = 6000 / 1e6 = 6e-3 m  =>  max_elem = 6e-3 / 10 * 1e6 = 600 µm
        micro = _micro_3d(resolution=600.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_element_size(micro, frequency_hz=1e6, wave_velocity_ms=6000.0)
        coarse_warnings = [x for x in w if "too coarse" in str(x.message)]
        assert len(coarse_warnings) == 0

    def test_slow_wave_velocity_tightens_requirement(self):
        """At 1 MHz in water (1500 m/s), a 200 µm voxel should be too coarse."""
        # λ = 1500 / 1e6 = 1.5e-3 m  =>  max_elem = 150 µm  =>  200 µm > 150 µm
        micro = _micro_3d(resolution=200.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_element_size(micro, frequency_hz=1e6, wave_velocity_ms=1500.0)
        assert any("too coarse" in str(warning.message) for warning in w)

    def test_warning_message_contains_frequency(self):
        """The warning text should include the frequency in MHz."""
        micro = _micro_3d(resolution=1000.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_element_size(micro, frequency_hz=5e6, wave_velocity_ms=6000.0)
        assert any("5.0 MHz" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# _validate_against_voxels
# ---------------------------------------------------------------------------


class TestValidateAgainstVoxels:

    def test_returns_high_match_rate_for_correct_grid(self):
        """Sampling a grid built from the same microstructure should give ≥ 0.90 match."""
        micro = _micro_3d(dims=(8, 8, 8), resolution=10.0)
        grid = _build_regular_grid(micro)
        rate = _validate_against_voxels(micro, grid, sample_fraction=0.1)
        assert rate is not None
        assert rate >= 0.90

    def test_returns_none_when_grain_id_missing(self):
        """Should return None and emit a warning when the mesh has no grain_id field."""
        micro = _micro_3d(dims=(4, 4, 4))
        grid = _build_regular_grid(micro)

        del grid.cell_data["grain_id"]

        with pytest.warns(UserWarning, match="grain_id"):
            result = _validate_against_voxels(micro, grid, sample_fraction=0.1)

        assert result is None

    def test_warns_on_grain_id_field_missing(self):
        """A UserWarning should be emitted when grain_id is absent from the mesh."""
        micro = _micro_3d(dims=(4, 4, 4))
        grid = _build_regular_grid(micro)
        del grid.cell_data["grain_id"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_against_voxels(micro, grid, sample_fraction=0.1)
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_warns_on_high_mismatch(self):
        """Scrambled grain_ids should trigger the mismatch UserWarning."""
        micro = _micro_3d(dims=(6, 6, 6), resolution=10.0, num_grains=10)
        grid = _build_regular_grid(micro)
        grid.cell_data["grain_id"] = (
            np.zeros_like(grid.cell_data["grain_id"]) + 999
        ).astype(np.int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_against_voxels(micro, grid, sample_fraction=1.0)
        assert any("mismatch" in str(warning.message) for warning in w)

    def test_match_rate_is_between_zero_and_one(self):
        """Return value should always be in [0.0, 1.0]."""
        micro = _micro_3d(dims=(6, 6, 6))
        grid = _build_regular_grid(micro)
        rate = _validate_against_voxels(micro, grid, sample_fraction=0.2)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# _write_format dispatch
# ---------------------------------------------------------------------------


class TestWriteFormat:

    def test_damask_vti_creates_file(self, tmp_path):
        """_write_format with DAMASK_VTI should produce a .vti file on disk."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "test.vti")
        _write_format(grid, micro, out, SolverFormat.DAMASK_VTI)
        assert Path(out).exists()

    def test_damask_hdf5_creates_file(self, tmp_path):
        """_write_format with DAMASK_HDF5 should produce an .hdf5 file on disk."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "test.hdf5")
        _write_format(grid, micro, out, SolverFormat.DAMASK_HDF5)
        assert Path(out).exists()

    def test_abaqus_explicit_creates_file(self, tmp_path):
        """_write_format with ABAQUS_EXPLICIT should produce a .inp file on disk."""
        micro = _micro_3d(dims=(2, 2, 2))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "test.inp")
        _write_format(grid, micro, out, SolverFormat.ABAQUS_EXPLICIT)
        assert Path(out).exists()

    def test_custom_hdf5_creates_file(self, tmp_path):
        """_write_format with CUSTOM_HDF5 should produce a .h5 file on disk."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "test.h5")
        _write_format(grid, micro, out, SolverFormat.CUSTOM_HDF5)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# _write_damask_hdf5
# ---------------------------------------------------------------------------


class TestWriteDamaskHDF5:

    def test_required_groups_and_datasets_exist(self, tmp_path):
        """Output HDF5 should contain geometry/cells, size, origin, and material."""
        micro = _micro_3d(dims=(3, 4, 5), resolution=20.0)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "geom.hdf5")
        _write_damask_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            assert "geometry" in f
            for dataset in ("cells", "size", "origin", "material"):
                assert dataset in f["geometry"]

    def test_cells_dataset_matches_dimensions(self, tmp_path):
        """geometry/cells should store the microstructure dimensions exactly."""
        micro = _micro_3d(dims=(3, 4, 5))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "geom.hdf5")
        _write_damask_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["geometry/cells"][:], [3, 4, 5])

    def test_material_is_zero_indexed(self, tmp_path):
        """geometry/material values should start at 0 (DAMASK convention)."""
        micro = _micro_3d(dims=(2, 2, 2), num_grains=3)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "geom.hdf5")
        _write_damask_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            material = f["geometry/material"][:]
        assert material.min() >= 0

    def test_material_max_equals_num_grains_minus_one(self, tmp_path):
        """Maximum material index should be num_grains - 1."""
        micro = _micro_3d(dims=(4, 4, 4), num_grains=3)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "geom.hdf5")
        _write_damask_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            material = f["geometry/material"][:]
        assert material.max() == micro.num_grains - 1

    def test_size_equals_dimensions_times_resolution(self, tmp_path):
        """geometry/size should equal dimensions * resolution for each axis."""
        micro = _micro_3d(dims=(3, 4, 5), resolution=20.0)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "geom.hdf5")
        _write_damask_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            size = f["geometry/size"][:]
        expected = np.array([3, 4, 5]) * 20.0
        np.testing.assert_array_almost_equal(size, expected)


# ---------------------------------------------------------------------------
# _write_abaqus_inp
# ---------------------------------------------------------------------------


class TestWriteAbaqusInp:

    def test_required_keywords_present(self, tmp_path):
        """The .inp file should contain all mandatory Abaqus keyword blocks."""
        micro = _micro_3d(dims=(2, 2, 2))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "model.inp")
        _write_abaqus_inp(grid, micro, out)

        text = Path(out).read_text()
        for keyword in (
            "*Node",
            "*Element",
            "C3D8R",
            "*Elset",
            "*Material",
            "*Step",
            "*End Step",
        ):
            assert keyword in text

    @pytest.mark.parametrize("face", ["X_LO", "X_HI", "Y_LO", "Y_HI", "Z_LO", "Z_HI"])
    def test_boundary_node_sets_present(self, face, tmp_path):
        """All six domain-face node sets should be written to the .inp file."""
        micro = _micro_3d(dims=(2, 2, 2))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "model.inp")
        _write_abaqus_inp(grid, micro, out)
        assert face in Path(out).read_text()

    def test_one_elset_per_grain(self, tmp_path):
        """There should be exactly one *Elset block per unique grain_id."""
        micro = _micro_3d(dims=(4, 4, 4), num_grains=3)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "model.inp")
        _write_abaqus_inp(grid, micro, out)

        text = Path(out).read_text()
        elset_count = text.count("*Elset")
        unique_grains = len(np.unique(grid.cell_data["grain_id"]))
        assert elset_count == unique_grains

    def test_one_material_block_per_grain(self, tmp_path):
        """There should be exactly one *Material block per unique grain_id."""
        micro = _micro_3d(dims=(4, 4, 4), num_grains=3)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "model.inp")
        _write_abaqus_inp(grid, micro, out)

        text = Path(out).read_text()
        material_count = text.count("*Material")
        unique_grains = len(np.unique(grid.cell_data["grain_id"]))
        assert material_count == unique_grains


# ---------------------------------------------------------------------------
# _write_custom_hdf5
# ---------------------------------------------------------------------------


class TestWriteCustomHDF5:

    def test_top_level_groups_exist(self, tmp_path):
        """Output .h5 should contain mesh, grains, phases, and metadata groups."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "custom.h5")
        _write_custom_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            for group in ("mesh", "grains", "phases", "metadata"):
                assert group in f

    def test_grain_id_stored_in_mesh_group(self, tmp_path):
        """mesh/grain_id dataset should be present in the output file."""
        micro = _micro_3d()
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "custom.h5")
        _write_custom_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            assert "grain_id" in f["mesh"]

    def test_orientations_stored_when_present(self, tmp_path):
        """grains/orientations should be written when micro.orientations is set."""
        micro = _micro_3d(num_grains=3)
        micro.orientations = np.zeros((micro.num_grains + 1, 3))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "custom.h5")
        _write_custom_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            assert "orientations" in f["grains"]

    def test_stiffness_stored_when_present(self, tmp_path):
        """grains/stiffness should be written when micro.stiffness is set."""
        micro = _micro_3d(num_grains=3)
        micro.stiffness = np.zeros((micro.num_grains + 1, 6, 6))
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "custom.h5")
        _write_custom_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            assert "stiffness" in f["grains"]

    def test_metadata_resolution_correct(self, tmp_path):
        """metadata/resolution should store the microstructure resolution value."""
        micro = _micro_3d(resolution=25.0)
        grid = _build_regular_grid(micro)
        out = str(tmp_path / "custom.h5")
        _write_custom_hdf5(grid, micro, out)

        with h5py.File(out, "r") as f:
            assert f["metadata/resolution"][()] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# export_microstructure
# ---------------------------------------------------------------------------


class TestExportMicrostructure:

    def test_returns_pyvista_dataset(self, tmp_path):
        """export_microstructure should return a PyVista DataSet."""
        micro = _micro_3d()
        result = export_microstructure(
            micro, str(tmp_path / "out"), SolverFormat.DAMASK_VTI
        )
        assert isinstance(result, pv.DataSet)

    @pytest.mark.parametrize(
        "fmt, extension",
        [
            (SolverFormat.DAMASK_VTI, ".vti"),
            (SolverFormat.DAMASK_HDF5, ".hdf5"),
            (SolverFormat.ABAQUS_EXPLICIT, ".inp"),
            (SolverFormat.CUSTOM_HDF5, ".h5"),
        ],
    )
    def test_output_file_created_for_each_format(self, fmt, extension, tmp_path):
        """Each SolverFormat should produce an output file with the correct extension."""
        micro = _micro_3d(dims=(2, 2, 2))
        export_microstructure(micro, str(tmp_path / "out"), fmt)
        assert (tmp_path / f"out{extension}").exists()

    @pytest.mark.parametrize(
        "fmt, wrong_ext, correct_ext",
        [
            (SolverFormat.DAMASK_VTI, ".wrong", ".vti"),
            (SolverFormat.DAMASK_HDF5, ".txt", ".hdf5"),
            (SolverFormat.ABAQUS_EXPLICIT, ".h5", ".inp"),
        ],
    )
    def test_extension_auto_corrected(self, fmt, wrong_ext, correct_ext, tmp_path):
        """A wrong file extension should be silently corrected to the format's extension."""
        micro = _micro_3d()
        export_microstructure(micro, str(tmp_path / f"out{wrong_ext}"), fmt)
        assert (tmp_path / f"out{correct_ext}").exists()

    def test_wave_check_output_printed_when_frequency_set(self, tmp_path, capsys):
        """Wave check summary line should be printed when target_frequency_hz is provided."""
        micro = _micro_3d(resolution=5.0)
        export_microstructure(
            micro,
            str(tmp_path / "out"),
            SolverFormat.DAMASK_VTI,
            target_frequency_hz=1e6,
            wave_velocity=6000.0,
        )
        assert "Wave check" in capsys.readouterr().out

    def test_wave_check_skipped_when_frequency_none(self, tmp_path, capsys):
        """No wave check output should be printed when target_frequency_hz is None."""
        micro = _micro_3d(resolution=5.0)
        export_microstructure(
            micro,
            str(tmp_path / "out"),
            SolverFormat.DAMASK_VTI,
            target_frequency_hz=None,
        )
        assert "Wave check" not in capsys.readouterr().out

    def test_validate_flag_triggers_spot_check_output(self, tmp_path, capsys):
        """validate=True should run the voxel spot-check and print a Validation line."""
        micro = _micro_3d(dims=(6, 6, 6))
        export_microstructure(
            micro,
            str(tmp_path / "out"),
            SolverFormat.DAMASK_VTI,
            validate=True,
            sample_fraction=0.1,
        )
        assert "Validation" in capsys.readouterr().out

    def test_validate_false_skips_spot_check(self, tmp_path, capsys):
        """validate=False should not print a Validation line."""
        micro = _micro_3d(dims=(6, 6, 6))
        export_microstructure(
            micro,
            str(tmp_path / "out"),
            SolverFormat.DAMASK_VTI,
            validate=False,
        )
        assert "Validation" not in capsys.readouterr().out
