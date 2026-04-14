# synth_struct/tests/test_phase.py

# synth_struct/tests/test_phase.py

import pytest
import numpy as np
from synth_struct import Microstructure
from synth_struct.orientation.phase import Phase, available_presets
from synth_struct.orientation.phase_constants import KNOWN_PHASES, ALIASES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cubic_phase(name="iron-bcc"):
    """Return a minimal valid cubic Phase."""
    return Phase(name=name, lattice_params=(2.87, 2.87, 2.87), point_group="m-3m")


def _hexagonal_phase(name="titanium"):
    """Return a minimal valid hexagonal Phase."""
    return Phase(name=name, lattice_params=(2.95, 2.95, 4.68), point_group="6/mmm")


def _micro_with_two_phases():
    """Return a 4x4x4 Microstructure pre-loaded with two phases and phase_ids."""
    micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
    p0 = _cubic_phase("alpha")
    p1 = _hexagonal_phase("beta")
    micro.add_phase(0, p0)
    micro.add_phase(1, p1)

    phase_ids = np.zeros((4, 4, 4), dtype=np.int32)
    phase_ids[2:, :, :] = 1  # second half belongs to phase 1
    micro.phase_ids = phase_ids

    micro.grain_ids[:2, :, :] = 1
    micro.grain_ids[2:, :, :] = 2
    return micro, p0, p1


# ---------------------------------------------------------------------------
# Phase initialisation
# ---------------------------------------------------------------------------

class TestPhaseInitialization:

    def test_valid_phase_initializes(self):
        """A Phase with valid arguments should initialize without error."""
        phase = Phase(name="steel", lattice_params=(2.87, 2.87, 2.87), point_group="m-3m")
        assert phase.name == "steel"

    def test_lattice_params_stored(self):
        """lattice_params should be accessible after initialization."""
        phase = _cubic_phase()
        assert phase.lattice_params == (2.87, 2.87, 2.87)

    def test_point_group_stored(self):
        """point_group should be accessible after initialization."""
        phase = _cubic_phase()
        assert phase.point_group == "m-3m"

    def test_space_group_defaults_to_none(self):
        """space_group should default to None when not provided."""
        phase = _cubic_phase()
        assert phase.space_group is None

    def test_space_group_stored_when_provided(self):
        """space_group should be stored correctly when given."""
        phase = Phase(
            name="iron", lattice_params=(2.87, 2.87, 2.87),
            point_group="m-3m", space_group=229
        )
        assert phase.space_group == 229

    def test_metadata_defaults_to_empty_dict(self):
        """metadata should default to an empty dict when not provided."""
        phase = _cubic_phase()
        assert phase.metadata == {}

    def test_metadata_stored_when_provided(self):
        """metadata should be stored correctly when given."""
        phase = Phase(
            name="iron", lattice_params=(2.87, 2.87, 2.87),
            point_group="m-3m", metadata={"reference": "ICSD"}
        )
        assert phase.metadata["reference"] == "ICSD"


# ---------------------------------------------------------------------------
# Phase validation
# ---------------------------------------------------------------------------

class TestPhaseValidation:

    def test_wrong_number_of_lattice_params_raises(self):
        """lattice_params with length != 3 should raise ValueError."""
        with pytest.raises(ValueError, match="lattice_params must be"):
            Phase(name="bad", lattice_params=(2.87, 2.87), point_group="m-3m")

    def test_zero_lattice_parameter_raises(self):
        """A lattice parameter of zero should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            Phase(name="bad", lattice_params=(0.0, 2.87, 2.87), point_group="m-3m")

    def test_negative_lattice_parameter_raises(self):
        """A negative lattice parameter should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            Phase(name="bad", lattice_params=(-1.0, 2.87, 2.87), point_group="m-3m")

    def test_invalid_point_group_raises(self):
        """An unrecognised point group string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown point group"):
            Phase(name="bad", lattice_params=(2.87, 2.87, 2.87), point_group="xyz99")

    def test_space_group_zero_raises(self):
        """space_group of 0 is outside the valid range and should raise ValueError."""
        with pytest.raises(ValueError, match="Space group must be between"):
            Phase(
                name="bad", lattice_params=(2.87, 2.87, 2.87),
                point_group="m-3m", space_group=0
            )

    def test_space_group_231_raises(self):
        """space_group of 231 is outside the valid range and should raise ValueError."""
        with pytest.raises(ValueError, match="Space group must be between"):
            Phase(
                name="bad", lattice_params=(2.87, 2.87, 2.87),
                point_group="m-3m", space_group=231
            )

    def test_space_group_boundary_1_is_valid(self):
        """space_group of 1 is the lowest valid value and should not raise."""
        phase = Phase(
            name="ok", lattice_params=(5.0, 5.0, 5.0),
            point_group="1", space_group=1
        )
        assert phase.space_group == 1

    def test_space_group_boundary_230_is_valid(self):
        """space_group of 230 is the highest valid value and should not raise."""
        phase = Phase(
            name="ok", lattice_params=(2.87, 2.87, 2.87),
            point_group="m-3m", space_group=230
        )
        assert phase.space_group == 230


# ---------------------------------------------------------------------------
# Phase lattice parameter properties
# ---------------------------------------------------------------------------

class TestPhaseProperties:

    def test_a_property(self):
        """Phase.a should return the first lattice parameter."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group="m-3m")
        assert phase.a == 3.0

    def test_b_property(self):
        """Phase.b should return the second lattice parameter."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group="m-3m")
        assert phase.b == 4.0

    def test_c_property(self):
        """Phase.c should return the third lattice parameter."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group="m-3m")
        assert phase.c == 5.0


# ---------------------------------------------------------------------------
# crystal_system property
# ---------------------------------------------------------------------------

class TestCrystalSystem:

    @pytest.mark.parametrize("point_group", ["m-3m", "m-3", "432", "-43m", "23"])
    def test_cubic_point_groups(self, point_group):
        """All cubic point groups should return crystal_system == 'cubic'."""
        phase = Phase(name="x", lattice_params=(3.0, 3.0, 3.0), point_group=point_group)
        assert phase.crystal_system == "cubic"

    @pytest.mark.parametrize("point_group", ["6/mmm", "6mm", "-6m2", "622", "6/m", "-6", "6"])
    def test_hexagonal_point_groups(self, point_group):
        """All hexagonal point groups should return crystal_system == 'hexagonal'."""
        phase = Phase(name="x", lattice_params=(3.0, 3.0, 5.0), point_group=point_group)
        assert phase.crystal_system == "hexagonal"

    @pytest.mark.parametrize("point_group", ["3m", "-3m", "32", "-3", "3"])
    def test_trigonal_point_groups(self, point_group):
        """All trigonal point groups should return crystal_system == 'trigonal'."""
        phase = Phase(name="x", lattice_params=(3.0, 3.0, 5.0), point_group=point_group)
        assert phase.crystal_system == "trigonal"

    @pytest.mark.parametrize("point_group", ["4/mmm", "4mm", "-42m", "422", "4/m", "-4", "4"])
    def test_tetragonal_point_groups(self, point_group):
        """All tetragonal point groups should return crystal_system == 'tetragonal'."""
        phase = Phase(name="x", lattice_params=(3.0, 3.0, 4.0), point_group=point_group)
        assert phase.crystal_system == "tetragonal"

    @pytest.mark.parametrize("point_group", ["mmm", "mm2", "222"])
    def test_orthorhombic_point_groups(self, point_group):
        """All orthorhombic point groups should return crystal_system == 'orthorhombic'."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group=point_group)
        assert phase.crystal_system == "orthorhombic"

    @pytest.mark.parametrize("point_group", ["2/m", "m", "2"])
    def test_monoclinic_point_groups(self, point_group):
        """All monoclinic point groups should return crystal_system == 'monoclinic'."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group=point_group)
        assert phase.crystal_system == "monoclinic"

    @pytest.mark.parametrize("point_group", ["1", "-1"])
    def test_triclinic_point_groups(self, point_group):
        """All triclinic point groups should return crystal_system == 'triclinic'."""
        phase = Phase(name="x", lattice_params=(3.0, 4.0, 5.0), point_group=point_group)
        assert phase.crystal_system == "triclinic"


# ---------------------------------------------------------------------------
# Phase.from_preset
# ---------------------------------------------------------------------------

class TestFromPreset:

    def test_known_preset_returns_phase(self):
        """from_preset should return a Phase for every entry in KNOWN_PHASES."""
        name = next(iter(KNOWN_PHASES))
        phase = Phase.from_preset(name)
        assert isinstance(phase, Phase)

    @pytest.mark.parametrize("name", list(KNOWN_PHASES.keys()))
    def test_all_known_presets_instantiate(self, name):
        """Every preset in KNOWN_PHASES should instantiate without error."""
        phase = Phase.from_preset(name)
        assert isinstance(phase, Phase)

    def test_preset_lattice_params_match_constants(self):
        """from_preset lattice_params should match the KNOWN_PHASES entry exactly."""
        name = next(iter(KNOWN_PHASES))
        phase = Phase.from_preset(name)
        assert phase.lattice_params == KNOWN_PHASES[name]["lattice_params"]

    def test_preset_space_group_matches_constants(self):
        """from_preset space_group should match the KNOWN_PHASES entry exactly."""
        name = next(iter(KNOWN_PHASES))
        phase = Phase.from_preset(name)
        assert phase.space_group == KNOWN_PHASES[name]["space_group"]

    def test_unknown_preset_raises(self):
        """from_preset with an unrecognised name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            Phase.from_preset("not_a_real_material")

    def test_unknown_preset_error_lists_available(self):
        """The ValueError message should list available presets."""
        with pytest.raises(ValueError, match="Available presets"):
            Phase.from_preset("not_a_real_material")


# ---------------------------------------------------------------------------
# available_presets
# ---------------------------------------------------------------------------

class TestAvailablePresets:

    def test_returns_list(self):
        """available_presets should return a list."""
        assert isinstance(available_presets(), list)

    def test_known_phases_included(self):
        """Every key in KNOWN_PHASES should appear in available_presets."""
        presets = available_presets()
        for name in KNOWN_PHASES:
            assert name in presets

    def test_aliases_included(self):
        """Every key in ALIASES should appear in available_presets."""
        presets = available_presets()
        for alias in ALIASES:
            assert alias in presets

    def test_not_empty(self):
        """available_presets should return at least one entry."""
        assert len(available_presets()) > 0


# ---------------------------------------------------------------------------
# Microstructure.add_phase / get_phase
# ---------------------------------------------------------------------------

class TestAddAndGetPhase:

    def test_add_phase_stores_phase(self):
        """add_phase should make the phase retrievable via get_phase."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        phase = _cubic_phase()
        micro.add_phase(0, phase)
        assert micro.get_phase(0) is phase

    def test_add_phase_returns_phase_id(self):
        """add_phase should return the phase_id that was passed in."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        returned = micro.add_phase(5, _cubic_phase())
        assert returned == 5

    def test_add_phase_replaces_existing(self):
        """Adding a phase under an existing ID should replace the previous phase."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase("first"))
        replacement = _cubic_phase("second")
        micro.add_phase(0, replacement)
        assert micro.get_phase(0).name == "second"

    def test_add_non_phase_raises_type_error(self):
        """add_phase should raise TypeError when passed something that is not a Phase."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        with pytest.raises(TypeError):
            micro.add_phase(0, "not_a_phase")

    def test_get_phase_unknown_id_raises(self):
        """get_phase with an unregistered ID should raise KeyError."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        with pytest.raises(KeyError):
            micro.get_phase(99)

    def test_phases_property_reflects_added_phases(self):
        """The phases dict should contain every phase that has been added."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        micro.add_phase(1, _hexagonal_phase())
        assert 0 in micro.phases
        assert 1 in micro.phases


# ---------------------------------------------------------------------------
# Microstructure.phase_ids setter
# ---------------------------------------------------------------------------

class TestPhaseIdsSetter:

    def test_valid_phase_ids_accepted(self):
        """phase_ids matching the microstructure dimensions should be accepted."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        micro.phase_ids = np.zeros((4, 4, 4), dtype=np.int32)
        assert micro.phase_ids is not None

    def test_none_clears_phase_ids(self):
        """Setting phase_ids to None should clear them without error."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        micro.phase_ids = np.zeros((4, 4, 4), dtype=np.int32)
        micro.phase_ids = None
        assert micro.phase_ids is None

    def test_wrong_shape_raises(self):
        """phase_ids with a shape that does not match dimensions should raise ValueError."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        with pytest.raises(ValueError, match="does not match"):
            micro.phase_ids = np.zeros((3, 4, 4), dtype=np.int32)

    def test_unknown_phase_id_raises(self):
        """phase_ids containing an ID not registered via add_phase should raise ValueError."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        ids = np.zeros((4, 4, 4), dtype=np.int32)
        ids[0, 0, 0] = 99  # unknown phase
        with pytest.raises(ValueError, match="unknown phase IDs"):
            micro.phase_ids = ids

    def test_minus_one_is_allowed_in_phase_ids(self):
        """phase_ids may contain -1 as a sentinel value without raising."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        ids = np.zeros((4, 4, 4), dtype=np.int32)
        ids[0, 0, 0] = -1
        micro.phase_ids = ids  # should not raise
        assert micro.phase_ids[0, 0, 0] == -1

    def test_phase_ids_stored_as_int32(self):
        """phase_ids should be cast to int32 regardless of input dtype."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        micro.phase_ids = np.zeros((4, 4, 4), dtype=np.int64)
        assert micro.phase_ids.dtype == np.int32


# ---------------------------------------------------------------------------
# Microstructure.get_phase_for_grain
# ---------------------------------------------------------------------------

class TestGetPhaseForGrain:

    def test_single_phase_returns_that_phase(self):
        """For a single-phase microstructure, any grain should return the sole phase."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        phase = _cubic_phase()
        micro.add_phase(0, phase)
        micro.grain_ids[:, :, :] = 1
        result = micro.get_phase_for_grain(1)
        assert result is phase

    def test_multiphase_returns_correct_phase_for_each_grain(self):
        """In a two-phase microstructure, each grain should map to its assigned phase."""
        micro, p0, p1 = _micro_with_two_phases()
        assert micro.get_phase_for_grain(1) is p0
        assert micro.get_phase_for_grain(2) is p1

    def test_returns_none_when_phase_ids_not_set(self):
        """get_phase_for_grain should return None when phase_ids have not been assigned."""
        micro = Microstructure(dimensions=(4, 4, 4), resolution=1.0)
        micro.add_phase(0, _cubic_phase())
        micro.add_phase(1, _hexagonal_phase())
        micro.grain_ids[:, :, :] = 1
        result = micro.get_phase_for_grain(1)
        assert result is None

    def test_unknown_grain_id_raises(self):
        """get_phase_for_grain should raise ValueError for a grain_id with no voxels."""
        micro, _, _ = _micro_with_two_phases()
        with pytest.raises(ValueError, match="No voxels found"):
            micro.get_phase_for_grain(999)