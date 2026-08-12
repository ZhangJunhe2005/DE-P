import numpy as np
import pytest

from policy.runtime_profile_v4_5_7 import runtime_safety_mapping_v4_5_7
from policy.runtime_profile_v4_5_10 import runtime_safety_mapping_v4_5_10
from policy.runtime_safety_v1 import RuntimeSafetyConfigV1, RuntimeTrajectorySafetyV1
from tests.test_runtime_safety_v1 import straight


def config(mapping):
    return RuntimeSafetyConfigV1.from_mapping(mapping({}))


def test_v4510_changes_only_representation_floor_over_v457():
    old = config(runtime_safety_mapping_v4_5_7)
    new = config(runtime_safety_mapping_v4_5_10)
    assert old.collision_floor_m == pytest.approx(0.30)
    assert new.vehicle_radius_m == pytest.approx(0.30)
    assert new.collision_representation_margin_m == pytest.approx(0.05)
    assert new.collision_floor_m == pytest.approx(0.35)
    old_contract = old.contract()
    new_contract = new.contract()
    ignored = {"collision_representation_margin_m", "collision_floor_m"}
    assert {
        key: value for key, value in old_contract.items() if key not in ignored
    } == {
        key: value for key, value in new_contract.items() if key not in ignored
    }


def test_v4510_rejects_34cm_representation_risk_while_v457_accepts_it():
    candidate = straight([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obstacle = np.asarray([[0.5, 0.34, 0.0]])
    arguments = (candidate,), 1.7, obstacle, np.zeros(3), np.eye(3)
    old = RuntimeTrajectorySafetyV1(
        config(runtime_safety_mapping_v4_5_7)
    ).evaluate(*arguments)[0]
    new = RuntimeTrajectorySafetyV1(
        config(runtime_safety_mapping_v4_5_10)
    ).evaluate(*arguments)[0]
    assert old.feasible
    assert not new.feasible
    assert "observed_collision_floor" in new.reasons


def test_v4510_preserves_next_feasible_selection_and_retiming():
    value = config(runtime_safety_mapping_v4_5_10)
    assert value.candidate_retiming_enabled
    assert not value.feasibility_projection_enabled
    assert not value.enforce_camera_visibility
    assert not value.enforce_minimum_progress
    assert not value.enforce_goal_progress
    assert not value.enforce_observed_tracking_clearance
    assert not value.enforce_stopping_distance
