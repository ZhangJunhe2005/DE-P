import numpy as np

from policy.runtime_profile_v4_5_7 import runtime_safety_mapping_v4_5_7
from policy.runtime_safety_v1 import RuntimeSafetyConfigV1, RuntimeTrajectorySafetyV1
from tests.test_runtime_safety_v1 import straight


def config():
    return RuntimeSafetyConfigV1.from_mapping(
        runtime_safety_mapping_v4_5_7({})
    )


def test_v457_retimes_hardware_excess_without_restoring_heuristic_gates():
    value = config()
    assert value.candidate_retiming_enabled is True
    assert value.feasibility_projection_enabled is False
    assert value.enforce_camera_visibility is False
    assert value.enforce_minimum_progress is False
    assert value.enforce_goal_progress is False
    assert value.enforce_observed_tracking_clearance is False
    assert value.enforce_stopping_distance is False


def test_v457_still_rejects_physical_collision_after_retiming():
    shield = RuntimeTrajectorySafetyV1(config())
    aggressive = straight([10.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    candidates, durations, scales, succeeded = shield.retime_candidates(
        [aggressive], 1.7
    )
    assert succeeded == (True,)
    assert scales[0] > 1.0
    result = shield.evaluate(
        candidates, durations, np.asarray([[5.0, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not result.feasible
    assert "observed_collision_floor" in result.reasons
