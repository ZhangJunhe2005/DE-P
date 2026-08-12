import numpy as np

from policy.runtime_profile_v4_5 import runtime_safety_mapping_v4_5
from policy.runtime_safety_v1 import RuntimeSafetyConfigV1, RuntimeTrajectorySafetyV1
from tests.test_runtime_safety_v1 import bounded_motion


def config():
    return RuntimeSafetyConfigV1.from_mapping(runtime_safety_mapping_v4_5({}))


def test_v45_keeps_obstacle_and_stopping_safety_but_softens_outer_preference():
    result = config()
    assert result.enforce_observed_tracking_clearance is True
    assert result.enforce_stopping_distance is True
    assert result.enforce_preferred_flight_volume_clearance is False
    assert result.prefer_largest_boundary_escape_clearance is False
    assert result.collision_floor_m == 0.30
    assert np.isclose(result.required_clearance_m, 0.65)


def test_v45_outer_boundary_uses_physical_floor_and_preserves_network_score():
    shield = RuntimeTrajectorySafetyV1(config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.50, 0.0, 0.0])
    deep_inward = bounded_motion(start, [0.0, 0.0, 0.0])
    shallow_inward = bounded_motion(start, [0.30, 0.0, 0.0])
    evaluations = shield.evaluate(
        [deep_inward, shallow_inward], 1.7, np.empty((0, 3)),
        start, np.eye(3), flight_bounds=bounds,
    )
    assert all(item.feasible for item in evaluations)
    assert all(not item.flight_volume_escape for item in evaluations)
    selection = shield.select([100.0, 0.0], evaluations)
    assert selection.mode == "network_safe"
    assert selection.action_id == 1


def test_v45_still_rejects_crossing_the_physical_outer_boundary_floor():
    shield = RuntimeTrajectorySafetyV1(config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.50, 0.0, 0.0])
    crossing = bounded_motion(start, [0.80, 0.0, 0.0])
    evaluation = shield.evaluate(
        [crossing], 1.7, np.empty((0, 3)), start, np.eye(3),
        flight_bounds=bounds,
    )[0]
    assert not evaluation.feasible
    assert "flight_volume" in evaluation.reasons
