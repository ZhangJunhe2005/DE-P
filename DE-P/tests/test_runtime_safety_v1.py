import numpy as np
from types import SimpleNamespace

from policy.dynamic.types import CameraModel
from policy.poly_solver import Poly5Solver
from policy.runtime_safety_v1 import (
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
    clamp_vector_norm_v1,
    depth_to_body_points_v1,
)


def straight(endpoint, velocity, duration=1.7):
    return tuple(
        Poly5Solver(0.0, velocity[i], 0.0, endpoint[i], velocity[i], 0.0, duration)
        for i in range(3)
    )


def test_depth_backprojection_uses_optical_to_body_contract():
    model = CameraModel(5, 3, 2.0, 2.0, 2.0, 1.0, 1.0, 0.1, 20.0)
    depth = np.full((3, 5), 20.0, dtype=np.float32)
    depth[1, 2] = 2.0
    rotation = np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    points = depth_to_body_points_v1(depth, model, rotation, stride=1)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [2.0, 0.0, 0.0])


def test_safety_first_selection_rejects_lowest_score_collision():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    collision = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    clear = straight([0.0, 3.4, 0.0], [0.0, 2.0, 0.0])
    evaluations = shield.evaluate(
        [collision, clear], 1.7, np.asarray([[2.0, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )
    selection = shield.select([0.0, 10.0], evaluations)
    assert not evaluations[0].feasible
    assert "observed_collision_floor" in evaluations[0].reasons
    assert evaluations[1].feasible
    assert selection.action_id == 1
    assert selection.mode == "network_safe"


def test_speed_and_acceleration_are_hard_candidate_limits():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    too_fast = straight([11.9, 0.0, 0.0], [7.0, 0.0, 0.0])
    high_acceleration = tuple(
        Poly5Solver(0.0, 0.0, 0.0, 10.0 if axis == 0 else 0.0, 0.0, 0.0, 1.7)
        for axis in range(3)
    )
    evaluations = shield.evaluate(
        [too_fast, high_acceleration], 1.7, np.empty((0, 3)),
        np.zeros(3), np.eye(3),
    )
    assert "speed_limit" in evaluations[0].reasons
    assert "acceleration_limit" in evaluations[1].reasons
    assert shield.select([0.0, 1.0], evaluations).mode == "finite_horizon_braking"


def test_braking_is_finite_and_limit_compliant_from_nominal_flight():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    polynomial, duration, compliant = shield.braking_trajectory(
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    assert polynomial is not None
    assert 0.8 <= duration <= 2.5
    assert compliant
    np.testing.assert_allclose(
        [axis.get_velocity(duration) for axis in polynomial], np.zeros(3), atol=1e-8
    )


def test_command_backstop_clamps_norm_not_each_axis():
    clamped, changed = clamp_vector_norm_v1([6.0, 6.0, 0.0], 6.0)
    assert changed
    np.testing.assert_allclose(np.linalg.norm(clamped), 6.0, atol=1e-12)


def test_progress_contract_rejects_hover_but_not_continuous_motion():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1(minimum_progress_m=1.0))
    hover = straight([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cruise = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluations = shield.evaluate(
        [hover, cruise], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3)
    )
    assert "insufficient_progress" in evaluations[0].reasons
    assert evaluations[1].feasible


def test_hardware_projection_keeps_direction_and_produces_motion():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    endstate = np.zeros((1, 3, 3), dtype=np.float64)
    endstate[0, 0] = [10.0, 9.0, 12.0]
    projected, scales, succeeded = shield.project_endstate_candidates(
        np.zeros(3), np.zeros(3), np.zeros(3), endstate, 1.7
    )
    assert succeeded == (True,)
    assert 0.15 <= scales[0] < 1.0
    evaluation = shield.evaluate(
        projected, 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3)
    )[0]
    assert evaluation.feasible
    assert evaluation.endpoint_progress_m >= 1.0


def test_close_but_collision_free_start_can_escape_margin():
    config = RuntimeSafetyConfigV1(
        vehicle_radius_m=0.30, tracking_margin_m=0.35,
        clearance_sensor_tolerance_m=0.08,
    )
    shield = RuntimeTrajectorySafetyV1(config)
    # Surface lies 0.28 m behind the vehicle. Moving forward increases
    # clearance; requiring 0.65 m at t=0 would make this escape impossible.
    escape = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [escape], 1.7, np.asarray([[-0.28, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert evaluation.feasible


def test_close_start_moving_closer_is_rejected():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    toward = straight([-3.4, 0.0, 0.0], [-2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [toward], 1.7, np.asarray([[-0.40, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not evaluation.feasible
    assert any(reason.startswith("clearance_escape") or reason == "observed_collision_floor"
               for reason in evaluation.reasons)


def test_causal_dynamic_track_prediction_rejects_future_crossing():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    # Actor is currently clear on the left, but crosses the candidate during
    # the 1.7 s horizon.  No simulator identity or future ground truth is used.
    track = SimpleNamespace(
        is_dynamic=True, timestamp=10.0,
        position_world=(1.7, 2.0, 0.0),
        velocity_world=(0.0, -2.35, 0.0),
    )
    evaluation = shield.evaluate(
        [forward], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3),
        dynamic_tracks=(track,), query_timestamp=10.0,
    )[0]
    assert not evaluation.feasible
    assert "predicted_dynamic_clearance" in evaluation.reasons
    assert evaluation.min_predicted_dynamic_clearance_m < 0.0


def test_static_or_stale_tracks_do_not_create_dynamic_veto():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    tracks = (
        SimpleNamespace(
            is_dynamic=False, timestamp=10.0, position_world=(1.7, 0.0, 0.0),
            velocity_world=(0.0, 0.0, 0.0),
        ),
        SimpleNamespace(
            is_dynamic=True, timestamp=9.0, position_world=(1.7, 0.0, 0.0),
            velocity_world=(0.0, 0.0, 0.0),
        ),
    )
    evaluation = shield.evaluate(
        [forward], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3),
        dynamic_tracks=tracks, query_timestamp=10.0,
    )[0]
    assert evaluation.feasible
    assert evaluation.min_predicted_dynamic_clearance_m is None
