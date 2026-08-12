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


def bounded_motion(start, endpoint, duration=1.7):
    return tuple(
        Poly5Solver(start[i], 0.0, 0.0, endpoint[i], 0.0, 0.0, duration)
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
    clear = straight([3.4, 2.5, 0.0], [2.0, 1.47, 0.0])
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


def test_candidate_retiming_preserves_endpoint_and_removes_limit_rejection():
    config = RuntimeSafetyConfigV1(
        candidate_retiming_enabled=True,
        candidate_retiming_max_scale=3.0,
        candidate_retiming_steps=21,
        feasibility_projection_enabled=False,
        enforce_camera_visibility=False,
        enforce_minimum_progress=False,
        enforce_goal_progress=False,
        enforce_observed_tracking_clearance=False,
        enforce_stopping_distance=False,
    )
    shield = RuntimeTrajectorySafetyV1(config)
    too_aggressive = tuple(
        Poly5Solver(
            0.0, 0.0, 0.0,
            10.0 if axis == 0 else 0.0, 0.0, 0.0, 1.7,
        )
        for axis in range(3)
    )
    original = shield.evaluate(
        [too_aggressive], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3)
    )[0]
    assert "speed_limit" in original.reasons
    assert "acceleration_limit" in original.reasons

    candidates, durations, scales, succeeded = shield.retime_candidates(
        [too_aggressive], 1.7
    )
    assert succeeded == (True,)
    assert 1.0 < scales[0] <= 3.0
    assert durations[0] == 1.7 * scales[0]
    np.testing.assert_allclose(
        [axis.get_position(durations[0]) for axis in candidates[0]],
        [10.0, 0.0, 0.0], atol=1.0e-8,
    )
    evaluation = shield.evaluate(
        candidates, durations, np.empty((0, 3)), np.zeros(3), np.eye(3)
    )[0]
    assert evaluation.feasible
    assert "speed_limit" not in evaluation.reasons
    assert "acceleration_limit" not in evaluation.reasons


def test_candidate_retiming_never_authorizes_static_collision():
    config = RuntimeSafetyConfigV1(
        candidate_retiming_enabled=True,
        candidate_retiming_max_scale=3.0,
        candidate_retiming_steps=21,
        feasibility_projection_enabled=False,
        enforce_camera_visibility=False,
        enforce_minimum_progress=False,
        enforce_goal_progress=False,
        enforce_observed_tracking_clearance=False,
        enforce_stopping_distance=False,
    )
    shield = RuntimeTrajectorySafetyV1(config)
    collision = tuple(
        Poly5Solver(
            0.0, 0.0, 0.0,
            10.0 if axis == 0 else 0.0, 0.0, 0.0, 1.7,
        )
        for axis in range(3)
    )
    candidates, durations, _, succeeded = shield.retime_candidates(
        [collision], 1.7
    )
    assert succeeded == (True,)
    evaluation = shield.evaluate(
        candidates, durations, np.asarray([[5.0, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not evaluation.feasible
    assert "observed_collision_floor" in evaluation.reasons


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


def test_goal_progress_rejects_backward_motion_but_allows_visible_bypass():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1(
        minimum_progress_m=1.0, maximum_reverse_progress_m=0.05,
    ))
    backward = straight([-3.4, 0.0, 0.0], [-2.0, 0.0, 0.0])
    lateral = straight([3.0, 1.5, 0.0], [1.76, 0.88, 0.0])
    evaluations = shield.evaluate(
        [backward, lateral], 1.7, np.empty((0, 3)),
        np.zeros(3), np.eye(3), goal_world=np.asarray([10.0, 0.0, 0.0]),
    )
    assert "reverse_goal_progress" in evaluations[0].reasons
    assert evaluations[0].endpoint_goal_progress_m < 0.0
    assert evaluations[1].feasible
    assert evaluations[1].endpoint_goal_progress_m > 0.0


def test_unobserved_side_translation_requires_rotation_first():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    side = straight([0.0, 3.4, 0.0], [0.0, 2.0, 0.0])
    evaluation = shield.evaluate(
        [side], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3)
    )[0]
    assert not evaluation.feasible
    assert "outside_camera_visibility" in evaluation.reasons


def test_physical_stopping_distance_rejects_late_braking_candidate():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [forward], 1.7, np.asarray([[3.0, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not evaluation.feasible
    assert "stopping_distance" in evaluation.reasons
    assert evaluation.minimum_stopping_reserve_m < 0.0


def test_flight_volume_is_a_runtime_hard_boundary():
    shield = RuntimeTrajectorySafetyV1(RuntimeSafetyConfigV1())
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [forward], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3),
        flight_bounds=(np.asarray([-1.0, -1.0, -1.0]),
                       np.asarray([3.0, 1.0, 1.0])),
    )[0]
    assert not evaluation.feasible
    assert "flight_volume" in evaluation.reasons


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


def test_start_inside_tracking_margin_but_outside_body_can_escape():
    config = RuntimeSafetyConfigV1(
        vehicle_radius_m=0.30, tracking_margin_m=0.35,
        clearance_sensor_tolerance_m=0.08,
    )
    shield = RuntimeTrajectorySafetyV1(config)
    # Surface lies 0.40 m behind the vehicle: inside the preferred 0.65 m
    # tracking envelope, but outside the 0.30 m physical body. Moving forward
    # must remain a valid escape instead of deadlocking on the t=0 sample.
    escape = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [escape], 1.7, np.asarray([[-0.40, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert evaluation.feasible


def test_sensor_tolerance_never_shrinks_physical_collision_floor():
    config = RuntimeSafetyConfigV1(
        vehicle_radius_m=0.30, tracking_margin_m=0.35,
        clearance_sensor_tolerance_m=0.08,
        enforce_camera_visibility=False,
        enforce_minimum_progress=False,
        enforce_goal_progress=False,
        enforce_observed_tracking_clearance=False,
        enforce_stopping_distance=False,
    )
    assert config.collision_floor_m == 0.30
    shield = RuntimeTrajectorySafetyV1(config)
    escape = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [escape], 1.7, np.asarray([[-0.28, 0.0, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not evaluation.feasible
    assert "observed_collision_floor" in evaluation.reasons


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


def v43_minimal_config(**overrides):
    return RuntimeSafetyConfigV1(
        feasibility_projection_enabled=False,
        enforce_camera_visibility=False,
        enforce_minimum_progress=False,
        enforce_goal_progress=False,
        enforce_observed_tracking_clearance=False,
        enforce_stopping_distance=False,
        **overrides,
    )


def test_v43_minimal_profile_does_not_reject_fov_or_progress_heuristics():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    side = straight([0.0, 0.5, 0.0], [0.0, 0.3, 0.0])
    evaluation = shield.evaluate(
        [side], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3),
        goal_world=np.asarray([10.0, 0.0, 0.0]),
    )[0]
    assert not evaluation.camera_visible
    assert evaluation.endpoint_progress_m < 1.0
    assert evaluation.feasible
    assert "outside_camera_visibility" not in evaluation.reasons
    assert "insufficient_progress" not in evaluation.reasons


def test_v43_minimal_profile_still_vetoes_causal_dynamic_collision():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    forward = straight([3.4, 0.0, 0.0], [2.0, 0.0, 0.0])
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
    assert evaluation.reasons == ("predicted_dynamic_clearance",)


def test_flight_bounds_include_tracking_error_envelope():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    forward = straight([2.5, 0.0, 0.0], [1.47, 0.0, 0.0])
    evaluation = shield.evaluate(
        [forward], 1.7, np.empty((0, 3)), np.zeros(3), np.eye(3),
        flight_bounds=(np.asarray([-1.0, -1.0, -1.0]),
                       np.asarray([3.0, 1.0, 1.0])),
    )[0]
    assert not evaluation.flight_volume_compliant
    assert "flight_volume" in evaluation.reasons


def test_start_inside_preferred_boundary_band_can_escape_inward():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.50, 0.0, 0.0])
    inward = bounded_motion(start, [0.20, 0.0, 0.0])
    evaluation = shield.evaluate(
        [inward], 1.7, np.empty((0, 3)), start, np.eye(3),
        flight_bounds=bounds,
    )[0]
    assert evaluation.feasible
    assert evaluation.flight_volume_compliant
    assert evaluation.flight_volume_escape
    np.testing.assert_allclose(evaluation.initial_boundary_clearance_m, 0.50)
    np.testing.assert_allclose(evaluation.final_boundary_clearance_m, 0.80)


def test_boundary_escape_must_not_move_outward_or_cross_physical_floor():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.50, 0.0, 0.0])
    outward = bounded_motion(start, [0.65, 0.0, 0.0])
    physical_crossing = bounded_motion(start, [0.80, 0.0, 0.0])
    evaluations = shield.evaluate(
        [outward, physical_crossing], 1.7, np.empty((0, 3)),
        start, np.eye(3), flight_bounds=bounds,
    )
    assert all(not item.feasible for item in evaluations)
    assert all("flight_volume" in item.reasons for item in evaluations)
    assert all(not item.flight_volume_escape for item in evaluations)


def test_start_inside_physical_boundary_floor_cannot_use_escape_exception():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.75, 0.0, 0.0])
    inward = bounded_motion(start, [0.0, 0.0, 0.0])
    evaluation = shield.evaluate(
        [inward], 1.7, np.empty((0, 3)), start, np.eye(3),
        flight_bounds=bounds,
    )[0]
    assert not evaluation.feasible
    assert "flight_volume" in evaluation.reasons
    assert not evaluation.flight_volume_escape
    assert evaluation.initial_boundary_clearance_m < 0.30


def test_boundary_escape_selection_prioritizes_largest_clearance_gain():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    bounds = (np.asarray([-1.0, -1.0, -1.0]),
              np.asarray([1.0, 1.0, 1.0]))
    start = np.asarray([0.50, 0.0, 0.0])
    deep_inward = bounded_motion(start, [0.0, 0.0, 0.0])
    shallow_inward = bounded_motion(start, [0.30, 0.0, 0.0])
    evaluations = shield.evaluate(
        [deep_inward, shallow_inward], 1.7, np.empty((0, 3)),
        start, np.eye(3), flight_bounds=bounds,
    )
    selection = shield.select([100.0, 0.0], evaluations)
    assert selection.mode == "boundary_escape"
    assert selection.action_id == 0


def test_flight_volume_state_reports_six_signed_clearances_and_region():
    shield = RuntimeTrajectorySafetyV1(v43_minimal_config())
    state = shield.flight_volume_state(
        [0.50, 0.0, 0.0],
        (np.asarray([-1.0, -1.0, -1.0]),
         np.asarray([1.0, 1.0, 1.0])),
    )
    assert state["region"] == "recovery_band"
    assert state["minimum_signed_clearance_m"] == 0.50
    assert state["signed_clearance_m"] == {
        "x_min": 1.5, "y_min": 1.0, "z_min": 1.0,
        "x_max": 0.5, "y_max": 1.0, "z_max": 1.0,
    }
