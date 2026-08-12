import numpy as np

from policy.deadlock_recovery_v2 import (
    DeadlockRecoveryConfigV2,
    DeadlockRecoveryV2,
)
from policy.poly_solver import Poly5Solver
from policy.runtime_profile_v4_4 import (
    calculate_original_yopo_yaw_v4_4,
    deadlock_recovery_mapping_v4_4,
    runtime_safety_mapping_v4_4,
)
from policy.runtime_safety_v1 import (
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
)


def test_v44_profile_preserves_dynamic_filter_and_one_physical_envelope():
    result = runtime_safety_mapping_v4_4({
        "dynamic_track_prediction_enabled": True,
        "max_speed_mps": 6.0,
    })
    assert result["dynamic_track_prediction_enabled"] is True
    assert result["max_speed_mps"] == 6.0
    for key in (
        "feasibility_projection_enabled", "enforce_camera_visibility",
        "enforce_minimum_progress", "enforce_goal_progress",
    ):
        assert result[key] is False
    assert result["enforce_observed_tracking_clearance"] is True
    assert result["enforce_stopping_distance"] is True


def test_v44_rejects_a_body_intersection_with_negative_stopping_reserve():
    mapping = runtime_safety_mapping_v4_4({
        "feasibility_projection_enabled": True,
        "enforce_camera_visibility": True,
        "enforce_minimum_progress": True,
        "enforce_goal_progress": True,
        "enforce_observed_tracking_clearance": False,
        "enforce_stopping_distance": False,
    })
    shield = RuntimeTrajectorySafetyV1(
        RuntimeSafetyConfigV1.from_mapping(mapping)
    )
    candidate = tuple(
        Poly5Solver(
            0.0, 3.0 if axis == 0 else 0.0, 0.0,
            5.0 if axis == 0 else 0.0,
            3.0 if axis == 0 else 0.0, 0.0, 1.7,
        )
        for axis in range(3)
    )
    evaluation = shield.evaluate(
        [candidate], 1.7, np.asarray([[2.5, 0.28, 0.0]]),
        np.zeros(3), np.eye(3),
    )[0]
    assert not evaluation.feasible
    assert evaluation.min_observed_clearance_m < 0.30
    assert evaluation.minimum_stopping_reserve_m < 0.0
    assert "observed_collision_floor" in evaluation.reasons
    assert "stopping_distance" in evaluation.reasons


def recovery():
    base = {
        "enabled": True,
        "zero_feasible_trigger_replans": 15,
        "collision_floor_trigger_replans": 5,
        "release_feasible_replans": 5,
        "scan_release_feasible_replans": 2,
        "stationary_speed_mps": 0.35,
        "yaw_scan_rate_deg_s": 45.0,
        "max_scan_angle_deg": 90.0,
        "max_scan_legs": 2,
        "breadcrumb_spacing_m": 0.12,
        "breadcrumb_capacity": 600,
        "retreat_enabled": True,
        "retreat_distance_m": 1.2,
        "retreat_arrival_radius_m": 0.2,
        "observed_escape_enabled": True,
        "observed_escape_distance_m": 1.2,
        "observed_escape_arrival_radius_m": 0.2,
        "observed_escape_min_depth_m": 2.0,
        "observed_escape_min_goal_alignment": -0.05,
        "goal_biased_scan_deadband_deg": 10.0,
    }
    config = DeadlockRecoveryConfigV2.from_mapping(
        deadlock_recovery_mapping_v4_4(base)
    )
    assert not config.retreat_enabled
    assert not config.observed_escape_enabled
    assert config.zero_feasible_trigger_replans == 15
    assert config.yaw_scan_rate_deg_s == 90.0
    return DeadlockRecoveryV2(config)


def observe(value, feasible, goal_feasible, speed):
    return value.observe(
        feasible_candidate_count=feasible,
        goal_feasible_candidate_count=goal_feasible,
        collision_floor_present=False,
        speed_mps=speed,
        position_world=np.zeros(3),
        depth=np.full((90, 160), 5.0, dtype=np.float32),
        rotation_world_from_body=np.eye(3),
        goal_world=np.asarray([10.0, 0.0, 0.0]),
    )


def test_v44_scan_is_rare_and_releases_immediately_to_network():
    value = recovery()
    for _ in range(14):
        decision = observe(value, feasible=0, goal_feasible=0, speed=0.0)
        assert decision.mode == DeadlockRecoveryV2.NORMAL
    decision = observe(value, feasible=0, goal_feasible=0, speed=0.0)
    assert decision.mode == DeadlockRecoveryV2.BRAKING
    decision = observe(value, feasible=0, goal_feasible=0, speed=0.0)
    assert decision.mode == DeadlockRecoveryV2.YAW_SCAN
    decision = observe(value, feasible=3, goal_feasible=1, speed=0.0)
    assert decision.mode == DeadlockRecoveryV2.NORMAL
    assert decision.transition == "yaw_scan_to_network"


def test_v44_normal_yaw_matches_original_ninety_degree_per_second_limit():
    yaw, yaw_rate = calculate_original_yopo_yaw_v4_4(
        velocity_world=np.asarray([1.0, 0.0, 0.0]),
        goal_direction_world=np.asarray([-10.0, 0.0, 0.0]),
        last_yaw=0.0,
        dt=0.1,
    )
    np.testing.assert_allclose(abs(yaw), np.deg2rad(9.0), atol=1.0e-7)
    np.testing.assert_allclose(abs(yaw_rate), np.deg2rad(90.0), atol=1.0e-7)


def test_v44_original_yaw_turns_toward_goal_instead_of_outward_velocity():
    yaw, _ = calculate_original_yopo_yaw_v4_4(
        velocity_world=np.asarray([1.0, 0.0, 0.0]),
        goal_direction_world=np.asarray([0.0, 10.0, 0.0]),
        last_yaw=0.0,
        dt=0.1,
    )
    assert yaw > 0.0
