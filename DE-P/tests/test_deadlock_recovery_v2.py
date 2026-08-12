import numpy as np

from policy.deadlock_recovery_v2 import (
    DeadlockRecoveryConfigV2,
    DeadlockRecoveryV2,
)


def config(**updates):
    values = {
        "zero_feasible_trigger_replans": 2,
        "collision_floor_trigger_replans": 2,
        "release_feasible_replans": 2,
        "stationary_speed_mps": 0.35,
        "yaw_scan_rate_deg_s": 90.0,
        "max_scan_angle_deg": 45.0,
        "max_scan_legs": 2,
        "breadcrumb_spacing_m": 0.1,
        "retreat_distance_m": 0.8,
        "retreat_arrival_radius_m": 0.15,
        "observed_escape_distance_m": 1.2,
        "observed_escape_arrival_radius_m": 0.2,
        "observed_escape_min_depth_m": 2.0,
    }
    values.update(updates)
    return DeadlockRecoveryConfigV2(**values)


def enter_scan(recovery, depth):
    position = np.zeros(3)
    recovery.observe(0, False, 1.0, position, depth, np.eye(3))
    assert recovery.observe(
        0, False, 1.0, position, depth, np.eye(3)
    ).transition == "network_to_braking"
    assert recovery.observe(
        0, False, 0.0, position, depth, np.eye(3)
    ).transition == "braking_to_yaw_scan"


def finish_scan_leg(recovery):
    for _ in range(20):
        recovery.yaw_command(0.0, 0.1)
        if recovery.scan_leg_complete:
            return
    raise AssertionError("scan leg did not reach its bounded angle")


def test_bounded_scan_enters_observed_escape_when_corridor_is_clear():
    recovery = DeadlockRecoveryV2(config())
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    enter_scan(recovery, depth)
    finish_scan_leg(recovery)
    decision = recovery.observe(0, False, 0.0, np.zeros(3), depth, np.eye(3))
    assert decision.transition == "yaw_scan_to_observed_escape"
    assert decision.mode == recovery.OBSERVED_ESCAPE
    np.testing.assert_allclose(decision.escape_target_world, [1.2, 0.0, 0.0])


def test_blocked_first_leg_reverses_then_bounded_hold():
    recovery = DeadlockRecoveryV2(config())
    blocked = np.full((20, 30), 1.0, dtype=np.float32)
    enter_scan(recovery, blocked)
    finish_scan_leg(recovery)
    decision = recovery.observe(
        0, False, 0.0, np.zeros(3), blocked, np.eye(3)
    )
    assert decision.transition == "yaw_scan_reverse"
    assert recovery.scan_direction < 0
    finish_scan_leg(recovery)
    decision = recovery.observe(
        0, False, 0.0, np.zeros(3), blocked, np.eye(3)
    )
    assert decision.transition == "yaw_scan_exhausted_to_hold"
    assert decision.mode == recovery.HOLD


def test_observed_escape_rejection_does_not_loop_forever():
    recovery = DeadlockRecoveryV2(config(max_scan_legs=1))
    clear = np.full((20, 30), 5.0, dtype=np.float32)
    enter_scan(recovery, clear)
    finish_scan_leg(recovery)
    recovery.observe(0, False, 0.0, np.zeros(3), clear, np.eye(3))
    decision = recovery.reject_observed_escape()
    assert decision.transition == "observed_escape_rejected_to_hold"
    assert decision.mode == recovery.HOLD


def test_yaw_scan_releases_after_short_stable_feasible_view():
    recovery = DeadlockRecoveryV2(config(
        release_feasible_replans=5,
        scan_release_feasible_replans=2,
    ))
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    enter_scan(recovery, depth)
    first = recovery.observe(
        1, False, 0.0, np.zeros(3), depth, np.eye(3)
    )
    assert first.mode == recovery.YAW_SCAN
    second = recovery.observe(
        1, False, 0.0, np.zeros(3), depth, np.eye(3)
    )
    assert second.transition == "yaw_scan_to_network"
    assert second.mode == recovery.NORMAL


def test_yaw_scan_does_not_release_for_only_non_goal_feasible_candidates():
    recovery = DeadlockRecoveryV2(config(
        release_feasible_replans=5,
        scan_release_feasible_replans=2,
    ))
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    enter_scan(recovery, depth)
    for _ in range(4):
        decision = recovery.observe(
            3, False, 0.0, np.zeros(3), depth, np.eye(3),
            goal_feasible_candidate_count=0,
        )
        assert decision.mode == recovery.YAW_SCAN
        assert decision.transition is None
    recovery.observe(
        3, False, 0.0, np.zeros(3), depth, np.eye(3),
        goal_feasible_candidate_count=1,
    )
    decision = recovery.observe(
        3, False, 0.0, np.zeros(3), depth, np.eye(3),
        goal_feasible_candidate_count=1,
    )
    assert decision.transition == "yaw_scan_to_network"


def test_hold_releases_for_independently_certified_boundary_escape():
    recovery = DeadlockRecoveryV2(config(
        release_feasible_replans=1,
        scan_release_feasible_replans=1,
    ))
    recovery.mode = recovery.HOLD
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    decision = recovery.observe(
        feasible_candidate_count=1,
        collision_floor_present=False,
        speed_mps=0.0,
        position_world=np.zeros(3),
        depth=depth,
        goal_feasible_candidate_count=0,
        recovery_feasible_candidate_count=1,
    )
    assert decision.transition == "hold_to_network"
    assert decision.mode == recovery.NORMAL


def test_goal_biases_scan_and_rejects_backward_observed_escape():
    recovery = DeadlockRecoveryV2(config(
        max_scan_angle_deg=45.0,
        observed_escape_min_goal_alignment=-0.05,
        goal_biased_scan_deadband_deg=5.0,
    ))
    depth = np.full((20, 30), 5.0, dtype=np.float32)
    position = np.zeros(3)
    goal = np.array([1.0, -2.0, 0.0])
    recovery.observe(0, False, 1.0, position, depth, np.eye(3), goal)
    recovery.observe(0, False, 1.0, position, depth, np.eye(3), goal)
    decision = recovery.observe(
        0, False, 0.0, position, depth, np.eye(3), goal
    )
    assert decision.transition == "braking_to_yaw_scan"
    assert recovery.scan_direction < 0

    # Force the camera to face directly away from the goal at the leg end.
    finish_scan_leg(recovery)
    body_x_backwards = np.diag([-1.0, -1.0, 1.0])
    decision = recovery.observe(
        0, False, 0.0, position, depth, body_x_backwards,
        np.array([1.0, 0.0, 0.0]),
    )
    assert decision.transition == "yaw_scan_reverse"
    assert decision.escape_goal_alignment == -1.0
