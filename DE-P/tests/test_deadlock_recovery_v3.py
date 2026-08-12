import math

import numpy as np

from policy.deadlock_recovery_v3 import (
    DeadlockRecoveryConfigV3,
    DeadlockRecoveryV3,
    deadlock_recovery_mapping_v4_7_pillar,
)


def recovery_config(**updates):
    values = deadlock_recovery_mapping_v4_7_pillar({"enabled": True})
    values.update(updates)
    return DeadlockRecoveryConfigV3.from_mapping(values)


def observe(recovery, now_s, feasible=0, speed=0.0, selected=False,
            progress=None, depth=None):
    if depth is None:
        depth = np.full((20, 30), 5.0, dtype=np.float32)
    return recovery.observe(
        feasible_candidate_count=feasible,
        collision_floor_present=False,
        speed_mps=speed,
        position_world=np.zeros(3),
        depth=depth,
        selected_candidate_feasible=selected,
        selected_candidate_goal_progress_m=progress,
        now_s=now_s,
    )


def enter_scan(recovery, trigger=15):
    for index in range(trigger - 1):
        decision = observe(recovery, index * 0.03, speed=0.4)
        assert decision.mode == recovery.NORMAL
    decision = observe(recovery, (trigger - 1) * 0.03, speed=0.4)
    assert decision.transition == "network_to_braking"
    decision = observe(recovery, trigger * 0.03, speed=0.2)
    assert decision.transition == "braking_to_bounded_scan"


def test_only_fifteenth_consecutive_zero_candidate_replan_triggers():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    assert recovery.mode == recovery.YAW_SCAN


def test_safe_candidate_breaks_zero_streak_and_normal_mode_never_takes_over():
    recovery = DeadlockRecoveryV3(recovery_config())
    for index in range(14):
        observe(recovery, index * 0.03)
    decision = observe(
        recovery, 0.45, feasible=1, selected=True, progress=1.0
    )
    assert decision.mode == recovery.NORMAL
    assert decision.transition is None
    assert decision.zero_feasible_replans == 0


def test_scan_releases_only_for_selected_safe_progress_candidate():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    # Other feasible candidates do not release a network selection that lacks
    # the minimum positive goal progress.
    for now in (0.50, 0.53, 0.56):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=0.10
        )
        assert decision.mode == recovery.YAW_SCAN
    for now in (0.59, 0.62):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=0.30
        )
        assert decision.mode == recovery.YAW_SCAN
    decision = observe(
        recovery, 0.65, feasible=3, selected=True, progress=0.30
    )
    assert decision.transition == (
        "bounded_scan_to_network_selected_candidate"
    )
    assert decision.mode == recovery.NORMAL
    assert recovery.heading_commitment_active(now_s=0.72)
    assert not recovery.heading_commitment_active(now_s=1.17)


def test_scan_pauses_at_useful_view_while_release_is_confirmed():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    yaw, yaw_dot = recovery.yaw_command(0.0, 0.10)
    assert yaw_dot != 0.0
    observe(
        recovery, 0.50, feasible=2, selected=True, progress=0.30
    )
    held_yaw, yaw_dot = recovery.yaw_command(yaw, 0.10)
    assert held_yaw == yaw
    assert yaw_dot == 0.0
    # A one-frame flicker does not release translation and scanning resumes
    # when the selected candidate is no longer eligible.
    observe(recovery, 0.53, feasible=0, selected=False, progress=None)
    resumed_yaw, yaw_dot = recovery.yaw_command(held_yaw, 0.10)
    assert yaw_dot != 0.0
    assert resumed_yaw != held_yaw


def test_scan_limit_with_current_safe_progress_candidate_hands_off():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    while not recovery.scan_leg_complete:
        recovery.yaw_command(0.0, 0.05)
    decision = observe(
        recovery, 1.60, feasible=2, selected=True, progress=0.30
    )
    assert decision.transition == (
        "bounded_scan_limit_to_network_selected_candidate"
    )
    assert decision.mode == recovery.NORMAL
    assert recovery.heading_commitment_active(now_s=1.61)


def test_collision_floor_never_releases_scan_translation():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    for now in (0.50, 0.53, 0.56, 0.59):
        decision = recovery.observe(
            feasible_candidate_count=3,
            collision_floor_present=True,
            speed_mps=0.0,
            position_world=np.zeros(3),
            depth=np.full((20, 30), 5.0, dtype=np.float32),
            selected_candidate_feasible=True,
            selected_candidate_goal_progress_m=1.0,
            now_s=now,
        )
        assert decision.mode == recovery.YAW_SCAN
        assert not decision.selected_candidate_eligible


def test_scan_is_one_leg_sixty_degrees_and_never_enters_hold():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    yaw = 0.0
    for _ in range(100):
        yaw, _ = recovery.yaw_command(yaw, 0.02)
    assert recovery.scan_leg_complete
    assert math.isclose(
        abs(math.degrees(recovery.scan_offset_rad)), 60.0, abs_tol=1.0e-9
    )
    decision = observe(recovery, 1.60)
    assert decision.transition == "bounded_scan_limit_to_network"
    assert decision.mode == recovery.NORMAL
    assert decision.cooldown_remaining_s == recovery.config.cooldown_s


def test_timeout_returns_to_network_and_cooldown_suppresses_reentry():
    recovery = DeadlockRecoveryV3(recovery_config(scan_timeout_s=0.5))
    enter_scan(recovery)
    decision = observe(recovery, 1.0)
    assert decision.transition == "bounded_scan_timeout_to_network"
    assert decision.mode == recovery.NORMAL
    for index in range(100):
        decision = observe(recovery, 1.01 + index * 0.01)
        assert decision.mode == recovery.NORMAL
        assert decision.transition is None


def test_new_goal_reset_clears_scan_and_cooldown():
    recovery = DeadlockRecoveryV3(recovery_config())
    enter_scan(recovery)
    recovery.reset([1.0, 2.0, 3.0])
    assert recovery.mode == recovery.NORMAL
    assert recovery.cooldown_until_s == 0.0
    assert recovery.heading_commitment_until_s == 0.0
    assert recovery.zero_feasible_replans == 0


def test_scan_direction_uses_freer_depth_half_without_goal_bias():
    depth = np.ones((20, 30), dtype=np.float32)
    depth[:, :15] = 6.0
    assert DeadlockRecoveryV3.scan_direction_from_depth(depth) > 0.0
    depth[:, :15], depth[:, 15:] = 1.0, 6.0
    assert DeadlockRecoveryV3.scan_direction_from_depth(depth) < 0.0
