import math

import numpy as np
import pytest

from policy.deadlock_recovery_v3 import (
    DeadlockRecoveryConfigV3,
    DeadlockRecoveryV3,
    RecoveryDecisionV3,
    deadlock_recovery_mapping_v4_7_pillar,
    horizontal_sector_from_action_id,
)
from policy.runtime_profile_v4_8 import (
    deadlock_recovery_mapping_v4_8_pillar,
)
from policy.runtime_profile_v4_8_1 import (
    deadlock_recovery_mapping_v4_8_1_pillar,
)
from policy.runtime_profile_v4_8_2 import (
    deadlock_recovery_mapping_v4_8_2,
)
from policy.runtime_profile_v4_8_5 import (
    deadlock_recovery_mapping_v4_8_5,
)


def recovery_config(**updates):
    values = deadlock_recovery_mapping_v4_7_pillar({"enabled": True})
    values.update(updates)
    return DeadlockRecoveryConfigV3.from_mapping(values)


def observe(recovery, now_s, feasible=0, speed=0.0, selected=False,
            progress=None, depth=None, action_id=None, sector_id=None,
            clearance=None, position=None):
    if depth is None:
        depth = np.full((20, 30), 5.0, dtype=np.float32)
    return recovery.observe(
        feasible_candidate_count=feasible,
        collision_floor_present=False,
        speed_mps=speed,
        position_world=(
            np.zeros(3) if position is None else np.asarray(position)
        ),
        depth=depth,
        selected_candidate_feasible=selected,
        selected_candidate_goal_progress_m=progress,
        selected_candidate_action_id=action_id,
        selected_candidate_horizontal_sector_id=sector_id,
        selected_candidate_min_observed_clearance_m=clearance,
        now_s=now_s,
    )


def test_v4_8_2_motion_stagnation_triggers_with_formally_safe_candidates():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_2({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    decision = None
    # Candidate count and predicted goal progress remain healthy, matching the
    # failed live run.  Only odometry reveals that translation is stalled.
    for index in range(160):
        decision = observe(
            recovery, index * 0.03,
            feasible=4, selected=True, progress=1.0,
            action_id=7, sector_id=2,
            position=[0.03 * math.sin(index * 0.1), 0.0, 0.0],
        )
        if decision.transition is not None:
            break
    assert decision.transition == "network_stagnation_to_braking"
    assert decision.recovery_trigger_reason == "observed_motion_stagnation"
    assert decision.motion_stagnation_replans == 30
    assert decision.motion_window_displacement_m < 0.20


def test_v4_8_2_normal_translation_never_triggers_motion_recovery():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_2({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    for index in range(240):
        decision = observe(
            recovery, index * 0.03,
            feasible=3, selected=True, progress=1.0,
            action_id=7, sector_id=2,
            position=[index * 0.03, 0.0, 0.0],
        )
        assert decision.mode == recovery.NORMAL
        assert decision.transition is None
    assert decision.motion_stagnation_replans == 0
    assert decision.motion_window_displacement_m > 0.20


def v4_8_5_recovery():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_5({"enabled": True})
    )
    return DeadlockRecoveryV3(config)


def release_v4_8_5_scan(recovery, start_s=0.40, position=None, depth=None):
    now = float(start_s)
    for action_id in (2, 7, 12, 2, 7):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=1.0,
            action_id=action_id, sector_id=2, clearance=0.8,
            position=position, depth=depth,
        )
        now += 0.03
    assert decision.transition == "bounded_scan_to_network_selected_candidate"
    assert decision.handoff_validation_active is True
    return now, decision


def test_v4_8_5_handoff_resets_scan_only_after_measured_escape_motion():
    recovery = v4_8_5_recovery()
    enter_scan(recovery)
    now, decision = release_v4_8_5_scan(recovery)
    assert recovery.unsuccessful_scan_attempts == 1
    assert decision.current_scan_limit_deg == 90.0

    # Candidate availability alone must not reset the escalation chain.
    decision = observe(
        recovery, now, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.49, 0.0, 0.0],
    )
    assert decision.transition is None
    assert decision.handoff_validation_active is True
    assert recovery.unsuccessful_scan_attempts == 1

    decision = observe(
        recovery, now + 0.03, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.51, 0.0, 0.0],
    )
    assert decision.transition == "handoff_measured_motion_verified"
    assert decision.handoff_validation_result == "measured_motion_verified"
    assert decision.handoff_validation_active is False
    assert recovery.unsuccessful_scan_attempts == 0
    assert decision.current_scan_limit_deg == 60.0


def test_v4_8_5_failed_handoff_resumes_same_scan_at_ninety_degrees():
    recovery = v4_8_5_recovery()
    blocked = np.ones((20, 30), dtype=np.float32)
    enter_scan(recovery)
    # Record that the first scan had genuinely moved away from its origin.
    yaw = 0.0
    for _ in range(10):
        yaw, _ = recovery.yaw_command(yaw, 0.02)
    preserved_offset = math.degrees(recovery.scan_offset_rad)
    preserved_direction = recovery.scan_direction
    now, _ = release_v4_8_5_scan(
        recovery, position=[0.0, 0.0, 0.0], depth=blocked
    )

    decision = observe(
        recovery, now + 1.21, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.1, 0.0, 0.0], depth=blocked,
    )
    assert decision.transition == "handoff_validation_failed_to_braking"
    assert decision.mode == recovery.BRAKING
    assert decision.handoff_validation_result == "insufficient_escape_progress"
    assert decision.current_scan_limit_deg == 90.0
    assert math.isclose(
        decision.handoff_preserved_scan_offset_deg,
        preserved_offset, abs_tol=1.0e-9,
    )

    decision = observe(
        recovery, now + 1.24, speed=0.2,
        position=[0.1, 0.0, 0.0], depth=blocked,
    )
    assert decision.transition == (
        "handoff_failure_braking_to_resumed_bounded_scan"
    )
    assert decision.mode == recovery.YAW_SCAN
    assert decision.current_scan_limit_deg == 90.0
    assert math.isclose(decision.scan_offset_deg, preserved_offset, abs_tol=1e-9)
    assert decision.scan_direction == preserved_direction


def test_v4_8_5_second_failed_handoff_escalates_to_one_twenty_degrees():
    recovery = v4_8_5_recovery()
    blocked = np.ones((20, 30), dtype=np.float32)
    enter_scan(recovery)
    now, _ = release_v4_8_5_scan(recovery, depth=blocked)
    observe(recovery, now + 1.21, position=[0.0, 0.0, 0.0], depth=blocked)
    observe(
        recovery, now + 1.24, speed=0.2,
        position=[0.0, 0.0, 0.0], depth=blocked,
    )
    now, _ = release_v4_8_5_scan(
        recovery, start_s=now + 1.27,
        position=[0.0, 0.0, 0.0], depth=blocked,
    )
    assert recovery.current_scan_limit_deg == 120.0
    decision = observe(
        recovery, now + 1.21,
        position=[0.0, 0.0, 0.0], depth=blocked,
    )
    assert decision.transition == "handoff_validation_failed_to_braking"
    decision = observe(
        recovery, now + 1.24, speed=0.2,
        position=[0.0, 0.0, 0.0], depth=blocked,
    )
    assert decision.current_scan_limit_deg == 120.0


def test_dynamic_yield_pause_breaks_confirmation_but_preserves_scan_escalation():
    recovery = v4_8_5_recovery()
    recovery.unsuccessful_scan_attempts = 2
    recovery.current_scan_limit_deg = 120.0
    for index in range(4):
        decision = observe(
            recovery, index * 0.03,
            feasible=3, selected=True, progress=1.0,
            action_id=7, sector_id=2, clearance=0.8,
        )
    assert decision.selected_release_replans == 4
    assert decision.selected_confirmation_horizontal_sector_id is None
    assert decision.selected_candidate_action_id == 7

    decision = recovery.set_dynamic_yield_pause(True, now_s=0.20)
    assert isinstance(decision, RecoveryDecisionV3)
    assert decision.transition == "network_dynamic_yield_pause_started"
    assert decision.mode == recovery.NORMAL
    assert decision.selected_release_replans == 0
    assert decision.selected_candidate_eligible is False
    assert decision.selected_candidate_action_id is None
    assert recovery.selected_confirmation_horizontal_sector_id is None
    assert recovery.unsuccessful_scan_attempts == 2
    assert recovery.current_scan_limit_deg == 120.0

    # Repeated yield frames remain public snapshots and retain escalation.
    decision = recovery.set_dynamic_yield_pause(True, now_s=3.20)
    assert decision.transition is None
    assert recovery.motion_stagnation_replans == 0
    assert recovery.unsuccessful_scan_attempts == 2
    assert recovery.current_scan_limit_deg == 120.0

    decision = recovery.set_dynamic_yield_pause(False, now_s=3.23)
    assert decision.transition == "network_dynamic_yield_pause_ended"
    decision = observe(
        recovery, 3.23,
        feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, clearance=0.8,
    )
    # Four observations before the yield plus one after it must not release or
    # reset the retained escalation evidence as a stitched five-frame streak.
    assert decision.selected_release_replans == 1
    assert recovery.unsuccessful_scan_attempts == 2
    assert recovery.current_scan_limit_deg == 120.0


def test_dynamic_yield_pause_discards_zero_and_motion_stagnation_evidence():
    mapping = deadlock_recovery_mapping_v4_8_5({"enabled": True})
    mapping.update({
        "zero_feasible_trigger_replans": 100,
        "motion_stagnation_window_s": 0.10,
        "motion_stagnation_trigger_replans": 100,
    })
    config = DeadlockRecoveryConfigV3.from_mapping(mapping)
    recovery = DeadlockRecoveryV3(config)
    for index in range(20):
        decision = observe(
            recovery, index * 0.03, feasible=0,
            position=[0.0, 0.0, 0.0],
        )
    assert decision.zero_feasible_replans == 20
    assert decision.motion_stagnation_replans > 0
    assert len(recovery.motion_history) > 0

    decision = recovery.set_dynamic_yield_pause(True, now_s=2.0)
    assert decision.zero_feasible_replans == 0
    assert decision.motion_stagnation_replans == 0
    assert recovery.motion_window_displacement_m is None
    assert len(recovery.motion_history) == 0
    for index in range(100):
        recovery.set_dynamic_yield_pause(True, now_s=2.03 + index * 0.03)
    assert recovery.zero_feasible_replans == 0
    assert recovery.motion_stagnation_replans == 0
    assert len(recovery.motion_history) == 0

    recovery.set_dynamic_yield_pause(False, now_s=5.03)
    decision = observe(
        recovery, 5.03, feasible=0, position=[0.0, 0.0, 0.0]
    )
    assert decision.mode == recovery.NORMAL
    assert decision.transition is None
    assert decision.zero_feasible_replans == 1
    assert decision.motion_stagnation_replans == 0


def test_dynamic_yield_pause_requires_explicit_resume_before_observe():
    recovery = v4_8_5_recovery()
    recovery.set_dynamic_yield_pause(True, now_s=0.0)
    with pytest.raises(RuntimeError, match="resume dynamic-yield pause"):
        observe(recovery, 0.03, feasible=1, selected=True, progress=1.0)
    recovery.set_dynamic_yield_pause(False, now_s=0.04)
    assert observe(recovery, 0.04, feasible=1).mode == recovery.NORMAL


def test_dynamic_yield_pause_freezes_handoff_time_and_motion_evidence():
    recovery = v4_8_5_recovery()
    enter_scan(recovery)
    now, _ = release_v4_8_5_scan(recovery)
    decision = observe(
        recovery, now + 0.40, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.20, 0.0, 0.0],
    )
    assert decision.handoff_validation_active
    assert decision.handoff_validation_displacement_m == pytest.approx(0.20)
    scan_attempts = recovery.unsuccessful_scan_attempts
    scan_limit = recovery.current_scan_limit_deg

    paused = recovery.set_dynamic_yield_pause(True, now_s=now + 0.50)
    frozen_elapsed = paused.handoff_validation_elapsed_s
    assert paused.handoff_validation_active
    assert frozen_elapsed is not None
    # Ten seconds of actor yielding consumes none of the remaining handoff
    # window and cannot alter the already escalated scan chain.
    paused = recovery.set_dynamic_yield_pause(True, now_s=now + 10.50)
    assert paused.handoff_validation_elapsed_s == pytest.approx(frozen_elapsed)
    assert recovery.unsuccessful_scan_attempts == scan_attempts
    assert recovery.current_scan_limit_deg == scan_limit

    resumed = recovery.set_dynamic_yield_pause(False, now_s=now + 10.50)
    assert resumed.handoff_validation_elapsed_s == pytest.approx(frozen_elapsed)
    # Large brake/drift motion during the pause is rebased away.  Only the
    # pre-pause 0.20 m remains as handoff evidence.
    decision = observe(
        recovery, now + 10.51, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[4.0, 0.0, 0.0],
    )
    assert decision.transition is None
    assert decision.handoff_validation_displacement_m == pytest.approx(0.20)
    decision = observe(
        recovery, now + 10.70, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[4.31, 0.0, 0.0],
    )
    assert decision.transition == "handoff_measured_motion_verified"
    assert recovery.unsuccessful_scan_attempts == 0


def test_dynamic_yield_pause_resumes_only_the_remaining_handoff_window():
    recovery = v4_8_5_recovery()
    enter_scan(recovery)
    now, _ = release_v4_8_5_scan(recovery)
    observe(recovery, now + 0.70, position=[0.0, 0.0, 0.0])
    paused = recovery.set_dynamic_yield_pause(True, now_s=now + 0.80)
    elapsed = paused.handoff_validation_elapsed_s
    assert elapsed is not None
    remaining = recovery.config.handoff_validation_window_s - elapsed
    assert 0.0 < remaining < recovery.config.handoff_validation_window_s
    scan_attempts = recovery.unsuccessful_scan_attempts
    scan_limit = recovery.current_scan_limit_deg

    resume_time = now + 8.80
    recovery.set_dynamic_yield_pause(False, now_s=resume_time)
    decision = observe(
        recovery, resume_time + remaining - 0.01,
        position=[0.0, 0.0, 0.0],
    )
    assert decision.transition is None
    assert decision.handoff_validation_active
    decision = observe(
        recovery, resume_time + remaining + 0.01,
        position=[0.0, 0.0, 0.0],
    )
    assert decision.transition == "handoff_validation_failed_to_braking"
    assert recovery.unsuccessful_scan_attempts == scan_attempts
    assert recovery.current_scan_limit_deg == scan_limit


def test_dynamic_yield_pause_does_not_turn_changed_view_into_handoff_evidence():
    recovery = v4_8_5_recovery()
    enter_scan(recovery)
    baseline = np.full((20, 30), 5.0, dtype=np.float32)
    now, _ = release_v4_8_5_scan(recovery, depth=baseline)
    partial = np.full((20, 30), 5.3, dtype=np.float32)
    decision = observe(
        recovery, now + 0.10, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.0, 0.0, 0.0], depth=partial,
    )
    assert decision.handoff_validation_forward_clearance_gain_m == pytest.approx(
        0.30, abs=1.0e-5
    )
    assert decision.handoff_forward_clearance_confirmation_replans == 0

    recovery.set_dynamic_yield_pause(True, now_s=now + 0.15)
    recovery.set_dynamic_yield_pause(False, now_s=now + 5.15)
    changed_view = np.full((20, 30), 20.0, dtype=np.float32)
    decision = observe(
        recovery, now + 5.16, feasible=3, selected=True, progress=1.0,
        action_id=7, sector_id=2, position=[0.0, 0.0, 0.0],
        depth=changed_view,
    )
    # The large view change happened during the pause, so the preserved gain
    # is still 0.30 m and cannot instantly validate the handoff.
    assert decision.transition is None
    assert decision.handoff_validation_forward_clearance_gain_m == pytest.approx(
        0.30, abs=1.0e-5
    )
    assert decision.handoff_forward_clearance_confirmation_replans == 0

    clearer = np.full((20, 30), 20.3, dtype=np.float32)
    for index in range(5):
        decision = observe(
            recovery, now + 5.19 + 0.03 * index,
            feasible=3, selected=True, progress=1.0,
            action_id=7, sector_id=2, position=[0.0, 0.0, 0.0],
            depth=clearer,
        )
    assert decision.transition == "handoff_forward_clearance_verified"


def enter_scan(recovery, trigger=None):
    if trigger is None:
        trigger = recovery.config.zero_feasible_trigger_replans
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
    assert recovery.config.contract()["version"] == (
        "deadlock_recovery_v3_bounded_scan_only"
    )
    assert "accumulate_zero_during_cooldown" not in recovery.config.contract()


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


def test_v4_8_cooldown_preserves_new_zero_candidate_evidence():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    for now in (0.50, 0.53, 0.56):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=0.30
        )
    assert decision.transition == "bounded_scan_to_network_selected_candidate"
    assert recovery.unsuccessful_scan_attempts == 1

    # A vanished handoff route accumulates evidence during cooldown without
    # allowing immediate mode chatter.
    for index in range(config.zero_feasible_trigger_replans):
        decision = observe(recovery, 0.57 + index * 0.03)
        assert decision.mode == recovery.NORMAL
        assert decision.transition is None
    assert decision.zero_feasible_replans == config.zero_feasible_trigger_replans

    # At cooldown expiry the preserved streak immediately starts a wider
    # second attempt instead of waiting for another full dead interval.
    decision = observe(recovery, 1.17, speed=0.4)
    assert decision.transition == "network_to_braking"
    decision = observe(recovery, 1.20, speed=0.2)
    assert decision.transition == "braking_to_bounded_scan"
    assert recovery.current_scan_limit_deg == 90.0


def test_v4_8_sustained_handoff_resets_scan_escalation():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    for now in (0.50, 0.53, 0.56):
        observe(recovery, now, feasible=3, selected=True, progress=0.30)
    assert recovery.unsuccessful_scan_attempts == 1
    for index in range(config.post_release_confirmation_replans):
        observe(
            recovery, 0.59 + 0.03 * index,
            feasible=3, selected=True, progress=0.30,
        )
    assert recovery.unsuccessful_scan_attempts == 0
    assert recovery.current_scan_limit_deg == config.max_scan_angle_deg


def test_horizontal_sector_mapping_keeps_vertical_alternatives_together():
    assert [horizontal_sector_from_action_id(value, 5) for value in (0, 5, 10)] \
        == [0, 0, 0]
    assert [horizontal_sector_from_action_id(value, 5) for value in (4, 9, 14)] \
        == [4, 4, 4]
    assert horizontal_sector_from_action_id(None, 5) is None


def test_v4_8_1_triggers_after_ten_zero_candidate_replans():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_1_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    assert config.zero_feasible_trigger_replans == 10
    assert recovery.mode == recovery.YAW_SCAN


def test_v4_8_1_different_horizontal_sectors_cannot_form_one_handoff():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_1_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    now = 0.40
    for action_id in (0, 1, 0, 1, 0):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=1.0,
            action_id=action_id,
            sector_id=horizontal_sector_from_action_id(action_id, 5),
            clearance=0.8,
        )
        now += 0.03
        assert decision.mode == recovery.YAW_SCAN
        assert decision.transition is None
        assert decision.selected_release_replans == 1

    # Five consecutive observations in one horizontal column are required.
    # Different vertical rows (0, 5, 10) remain valid alternatives.
    for index, action_id in enumerate((2, 7, 12, 2, 7)):
        decision = observe(
            recovery, now, feasible=3, selected=True, progress=1.0,
            action_id=action_id,
            sector_id=horizontal_sector_from_action_id(action_id, 5),
            clearance=0.8,
        )
        now += 0.03
        if index < 4:
            assert decision.mode == recovery.YAW_SCAN
            assert decision.selected_release_replans == index + 1
    assert decision.transition == "bounded_scan_to_network_selected_candidate"
    assert decision.mode == recovery.NORMAL
    assert decision.selected_candidate_action_id == 7
    assert decision.selected_candidate_horizontal_sector_id == 2
    assert decision.selected_candidate_min_observed_clearance_m == 0.8
    assert decision.handoff_confirmation_replans == 5
    assert decision.handoff_horizontal_sector_id == 2


def test_v4_8_1_scan_limit_does_not_bypass_five_frame_confirmation():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_1_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    while not recovery.scan_leg_complete:
        recovery.yaw_command(0.0, 0.05)
    decision = observe(
        recovery, 1.60, feasible=1, selected=True, progress=1.0,
        action_id=3, sector_id=3,
    )
    assert decision.transition == "bounded_scan_limit_to_network"
    assert decision.handoff_confirmation_replans is None


def test_v4_8_1_ineligible_frame_resets_sector_confirmation():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_1_pillar({"enabled": True})
    )
    recovery = DeadlockRecoveryV3(config)
    enter_scan(recovery)
    for now in (0.40, 0.43, 0.46, 0.49):
        decision = observe(
            recovery, now, feasible=1, selected=True, progress=1.0,
            action_id=4, sector_id=4,
        )
    assert decision.selected_release_replans == 4
    decision = observe(recovery, 0.52, feasible=0)
    assert decision.selected_release_replans == 0
    assert decision.selected_confirmation_horizontal_sector_id is None


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
