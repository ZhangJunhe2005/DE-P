import numpy as np

from policy.deadlock_recovery_v3 import DeadlockRecoveryConfigV3
from policy.runtime_profile_v4_8_2 import runtime_safety_mapping_v4_8_2
from policy.runtime_profile_v4_8_5 import (
    RUNTIME_BEHAVIOR_VERSION,
    calculate_recovery_continuity_yaw_v4_8_5,
    deadlock_recovery_mapping_v4_8_5,
    runtime_safety_mapping_v4_8_5,
)


def test_v4_8_5_changes_no_candidate_safety_rule():
    base = {"enabled": True}
    assert runtime_safety_mapping_v4_8_5(base) == runtime_safety_mapping_v4_8_2(base)


def test_v4_8_5_enables_one_scene_agnostic_verified_handoff_contract():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_5({"enabled": True})
    )
    assert config.provisional_handoff_validation_enabled is True
    assert config.handoff_validation_window_s == 1.20
    assert config.handoff_min_displacement_m == 0.50
    assert config.handoff_min_forward_clearance_gain_m == 0.50
    assert config.handoff_forward_clearance_confirmation_replans == 5
    assert config.max_scan_angle_deg == 60.0
    assert config.scan_angle_step_deg == 30.0
    assert config.max_escalated_scan_angle_deg == 120.0
    assert config.contract()["version"] == (
        "deadlock_recovery_v4_3_motion_verified_handoff"
    )
    assert RUNTIME_BEHAVIOR_VERSION == (
        "v4_8_5_universal_motion_verified_recovery_handoff_v1"
    )


def test_v4_8_5_reuses_v4_8_2_selected_path_yaw():
    yaw, yaw_rate = calculate_recovery_continuity_yaw_v4_8_5(
        velocity_world=np.asarray([0.0, 2.0, 0.0]),
        goal_direction_world=np.asarray([10.0, 0.0, 0.0]),
        last_yaw=1.4,
        dt=0.1,
    )
    assert yaw > 1.3
    assert yaw_rate > 0.0
