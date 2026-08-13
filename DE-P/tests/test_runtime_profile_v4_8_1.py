import numpy as np

from policy.deadlock_recovery_v3 import DeadlockRecoveryConfigV3
from policy.runtime_profile_v4_8 import runtime_safety_mapping_v4_8
from policy.runtime_profile_v4_8_1 import (
    RUNTIME_BEHAVIOR_VERSION,
    calculate_recovery_continuity_yaw_v4_8_1,
    deadlock_recovery_mapping_v4_8_1_pillar,
    runtime_safety_mapping_v4_8_1,
)


def test_v4_8_1_changes_no_physical_candidate_rule():
    base = {"enabled": True}
    assert runtime_safety_mapping_v4_8_1(base) == runtime_safety_mapping_v4_8(base)


def test_v4_8_1_is_pillar_stable_sector_only():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_1_pillar({"enabled": True})
    )
    assert config.zero_feasible_trigger_replans == 10
    assert config.selected_release_replans == 5
    assert config.require_consistent_selected_horizontal_sector is True
    assert config.max_scan_angle_deg == 60.0
    assert config.scan_angle_step_deg == 30.0
    assert config.max_escalated_scan_angle_deg == 120.0
    assert config.contract()["version"] == (
        "deadlock_recovery_v4_1_stable_sector_handoff"
    )
    assert RUNTIME_BEHAVIOR_VERSION == (
        "v4_8_1_pillar_stable_sector_five_frame_handoff_v1"
    )


def test_v4_8_1_reuses_v4_8_yaw_continuity():
    yaw, yaw_rate = calculate_recovery_continuity_yaw_v4_8_1(
        velocity_world=np.asarray([0.0, 2.0, 0.0]),
        goal_direction_world=np.asarray([10.0, 0.0, 0.0]),
        last_yaw=1.4,
        dt=0.1,
    )
    assert yaw > 1.3
    assert yaw_rate > 0.0
