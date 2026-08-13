import math

import numpy as np

from policy.deadlock_recovery_v3 import DeadlockRecoveryConfigV3
from policy.runtime_profile_v4_7 import runtime_safety_mapping_v4_7
from policy.runtime_profile_v4_8 import (
    calculate_recovery_continuity_yaw_v4_8,
    deadlock_recovery_mapping_v4_8_pillar,
    runtime_safety_mapping_v4_8,
)


def test_v4_8_does_not_change_v4_7_physical_candidate_contract():
    base = {"enabled": True}
    assert runtime_safety_mapping_v4_8(base) == runtime_safety_mapping_v4_7(base)


def test_v4_8_yaw_follows_selected_lateral_path_not_blocked_goal():
    yaw, yaw_rate = calculate_recovery_continuity_yaw_v4_8(
        velocity_world=np.array([0.0, 2.0, 0.0]),
        goal_direction_world=np.array([10.0, 0.0, 0.0]),
        last_yaw=math.pi / 2.0,
        dt=0.1,
    )
    # The selected +Y path remains camera-forward; goal bias may only make a
    # small correction and cannot snap the camera back toward +X.
    assert yaw > math.radians(80.0)
    assert abs(yaw_rate) <= 0.5 * math.pi + 1.0e-9


def test_v4_8_low_speed_holds_observed_heading():
    yaw, yaw_rate = calculate_recovery_continuity_yaw_v4_8(
        velocity_world=np.array([0.05, 0.0, 0.0]),
        goal_direction_world=np.array([10.0, 0.0, 0.0]),
        last_yaw=1.2,
        dt=0.1,
    )
    assert yaw == 1.2
    assert yaw_rate == 0.0


def test_v4_8_recovery_mapping_is_evidence_preserving_not_a_new_gate():
    config = DeadlockRecoveryConfigV3.from_mapping(
        deadlock_recovery_mapping_v4_8_pillar({"enabled": True})
    )
    assert config.cooldown_s == 0.60
    assert config.accumulate_zero_during_cooldown is True
    assert config.defer_scan_reset_until_post_release_confirmation is True
    assert config.post_release_confirmation_replans == 15
    assert config.contract()["version"] == (
        "deadlock_recovery_v4_evidence_preserving_handoff"
    )
