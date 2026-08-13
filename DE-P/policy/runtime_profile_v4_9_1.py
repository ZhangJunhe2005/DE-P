"""V4.9.1 temporary-subgoal recovery over the frozen V4.9 runtime."""

from __future__ import annotations

from policy.runtime_profile_v4_9 import (
    calculate_recovery_continuity_yaw_v4_9,
    deadlock_recovery_mapping_v4_9,
    runtime_safety_mapping_v4_9,
)


PROFILE_NAME = "v4_9_1_recovery_temporary_subgoal"
RUNTIME_BEHAVIOR_VERSION = "v4_9_1_full_horizon_temporary_subgoal_handoff_v3"


def runtime_safety_mapping_v4_9_1(base):
    return runtime_safety_mapping_v4_9(base)


def deadlock_recovery_mapping_v4_9_1(base):
    value = deadlock_recovery_mapping_v4_9(base)
    value.update({
        # A temporary goal is a directional contract.  Give the network time
        # to retime inherited braking state, but never accept sideways or
        # backward displacement as proof that the escape succeeded.
        "handoff_validation_window_s": 2.50,
        "handoff_directional_progress_enabled": True,
        "handoff_max_directional_retreat_m": 0.10,
    })
    return value


def calculate_recovery_continuity_yaw_v4_9_1(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    return calculate_recovery_continuity_yaw_v4_9(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_9_1",
    "deadlock_recovery_mapping_v4_9_1",
    "runtime_safety_mapping_v4_9_1",
]
