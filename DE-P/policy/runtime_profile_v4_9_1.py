"""V4.9.1 temporary-subgoal recovery over the frozen V4.9 runtime."""

from __future__ import annotations

from policy.runtime_profile_v4_9 import (
    calculate_recovery_continuity_yaw_v4_9,
    deadlock_recovery_mapping_v4_9,
    runtime_safety_mapping_v4_9,
)


PROFILE_NAME = "v4_9_1_recovery_temporary_subgoal"
RUNTIME_BEHAVIOR_VERSION = "v4_9_1_certified_candidate_temporary_subgoal_v1"


def runtime_safety_mapping_v4_9_1(base):
    return runtime_safety_mapping_v4_9(base)


def deadlock_recovery_mapping_v4_9_1(base):
    return deadlock_recovery_mapping_v4_9(base)


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
