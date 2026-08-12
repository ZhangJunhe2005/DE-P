"""V4.5.10 minimal runtime shield with a representation-aware body floor."""

from __future__ import annotations

from policy.runtime_profile_v4_5_7 import (
    calculate_original_yopo_yaw_v4_5_7,
    deadlock_recovery_mapping_v4_5,
    runtime_safety_mapping_v4_5_7,
)


PROFILE_NAME = "v4_5_10_tail_aware_retimed"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_5_10_0_35m_representation_floor_candidate_retiming_v1"
)


def runtime_safety_mapping_v4_5_10(base):
    """Keep V4.5.7 selection/retiming and add only 5 cm floor margin."""
    value = runtime_safety_mapping_v4_5_7(base)
    value["collision_representation_margin_m"] = 0.05
    return value


def calculate_original_yopo_yaw_v4_5_10(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    return calculate_original_yopo_yaw_v4_5_7(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME", "RUNTIME_BEHAVIOR_VERSION",
    "runtime_safety_mapping_v4_5_10",
    "deadlock_recovery_mapping_v4_5",
    "calculate_original_yopo_yaw_v4_5_10",
]
