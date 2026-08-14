"""V4.9.1 temporary-subgoal recovery over the frozen V4.9 runtime."""

from __future__ import annotations

from policy.runtime_profile_v4_9 import (
    calculate_recovery_continuity_yaw_v4_9,
    deadlock_recovery_mapping_v4_9,
    runtime_safety_mapping_v4_9,
)


PROFILE_NAME = "v4_9_1_recovery_temporary_subgoal"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_9_1_long_escape_slow_handoff_rearmable_lifecycle_v6"
)


def runtime_safety_mapping_v4_9_1(base):
    return runtime_safety_mapping_v4_9(base)


def deadlock_recovery_mapping_v4_9_1(base):
    value = deadlock_recovery_mapping_v4_9(base)
    value.update({
        # A temporary goal is a directional contract.  Give the network time
        # to retime inherited braking state.  The runtime trace that motivated
        # V5 moved 0.191 m in 2.55 s while increasing forward clearance by
        # 2.54 m; the former 0.50 m / 2.50 s threshold incorrectly cancelled
        # that valid low-speed takeover.  A small measured directional start
        # is sufficient because the fixed long target and the ordinary
        # per-frame static/dynamic shield remain authoritative afterwards.
        "handoff_validation_window_s": 4.00,
        "handoff_min_displacement_m": 0.15,
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
