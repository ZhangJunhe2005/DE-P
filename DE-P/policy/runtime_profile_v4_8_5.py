"""Motion-verified recovery handoff over the frozen V4.8.3 policy.

This profile changes no network weight, candidate geometry, static/dynamic
collision rule, score or yaw controller.  It only prevents a bounded scan from
being declared successful until the provisional network handoff produces
measured displacement or a sustained improvement in forward visibility.
"""

from __future__ import annotations

from policy.runtime_profile_v4_8_2 import (
    calculate_recovery_continuity_yaw_v4_8_2,
    deadlock_recovery_mapping_v4_8_2,
    runtime_safety_mapping_v4_8_2,
)


PROFILE_NAME = "v4_8_5_motion_verified_recovery_handoff"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_8_5_universal_motion_verified_recovery_handoff_v1"
)


def runtime_safety_mapping_v4_8_5(base):
    """Keep the frozen V4.8.2/V4.8.3 candidate safety rules unchanged."""
    return runtime_safety_mapping_v4_8_2(base)


def deadlock_recovery_mapping_v4_8_5(base):
    """Require a real escape before resetting the 60/90/120 scan chain."""
    value = deadlock_recovery_mapping_v4_8_2(base)
    value.update({
        "provisional_handoff_validation_enabled": True,
        "handoff_validation_window_s": 1.20,
        "handoff_min_displacement_m": 0.50,
        "handoff_min_forward_clearance_gain_m": 0.50,
        "handoff_forward_clearance_confirmation_replans": 5,
    })
    return value


def calculate_recovery_continuity_yaw_v4_8_5(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    """Reuse V4.8.2 selected-path yaw continuity without semantic changes."""
    return calculate_recovery_continuity_yaw_v4_8_2(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_8_5",
    "deadlock_recovery_mapping_v4_8_5",
    "runtime_safety_mapping_v4_8_5",
]
