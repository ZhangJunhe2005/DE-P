"""Pillar-only stable-sector handoff layered on the V4.8 runtime profile.

V4.8.1 changes no trajectory, collision, dynamic-track or score-selection
rule.  It only makes the emergency bounded scan less eager to hand translation
back to the network when isolated safe candidates flicker across unrelated
horizontal columns of the 3 x 5 YOPO lattice.
"""

from __future__ import annotations

from policy.runtime_profile_v4_8 import (
    calculate_recovery_continuity_yaw_v4_8,
    deadlock_recovery_mapping_v4_8_pillar,
    runtime_safety_mapping_v4_8,
)


PROFILE_NAME = "v4_8_1_stable_sector_handoff"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_8_1_pillar_stable_sector_five_frame_handoff_v1"
)


def runtime_safety_mapping_v4_8_1(base):
    """Keep the already validated V4.8 physical candidate contract exactly."""
    return runtime_safety_mapping_v4_8(base)


def deadlock_recovery_mapping_v4_8_1_pillar(base):
    """Require a stable opening and react earlier to a true zero-candidate run."""
    value = deadlock_recovery_mapping_v4_8_pillar(base)
    value.update({
        # Recent successful Pillar runs contain at most seven consecutive
        # zero-candidate replans; the problematic run reaches fifteen.  Ten is
        # therefore an earlier takeover without touching ordinary motion.
        "zero_feasible_trigger_replans": 10,
        # Confirmation is still short (about 0.15 s at 33 Hz), but all five
        # observations must belong to one horizontal lattice column.  Vertical
        # alternatives within that column remain interchangeable.
        "selected_release_replans": 5,
        "require_consistent_selected_horizontal_sector": True,
    })
    return value


def calculate_recovery_continuity_yaw_v4_8_1(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    """Reuse V4.8 selected-path yaw continuity without semantic changes."""
    return calculate_recovery_continuity_yaw_v4_8(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_8_1",
    "deadlock_recovery_mapping_v4_8_1_pillar",
    "runtime_safety_mapping_v4_8_1",
]
