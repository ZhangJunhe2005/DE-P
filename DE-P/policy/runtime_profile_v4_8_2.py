"""Scene-agnostic bounded recovery for measured navigation stagnation.

V4.8.2 keeps the V4.8.1 candidate, safety, yaw and stable-sector handoff
contracts.  Its only additional trigger is based on odometry: if the vehicle
remains inside the same 20 cm displacement ball for a sustained observation
window, the same bounded 60/90/120 degree scan used for a zero-candidate run
is allowed to take over.  No map name or maze type is part of this contract.
"""

from __future__ import annotations

from policy.runtime_profile_v4_8_1 import (
    calculate_recovery_continuity_yaw_v4_8_1,
    deadlock_recovery_mapping_v4_8_1_pillar,
    runtime_safety_mapping_v4_8_1,
)


PROFILE_NAME = "v4_8_2_universal_stagnation_recovery"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_8_2_universal_odometry_stagnation_stable_sector_handoff_v1"
)


def runtime_safety_mapping_v4_8_2(base):
    """Keep the validated V4.8/V4.8.1 physical candidate rules exactly."""
    return runtime_safety_mapping_v4_8_1(base)


def deadlock_recovery_mapping_v4_8_2(base):
    """Apply one recovery contract to every supported map type."""
    value = deadlock_recovery_mapping_v4_8_1_pillar(base)
    value.update({
        "motion_stagnation_enabled": True,
        # Calibration against the latest fixed four-map runs:
        # forest had no 2 s window below 20 cm, wall at most 7 consecutive
        # replans, cave at most 23, while the failed run sustained 576.
        "motion_stagnation_window_s": 2.0,
        "motion_stagnation_min_displacement_m": 0.20,
        "motion_stagnation_trigger_replans": 30,
    })
    return value


def calculate_recovery_continuity_yaw_v4_8_2(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    """Reuse the selected-path yaw continuity without semantic changes."""
    return calculate_recovery_continuity_yaw_v4_8_1(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_8_2",
    "deadlock_recovery_mapping_v4_8_2",
    "runtime_safety_mapping_v4_8_2",
]
