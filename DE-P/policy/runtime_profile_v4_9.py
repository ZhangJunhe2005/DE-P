"""V4.9 motion-preserving dynamic safety over the frozen V4.8.5 runtime.

The learned V4.8.3 static policy, V4.8.5 recovery parameters and every hard
static collision rule remain unchanged.  This profile only adds a bounded
continuous moving-obstacle ranking term and an ordered 1.0/1.2/1.4 temporal
fallback for candidates blocked solely by causal dynamic prediction.
"""

from __future__ import annotations

from policy.runtime_profile_v4_8_5 import (
    calculate_recovery_continuity_yaw_v4_8_5,
    deadlock_recovery_mapping_v4_8_5,
    runtime_safety_mapping_v4_8_5,
)


PROFILE_NAME = "v4_9_dynamic_motion_preserving_safety"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_9_bounded_risk_ordered_retiming_fresh_prefix_v1"
)


def runtime_safety_mapping_v4_9(base):
    value = runtime_safety_mapping_v4_8_5(base)
    value.update({
        # A bounded ranking term among candidates that already passed every
        # hard physical check.  It cannot authorize predicted intersection.
        "dynamic_risk_ranking_enabled": True,
        "dynamic_risk_score_weight": 0.20,
        "dynamic_risk_score_cap": 1.0,
        "dynamic_risk_distance_scale_m": 0.75,
        "dynamic_risk_ttc_scale_s": 1.50,
        "dynamic_risk_closing_speed_scale_mps": 2.0,
        # Ordered global search.  1.4 is never considered when 1.2 already
        # contains an executable action.
        "dynamic_time_retiming_enabled": True,
        "dynamic_time_retiming_scales": (1.0, 1.2, 1.4),
        # A trajectory may be longer than the trustworthy 1.5 s constant-
        # velocity actor prediction.  It is executable only as a rolling MPC
        # prefix: fresh depth/perception must replace the command well before
        # that prefix expires.  Sensor/planner stalls therefore fail closed.
        "dynamic_command_freshness_watchdog_enabled": True,
        "dynamic_command_max_age_s": 0.20,
    })
    return value


def deadlock_recovery_mapping_v4_9(base):
    """Keep the frozen V4.8.5 recovery state-machine contract unchanged."""
    return deadlock_recovery_mapping_v4_8_5(base)


def calculate_recovery_continuity_yaw_v4_9(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    return calculate_recovery_continuity_yaw_v4_8_5(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_9",
    "deadlock_recovery_mapping_v4_9",
    "runtime_safety_mapping_v4_9",
]
