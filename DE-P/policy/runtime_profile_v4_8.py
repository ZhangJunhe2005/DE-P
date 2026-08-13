"""V4.8 recovery-to-network continuity profile.

This profile changes no trajectory feasibility rule and adds no selection
Gate.  It keeps the V4.7 safety mapping, but prevents the forward camera from
snapping back to the blocked global-goal bearing immediately after a bounded
scan has exposed a valid lateral route.
"""

from __future__ import annotations

import numpy as np

from policy.deadlock_recovery_v3 import (
    deadlock_recovery_mapping_v4_7_pillar,
)
from policy.poly_solver import calculate_path_tangent_yaw_v1
from policy.runtime_profile_v4_7 import runtime_safety_mapping_v4_7


PROFILE_NAME = "v4_8_recovery_continuity"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_8_selected_path_yaw_cooldown_evidence_continuity_v1"
)


def runtime_safety_mapping_v4_8(base):
    """Preserve V4.7 physical checks and bounded clearance preference."""
    return runtime_safety_mapping_v4_7(base)


def deadlock_recovery_mapping_v4_8_pillar(base):
    """Keep scan-only recovery while preserving evidence across handoff."""
    value = deadlock_recovery_mapping_v4_7_pillar(base)
    value.update({
        # The scan heading commitment is 0.45 s.  A 0.60 s takeover cooldown
        # avoids immediate mode chatter without imposing V4.7's two seconds of
        # blind braking when the handed-off candidate disappears.
        "cooldown_s": 0.60,
        "accumulate_zero_during_cooldown": True,
        "defer_scan_reset_until_post_release_confirmation": True,
        # Require about 0.45 s at the 33 Hz planning rate before treating a
        # provisional opening as a completed escape and resetting scan width.
        "post_release_confirmation_replans": 15,
    })
    return value


def calculate_recovery_continuity_yaw_v4_8(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    """Point the forward camera along the selected trajectory, not the wall.

    At meaningful speed the selected path tangent owns 90% of the direction
    and the global goal remains a gentle 10% bias.  At low speed we retain the
    last observed heading instead of snapping back to the global goal.  New
    goals are still aligned by the existing pre-planning goal-alignment state,
    and bounded scan recovery remains authoritative while it is active.
    """
    velocity = np.asarray(velocity_world, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(velocity[:2])) < 0.20:
        return float(last_yaw), 0.0
    return calculate_path_tangent_yaw_v1(
        velocity,
        goal_direction_world,
        last_yaw,
        dt,
        max_yaw_rate=0.5,
        tangent_weight=0.90,
        minimum_tracking_speed=0.20,
    )


__all__ = [
    "PROFILE_NAME",
    "RUNTIME_BEHAVIOR_VERSION",
    "calculate_recovery_continuity_yaw_v4_8",
    "deadlock_recovery_mapping_v4_8_pillar",
    "runtime_safety_mapping_v4_8",
]
