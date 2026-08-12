"""Minimal V4.5 runtime profile: hard body floor, soft outer envelope."""

from __future__ import annotations

from policy.runtime_profile_v4_4 import (
    calculate_original_yopo_yaw_v4_4 as calculate_original_yopo_yaw_v4_5,
    deadlock_recovery_mapping_v4_4,
)


PROFILE_NAME = "v4_5_bounded_danger"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_5_original_yaw_physical_boundary_network_score_v1"
)


def runtime_safety_mapping_v4_5(base):
    """Keep physical limits without turning the 0.65 m preference into a Gate."""
    value = dict(base)
    value.update({
        "feasibility_projection_enabled": False,
        "enforce_camera_visibility": False,
        "enforce_minimum_progress": False,
        "enforce_goal_progress": False,
        "enforce_observed_tracking_clearance": True,
        "enforce_stopping_distance": True,
        "enforce_preferred_flight_volume_clearance": False,
        "prefer_largest_boundary_escape_clearance": False,
    })
    return value


def deadlock_recovery_mapping_v4_5(base):
    """Reuse V4.4's rare, observation-only emergency scan unchanged."""
    return deadlock_recovery_mapping_v4_4(base)
