"""V4.5.7 minimal collision shield with limit-compliant time retiming."""

from __future__ import annotations

from policy.runtime_profile_v4_4 import calculate_original_yopo_yaw_v4_4
from policy.runtime_profile_v4_5 import deadlock_recovery_mapping_v4_5


PROFILE_NAME = "v4_5_7_high_safety_retimed"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_5_7_collision_floor_dynamic_filter_candidate_retiming_v1"
)


def runtime_safety_mapping_v4_5_7(base):
    """Retain only physical collision vetoes and retime hardware excess.

    Speed and acceleration excess are not reasons to discard an otherwise
    useful candidate: its duration is increased and the rebuilt quintic is
    checked again.  Actual static/dynamic collision, non-finite state and the
    physical map boundary remain authoritative vetoes.  Former FOV/progress
    heuristics and the wider 0.65 m preference do not become new Gates.
    """
    value = dict(base)
    value.update({
        "feasibility_projection_enabled": False,
        "candidate_retiming_enabled": True,
        "candidate_retiming_max_scale": 3.0,
        "candidate_retiming_steps": 17,
        "candidate_retiming_margin": 1.02,
        "enforce_camera_visibility": False,
        "enforce_minimum_progress": False,
        "enforce_goal_progress": False,
        "enforce_observed_tracking_clearance": False,
        "enforce_stopping_distance": False,
        "enforce_preferred_flight_volume_clearance": False,
        "prefer_largest_boundary_escape_clearance": False,
    })
    return value


def calculate_original_yopo_yaw_v4_5_7(
    velocity_world, goal_direction_world, last_yaw, dt,
):
    return calculate_original_yopo_yaw_v4_4(
        velocity_world, goal_direction_world, last_yaw, dt,
    )


__all__ = [
    "PROFILE_NAME", "RUNTIME_BEHAVIOR_VERSION",
    "runtime_safety_mapping_v4_5_7",
    "deadlock_recovery_mapping_v4_5",
    "calculate_original_yopo_yaw_v4_5_7",
]
