"""Minimal physical runtime profile and original-YOPO yaw for V4.4."""

from __future__ import annotations

import numpy as np


PROFILE_NAME = "v4_4_static_parity"
RUNTIME_BEHAVIOR_VERSION = (
    "v4_4_original_yaw_physical_envelope_boundary_escape_v4"
)


def runtime_safety_mapping_v4_4(base):
    """Keep one physical envelope while removing heuristic selection Gates.

    The observed tracking clearance and its velocity-dependent stopping term
    are two parts of the same vehicle-safety contract.  Camera visibility,
    minimum progress, goal progress, and feasibility projection remain
    disabled so this profile does not reintroduce the former Gate stack.
    """
    value = dict(base)
    value.update({
        "feasibility_projection_enabled": False,
        "enforce_camera_visibility": False,
        "enforce_minimum_progress": False,
        "enforce_goal_progress": False,
        "enforce_observed_tracking_clearance": True,
        "enforce_stopping_distance": True,
    })
    return value


def deadlock_recovery_mapping_v4_4(base):
    """Return a brake-and-observe policy that never commands translation.

    The scan starts only after many consecutive replans without a viable
    candidate.  One goal-progressing candidate releases control immediately
    back to YOPO.  Breadcrumb retreat and camera-forward escape are disabled.
    """
    value = dict(base)
    value.update({
        # Fifteen consecutive empty replans is still an emergency-only event,
        # but it is short enough to react before an outward-facing vehicle
        # spends several seconds braking at the canonical map boundary.
        "zero_feasible_trigger_replans": 15,
        "collision_floor_trigger_replans": 5,
        "release_feasible_replans": 1,
        "scan_release_feasible_replans": 1,
        "stationary_speed_mps": 0.25,
        # Original YOPO limits normal yaw by 0.5*pi rad/s = 90 deg/s.
        # The observation-only recovery scan uses the same physical rate.
        "yaw_scan_rate_deg_s": 90.0,
        "max_scan_angle_deg": 120.0,
        "max_scan_legs": 2,
        "retreat_enabled": False,
        "observed_escape_enabled": False,
    })
    return value


def calculate_original_yopo_yaw_v4_4(
    velocity_world, goal_direction_world, last_yaw, dt,
    max_yaw_rate_half_turns_per_s=0.5,
):
    """Reproduce the yaw update used by the original YOPO ROS node.

    The historical ``max_yaw_rate`` value is expressed in half-turns per
    second and multiplied by pi in the implementation.  Its default 0.5 is
    therefore 90 degrees/s, not 0.5 radians/s.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("yaw control dt must be finite and positive")
    velocity = np.asarray(velocity_world, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_direction_world, dtype=np.float64).reshape(3)
    if not np.isfinite(velocity).all() or not np.isfinite(goal).all():
        raise ValueError("yaw inputs contain NaN/Inf")

    velocity_direction = velocity / (np.linalg.norm(velocity) + 1.0e-5)
    goal_distance = float(np.linalg.norm(goal))
    goal_direction = goal / (goal_distance + 1.0e-5)
    goal_yaw = float(np.arctan2(goal_direction[1], goal_direction[0]))
    goal_delta = (goal_yaw - float(last_yaw) + np.pi) % (2.0 * np.pi) - np.pi
    goal_weight = 6.0 * abs(goal_delta) / np.pi
    desired_direction = velocity_direction + goal_weight * goal_direction
    desired_yaw = (
        float(np.arctan2(desired_direction[1], desired_direction[0]))
        if goal_distance > 0.5 else float(last_yaw)
    )
    yaw_delta = (
        desired_yaw - float(last_yaw) + np.pi
    ) % (2.0 * np.pi) - np.pi
    maximum_change = (
        float(max_yaw_rate_half_turns_per_s) * np.pi * dt
    )
    applied_change = float(np.clip(
        yaw_delta, -maximum_change, maximum_change
    ))
    yaw = (float(last_yaw) + applied_change + np.pi) % (2.0 * np.pi) - np.pi
    return yaw, applied_change / dt
