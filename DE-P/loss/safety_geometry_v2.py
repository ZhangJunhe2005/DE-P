"""Versioned physical and planning geometry for Safety Evaluator V2.

This module is intentionally independent from the V1 loss implementations.
It is an offline evaluator and does not participate in network training.
"""

from __future__ import annotations

import math

import numpy as np


def sphere_signed_gap(uav_position, actor_position, uav_radius, actor_radius):
    return float(
        np.linalg.norm(
            np.asarray(uav_position, dtype=float)
            - np.asarray(actor_position, dtype=float)
        )
        - float(uav_radius)
        - float(actor_radius)
    )


def finite_vertical_cylinder_signed_gap(
    uav_position,
    cylinder_position,
    uav_radius,
    cylinder_radius,
    cylinder_height,
):
    """Signed gap used by Simulator actorCollidesWithUav."""
    difference = np.asarray(uav_position, dtype=float) - np.asarray(
        cylinder_position, dtype=float
    )
    radial_gap = max(
        0.0, math.hypot(difference[0], difference[1]) - float(cylinder_radius)
    )
    vertical_gap = max(
        0.0, abs(difference[2]) - 0.5 * float(cylinder_height)
    )
    return math.hypot(radial_gap, vertical_gap) - float(uav_radius)


def actor_physical_clearance(uav_position, actor, uav_radius):
    shape = str(actor.get("type", actor.get("shape", "sphere"))).lower()
    if shape == "sphere":
        return sphere_signed_gap(
            uav_position, actor["position_world"], uav_radius, actor["radius"]
        )
    if shape in {"vertical_cylinder", "cylinder"}:
        return finite_vertical_cylinder_signed_gap(
            uav_position,
            actor["position_world"],
            uav_radius,
            actor["radius"],
            actor["height"],
        )
    raise ValueError(f"unsupported actor shape: {shape}")


def static_physical_clearance(raw_esdf_distance, uav_radius):
    return np.asarray(raw_esdf_distance, dtype=float) - float(uav_radius)


def directional_covariance_margin(
    uav_position, actor_position, covariance, confidence_sigma
):
    separation = np.asarray(uav_position, dtype=float) - np.asarray(
        actor_position, dtype=float
    )
    norm = float(np.linalg.norm(separation))
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("position covariance must be finite [3,3]")
    if norm <= 1e-12:
        variance = float(np.linalg.eigvalsh(covariance).max())
    else:
        direction = separation / norm
        variance = float(direction @ covariance @ direction)
    return float(confidence_sigma) * math.sqrt(max(0.0, variance))


def maximum_eigenvalue_margin(covariance, confidence_sigma):
    covariance = np.asarray(covariance, dtype=float)
    return float(confidence_sigma) * math.sqrt(
        max(0.0, float(np.linalg.eigvalsh(covariance).max()))
    )


def planning_clearance(
    physical_clearance,
    *,
    planning_margin=0.0,
    tracking_control_margin=0.0,
    uncertainty_margin=0.0,
):
    return (
        np.asarray(physical_clearance, dtype=float)
        - float(planning_margin)
        - float(tracking_control_margin)
        - np.asarray(uncertainty_margin, dtype=float)
    )


def continuous_sphere_segment_gap(
    uav_start,
    uav_end,
    actor_start,
    actor_end,
    uav_radius,
    actor_radius,
):
    relative_start = np.asarray(uav_start, dtype=float) - np.asarray(
        actor_start, dtype=float
    )
    relative_delta = (
        np.asarray(uav_end, dtype=float) - np.asarray(uav_start, dtype=float)
        - np.asarray(actor_end, dtype=float)
        + np.asarray(actor_start, dtype=float)
    )
    denominator = float(relative_delta @ relative_delta)
    fraction = 0.0 if denominator <= 1e-15 else float(
        np.clip(-(relative_start @ relative_delta) / denominator, 0.0, 1.0)
    )
    distance = float(np.linalg.norm(relative_start + fraction * relative_delta))
    return distance - float(uav_radius) - float(actor_radius), fraction
