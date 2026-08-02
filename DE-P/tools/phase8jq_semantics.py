"""Pure audit helpers for Phase 8J-Q safety and time semantics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


UAV_RADIUS_M = 0.30


def sphere_clearance(uav_position, actor_position, uav_radius, actor_radius):
    return float(
        np.linalg.norm(np.asarray(uav_position) - np.asarray(actor_position))
        - float(uav_radius) - float(actor_radius)
    )


def finite_vertical_cylinder_clearance(
    uav_position, actor_position, uav_radius, actor_radius, actor_height
):
    """Match Simulator's sphere-versus-finite-vertical-cylinder geometry."""
    delta = np.asarray(uav_position, dtype=float) - np.asarray(
        actor_position, dtype=float
    )
    radial_gap = max(0.0, math.hypot(delta[0], delta[1]) - actor_radius)
    vertical_gap = max(0.0, abs(delta[2]) - actor_height / 2.0)
    return math.hypot(radial_gap, vertical_gap) - uav_radius


def simulator_dynamic_clearance(uav_position, actor):
    shape = str(actor.get("type", actor.get("shape", "sphere"))).lower()
    if shape in {"cylinder", "vertical_cylinder"}:
        return finite_vertical_cylinder_clearance(
            uav_position,
            actor["position_world"],
            UAV_RADIUS_M,
            float(actor["radius"]),
            float(actor.get("height", 2.0 * float(actor["radius"]))),
        )
    return sphere_clearance(
        uav_position, actor["position_world"], UAV_RADIUS_M, actor["radius"]
    )


def formal_dynamic_clearance(
    uav_position,
    actor,
    delta_time,
    covariance_sigma=2.0,
    covariance_growth_rate=0.05,
):
    covariance = np.asarray(
        actor.get("position_covariance", np.zeros((3, 3))), dtype=float
    )
    eigenvalue = float(np.linalg.eigvalsh(covariance).max())
    uncertainty = covariance_sigma * math.sqrt(
        max(0.0, eigenvalue + covariance_growth_rate * delta_time**2)
    )
    return sphere_clearance(
        uav_position,
        actor["position_world"],
        UAV_RADIUS_M,
        float(actor["radius"]) + uncertainty,
    )


def point_map_clearance(nearest_surface_distance, uav_radius=UAV_RADIUS_M):
    return float(nearest_surface_distance) - float(uav_radius)


def sample_times(horizon, points, include_t0=False):
    values = np.linspace(
        float(horizon) / int(points), float(horizon), int(points)
    )
    return np.concatenate(([0.0], values)) if include_t0 else values


def first_negative_time(times, clearances):
    for time_value, clearance in zip(times, clearances):
        if clearance < 0.0:
            return float(time_value)
    return None


def classify_timeline(times, clearances, first_controllable_time):
    times = np.asarray(times, dtype=float)
    clearances = np.asarray(clearances, dtype=float)
    if times.shape != clearances.shape or times.size == 0:
        raise ValueError("times and clearances must be non-empty and aligned")
    t0 = float(clearances[0])
    minimum_index = int(np.argmin(clearances))
    first_negative = first_negative_time(times, clearances)
    if t0 < 0:
        if minimum_index == 0 and np.all(clearances[1:] >= t0 - 1e-9):
            category = "minimum_at_t0_only"
        elif np.any(clearances[1:] < t0 - 1e-6):
            category = "initially_unsafe_deepening"
        else:
            category = "initially_unsafe_recoverable_candidate"
    elif first_negative is not None and first_negative <= first_controllable_time:
        category = "collision_before_first_controllable_action"
    elif first_negative is not None:
        category = "initially_safe_future_collision"
    else:
        category = "minimum_during_latency" if (
            times[minimum_index] <= first_controllable_time
        ) else "minimum_late_horizon"
    return {
        "category": category,
        "t0_clearance_m": t0,
        "minimum_clearance_m": float(clearances[minimum_index]),
        "minimum_time_s": float(times[minimum_index]),
        "minimum_time_index": minimum_index,
        "first_negative_time_s": first_negative,
    }


def continuous_linear_closest_approach(
    uav_start, uav_end, actor_start, actor_end
):
    relative_start = np.asarray(uav_start, dtype=float) - np.asarray(
        actor_start, dtype=float
    )
    relative_delta = (
        np.asarray(uav_end, dtype=float) - np.asarray(uav_start, dtype=float)
        - np.asarray(actor_end, dtype=float) + np.asarray(actor_start, dtype=float)
    )
    denominator = float(relative_delta @ relative_delta)
    alpha = 0.0 if denominator <= 1e-15 else float(
        np.clip(-(relative_start @ relative_delta) / denominator, 0.0, 1.0)
    )
    relative = relative_start + alpha * relative_delta
    return float(np.linalg.norm(relative)), alpha


def stopping_distance(speed, deceleration, latency=0.0):
    if deceleration <= 0:
        raise ValueError("deceleration must be positive")
    speed = max(0.0, float(speed))
    return speed * max(0.0, float(latency)) + speed**2 / (2.0 * deceleration)


def integrate_two_phase_acceleration(
    position, velocity, acceleration_first, acceleration_second, horizon, samples
):
    """Two equal constant-acceleration controls; includes t=0."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    first = np.asarray(acceleration_first, dtype=float)
    second = np.asarray(acceleration_second, dtype=float)
    times = np.linspace(0.0, float(horizon), int(samples) + 1)
    split = float(horizon) / 2.0
    positions, velocities = [], []
    midpoint_position = position + velocity * split + 0.5 * first * split**2
    midpoint_velocity = velocity + first * split
    for value in times:
        if value <= split:
            positions.append(position + velocity * value + 0.5 * first * value**2)
            velocities.append(velocity + first * value)
        else:
            local = value - split
            positions.append(
                midpoint_position + midpoint_velocity * local
                + 0.5 * second * local**2
            )
            velocities.append(midpoint_velocity + second * local)
    return times, np.asarray(positions), np.asarray(velocities)


def max_norm(rows: Iterable[Iterable[float]]):
    values = np.asarray(list(rows), dtype=float)
    return float(np.linalg.norm(values, axis=-1).max()) if values.size else 0.0
