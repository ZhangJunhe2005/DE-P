"""Shape-specific occupancy unions over bounded center sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounded_dynamic_reachability_v1 import DynamicReachabilityStateV1


OCCUPANCY_VERSION = "shape_reachable_occupancy_v1"


@dataclass(frozen=True)
class PredictedReachableOccupancyV1:
    state: DynamicReachabilityStateV1
    times_s: np.ndarray
    effective_horizons_s: np.ndarray
    center_lower_world: np.ndarray
    center_upper_world: np.ndarray
    radial_growth_m: np.ndarray
    vertical_growth_m: np.ndarray
    geometry_age_s: float
    stale_age_included: bool
    double_age_guard: bool


def predict_reachable_occupancy(state, horizon):
    times = horizon.candidate_relative_times_s
    effective = horizon.effective_prediction_horizons_s
    lower_v, upper_v = (
        state.velocity_set_world_mps.lower,
        state.velocity_set_world_mps.upper,
    )
    lower_a, upper_a = (
        state.acceleration_set_world_mps2.lower,
        state.acceleration_set_world_mps2.upper,
    )
    lower = (
        state.position_set_world.lower[None, :]
        + effective[:, None]*lower_v[None, :]
        + .5*effective[:, None]**2*lower_a[None, :]
    )
    upper = (
        state.position_set_world.upper[None, :]
        + effective[:, None]*upper_v[None, :]
        + .5*effective[:, None]**2*upper_a[None, :]
    )
    drift = state.reference_drift_rate_mps*effective
    lower -= drift[:, None]
    upper += drift[:, None]
    return PredictedReachableOccupancyV1(
        state, times.copy(), effective.copy(), lower, upper,
        drift.copy(), drift.copy(), horizon.geometry_age_s,
        horizon.stale_age_included, horizon.double_age_guard,
    )


def _distance_to_interval(value, lower, upper):
    outside = np.maximum(np.maximum(lower-value, value-upper), 0.)
    inside = np.minimum(np.maximum(lower-value, value-upper), 0.)
    return outside, inside


def reachable_signed_distance(points, occupancy):
    points = np.asarray(points, dtype=np.float64)
    lower, upper = (
        occupancy.center_lower_world, occupancy.center_upper_world
    )
    state = occupancy.state
    if state.shape_type == "sphere":
        outside = np.maximum(np.maximum(lower-points, points-upper), 0.)
        center_distance = np.linalg.norm(outside, axis=1)
        inside = np.all((points >= lower) & (points <= upper), axis=1)
        center_distance[inside] = 0.
        return center_distance-float(state.radius_interval_m[1])
    if state.shape_type != "vertical_cylinder":
        raise ValueError("unsupported reachable shape")
    horizontal_outside = np.maximum(
        np.maximum(lower[:, :2]-points[:, :2], points[:, :2]-upper[:, :2]),
        0.,
    )
    radial = np.linalg.norm(horizontal_outside, axis=1)
    half = float(state.half_height_interval_m[1])
    vertical_lower = lower[:, 2]-half
    vertical_upper = upper[:, 2]+half
    vertical_outside, _ = _distance_to_interval(
        points[:, 2], vertical_lower, vertical_upper
    )
    q_radial = radial-float(state.radius_interval_m[1])
    q_vertical = vertical_outside
    outside = np.sqrt(
        np.maximum(q_radial, 0.)**2
        + np.maximum(q_vertical, 0.)**2
    )
    inside = np.minimum(np.maximum(q_radial, q_vertical), 0.)
    return outside+inside


__all__ = [
    "OCCUPANCY_VERSION", "PredictedReachableOccupancyV1",
    "predict_reachable_occupancy", "reachable_signed_distance",
]

