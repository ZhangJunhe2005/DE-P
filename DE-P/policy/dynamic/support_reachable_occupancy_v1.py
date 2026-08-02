"""Causal support occupancy with bounded advection and reference drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUPPORT_VERSION = "support_reachable_occupancy_v1"


@dataclass(frozen=True)
class SupportReachableOccupancyV1:
    support_min_world: np.ndarray
    support_max_world: np.ndarray
    visible_support_velocity_world: np.ndarray
    reference_drift_rate_mps: float
    acceleration_bound_mps2: float
    timestamp: float
    velocity_semantics: str = "visible_support_not_actor_center"
    runtime_gt_used: bool = False

    def bounds_at(self, times_s):
        times = np.asarray(times_s, dtype=np.float64)
        drift = (
            self.reference_drift_rate_mps*times
            + .5*self.acceleration_bound_mps2*times**2
        )
        displacement = (
            times[:, None]*self.visible_support_velocity_world[None, :]
        )
        return (
            self.support_min_world[None, :]+displacement-drift[:, None],
            self.support_max_world[None, :]+displacement+drift[:, None],
        )


def point_aabb_signed_distance(points, lower, upper):
    points = np.asarray(points, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    outside = np.maximum(np.maximum(lower-points, points-upper), 0.)
    outside_distance = np.linalg.norm(outside, axis=-1)
    inside_distance = np.minimum(
        np.max(np.maximum(lower-points, points-upper), axis=-1), 0.
    )
    return outside_distance+inside_distance


__all__ = [
    "SUPPORT_VERSION", "SupportReachableOccupancyV1",
    "point_aabb_signed_distance",
]

