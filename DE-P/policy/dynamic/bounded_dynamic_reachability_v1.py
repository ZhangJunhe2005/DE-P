"""Per-track, per-generation, per-shape bounded reachable states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .shape_aware_motion_state_v1 import MotionObservability
from .shape_motion_hypothesis_tracker_v1 import TrackedShapeMotionV1
from .stale_geometry_time_contract_v1 import (
    EffectiveHorizonV1, resolve_effective_horizon,
)


REACHABILITY_VERSION = "bounded_dynamic_reachability_v1"


class ReachabilityStatus(str, Enum):
    DIRECT_GEOMETRY = "DIRECT_GEOMETRY"
    CURRENT_MOTION_STATE = "CURRENT_MOTION_STATE"
    STALE_BOUNDED_REACHABILITY = "STALE_BOUNDED_REACHABILITY"
    UNRESOLVED_DYNAMIC_RISK = "UNRESOLVED_DYNAMIC_RISK"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class IntervalSetV1:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if (
            lower.shape != upper.shape or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)
        ):
            raise ValueError("invalid interval set")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())


@dataclass(frozen=True)
class DynamicReachabilityStateV1:
    track_id: int
    generation: str
    hypothesis_id: str
    shape_type: str
    source_geometry_timestamp: float
    state_timestamp: float
    query_timestamp: float
    geometry_age_s: float
    motion_observability: MotionObservability
    position_set_world: IntervalSetV1
    velocity_set_world_mps: IntervalSetV1
    acceleration_set_world_mps2: IntervalSetV1
    radius_interval_m: Tuple[float, float]
    half_height_interval_m: Tuple[float, float]
    center_z_interval_m: Tuple[float, float]
    reference_drift_rate_mps: float
    status: ReachabilityStatus
    expiry_timestamp: float
    position_reference_timestamp: float
    runtime_gt_used: bool = False


class BoundedDynamicReachabilityBuilderV1:
    version = REACHABILITY_VERSION

    def __init__(self, config, candidate):
        self.config = dict(config)
        self.candidate = dict(candidate)
        self._keys = set()

    def clear(self):
        self._keys.clear()

    def delete_missing(self, live_track_ids):
        live = {int(value) for value in live_track_ids}
        removed = {key for key in self._keys if key[0] not in live}
        self._keys -= removed
        return tuple(sorted(removed))

    def build(self, tracked: TrackedShapeMotionV1):
        if not isinstance(tracked, TrackedShapeMotionV1):
            raise TypeError("tracked must be TrackedShapeMotionV1")
        output = []
        maximum_age = float(self.config["time_origin"]["maximum_stale_age_s"])
        for index, item in enumerate(tracked.hypotheses):
            geometry = item.geometry_hypothesis
            motion = item.motion
            geometry_timestamp = float(motion.last_direct_timestamp)
            query_timestamp = float(tracked.timestamp)
            age = query_timestamp-geometry_timestamp
            key = (
                int(tracked.track_id), str(tracked.generation),
                f"{item.geometry_type}:{index}",
            )
            self._keys.add(key)
            if age > maximum_age+1e-9:
                status = ReachabilityStatus.EXPIRED
            elif tracked.prediction_only or age > 1e-9:
                status = ReachabilityStatus.STALE_BOUNDED_REACHABILITY
            else:
                status = ReachabilityStatus.DIRECT_GEOMETRY
            point_velocity = motion.velocity_world
            if self.candidate["use_velocity_set"]:
                configured = (
                    self.config["velocity_set"][
                        "sphere_minimum_half_width_mps"
                    ] if item.geometry_type == "sphere" else
                    self.config["velocity_set"][
                        "cylinder_minimum_half_width_mps"
                    ]
                )
                covariance_half = (
                    float(self.config["velocity_set"]["covariance_sigma_scale"])
                    * np.sqrt(np.maximum(
                        np.diag(motion.velocity_covariance_world), 0.
                    ))
                )
                half = np.maximum(np.asarray(configured), covariance_half)
                half = np.minimum(
                    half,
                    float(self.config["velocity_set"]["maximum_speed_mps"]),
                )
            else:
                half = np.zeros(3)
            velocity = IntervalSetV1(
                np.maximum(
                    point_velocity-half,
                    -float(self.config["velocity_set"]["maximum_speed_mps"]),
                ),
                np.minimum(
                    point_velocity+half,
                    float(self.config["velocity_set"]["maximum_speed_mps"]),
                ),
            )
            acceleration_bound = (
                float(self.config["acceleration_set"]["isotropic_bound_mps2"])
                if self.candidate["use_acceleration_set"] else 0.
            )
            acceleration = IntervalSetV1(
                np.full(3, -acceleration_bound),
                np.full(3, acceleration_bound),
            )
            if self.candidate["use_reference_drift"]:
                weak = motion.observability in {
                    MotionObservability.MOTION_WEAKLY_OBSERVABLE,
                    MotionObservability.MOTION_AMBIGUOUS,
                    MotionObservability.SUPPORT_PROPAGATION_ONLY,
                    MotionObservability.MOTION_INITIALIZING,
                }
                drift = float(self.config["reference_drift"][
                    "weak_or_ambiguous_rate_mps" if weak
                    else "stable_geometric_center_rate_mps"
                ])
            else:
                drift = 0.
            position_sigma = 2*np.sqrt(np.maximum(
                np.diag(geometry.model_covariance_world), 0.
            ))
            position = IntervalSetV1(
                motion.position_world-position_sigma,
                motion.position_world+position_sigma,
            )
            output.append(DynamicReachabilityStateV1(
                key[0], key[1], key[2], item.geometry_type,
                geometry_timestamp, query_timestamp, query_timestamp, age,
                motion.observability, position, velocity, acceleration,
                geometry.radius_interval_m,
                geometry.half_height_interval_m,
                geometry.center_z_interval_m, drift, status,
                geometry_timestamp+maximum_age, geometry_timestamp, False,
            ))
        if not output and tracked.observability != MotionObservability.MOTION_EXPIRED:
            # A once-dynamic track without a bounded hypothesis is explicitly
            # unresolved, never equivalent to no active risk.
            return (), ReachabilityStatus.UNRESOLVED_DYNAMIC_RISK
        if not output:
            return (), ReachabilityStatus.EXPIRED
        return tuple(output), None

    def horizons(self, state, candidate_times):
        return resolve_effective_horizon(
            geometry_timestamp=state.source_geometry_timestamp,
            motion_state_timestamp=state.state_timestamp,
            current_query_timestamp=state.query_timestamp,
            candidate_relative_times_s=candidate_times,
            position_reference_timestamp=state.position_reference_timestamp,
        )


__all__ = [
    "REACHABILITY_VERSION", "ReachabilityStatus", "IntervalSetV1",
    "DynamicReachabilityStateV1", "BoundedDynamicReachabilityBuilderV1",
]

