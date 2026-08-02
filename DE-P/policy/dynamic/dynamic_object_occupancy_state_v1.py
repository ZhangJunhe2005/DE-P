"""Shape-preserving shadow occupancy and per-model causal motion state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .dynamic_object_geometry_model_v1 import ShapeHypothesisV1
from .shape_hypothesis_tracker_v1 import TrackedShapeHypothesesV1


STATE_VERSION = "dynamic_object_occupancy_state_v1"


@dataclass(frozen=True)
class PredictedShapeOccupancyV1:
    geometry_type: str
    center_world: np.ndarray
    velocity_world: np.ndarray
    center_z_interval_m: Tuple[float, float]
    radius_interval_m: Tuple[float, float]
    half_height_interval_m: Tuple[float, float]
    model_covariance_world: np.ndarray
    motion_covariance_world: np.ndarray
    velocity_source: str


@dataclass(frozen=True)
class DynamicObjectOccupancyStateV1:
    track_id: int
    generation: str
    timestamp: float
    shape_hypotheses: Tuple[PredictedShapeOccupancyV1, ...]
    center_reference_semantics: str
    velocity_semantics: str
    observability: str
    prediction_only: bool
    last_direct_geometry_timestamp: float
    runtime_gt_used: bool = False


@dataclass
class _Motion:
    timestamp: float
    center: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    last_direct_timestamp: float


class DynamicObjectOccupancyStateBuilderV1:
    """Maintains independent motion for each geometry type."""

    version = STATE_VERSION

    def __init__(self, velocity_smoothing=.5, mode_switch_variance=1.0):
        self.velocity_smoothing = float(velocity_smoothing)
        self.mode_switch_variance = float(mode_switch_variance)
        self._motion: Dict[Tuple[int, str, str], _Motion] = {}

    def reset(self):
        self._motion.clear()

    def delete_missing(self, live_track_ids):
        live = {int(value) for value in live_track_ids}
        removed = [key for key in self._motion if key[0] not in live]
        for key in removed:
            del self._motion[key]
        return tuple(sorted(removed))

    def update(self, tracked: TrackedShapeHypothesesV1):
        outputs = []
        live_types = set()
        for hypothesis in tracked.hypotheses:
            key = (
                tracked.track_id, tracked.generation,
                hypothesis.geometry_type,
            )
            live_types.add(hypothesis.geometry_type)
            previous = self._motion.get(key)
            center = hypothesis.center_world
            if previous is None:
                velocity = np.zeros(3)
                velocity_covariance = (
                    np.eye(3)*self.mode_switch_variance
                )
                source = "same_mode_uninitialized"
                last_direct_timestamp = (
                    float("nan") if tracked.prediction_only
                    else tracked.timestamp
                )
            else:
                dt = tracked.timestamp-previous.timestamp
                if dt <= 0:
                    raise ValueError("occupancy timestamps must increase")
                if tracked.prediction_only:
                    center = previous.center+dt*previous.velocity
                    velocity = previous.velocity.copy()
                    velocity_covariance = (
                        previous.covariance
                        + np.eye(3)*self.mode_switch_variance*dt
                    )
                    source = "same_shape_prediction_only"
                    last_direct_timestamp = previous.last_direct_timestamp
                else:
                    measured = (center-previous.center)/dt
                    velocity = (
                        self.velocity_smoothing*previous.velocity
                        +(1-self.velocity_smoothing)*measured
                    )
                    velocity_covariance = (
                        previous.covariance
                        + hypothesis.model_covariance_world
                    )/dt**2
                    source = "same_shape_geometry_finite_difference"
                    last_direct_timestamp = tracked.timestamp
            self._motion[key] = _Motion(
                tracked.timestamp, center.copy(), velocity.copy(),
                velocity_covariance.copy(), last_direct_timestamp,
            )
            outputs.append(PredictedShapeOccupancyV1(
                hypothesis.geometry_type, center, velocity,
                hypothesis.center_z_interval_m,
                hypothesis.radius_interval_m,
                hypothesis.half_height_interval_m,
                hypothesis.model_covariance_world,
                velocity_covariance, source,
            ))
        for key in tuple(self._motion):
            if (
                key[0] == tracked.track_id
                and key[1] == tracked.generation
                and key[2] not in live_types
            ):
                del self._motion[key]
        return DynamicObjectOccupancyStateV1(
            tracked.track_id, tracked.generation, tracked.timestamp,
            tuple(outputs), "shape_specific_geometric_center_or_set",
            "independent_same_shape_causal_geometry_motion",
            tracked.mode, tracked.prediction_only,
            min(
                (
                    self._motion[
                        (tracked.track_id, tracked.generation,
                         hypothesis.geometry_type)
                    ].last_direct_timestamp
                    for hypothesis in tracked.hypotheses
                ),
                default=float("nan"),
            ),
        )


__all__ = [
    "STATE_VERSION", "PredictedShapeOccupancyV1",
    "DynamicObjectOccupancyStateV1",
    "DynamicObjectOccupancyStateBuilderV1",
]
