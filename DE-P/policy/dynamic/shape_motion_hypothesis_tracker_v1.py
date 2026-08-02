"""Independent temporal motion histories for each legal shape hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .dynamic_object_geometry_model_v1 import ShapeGeometryEvaluationV1
from .shape_aware_motion_state_v1 import (
    MotionEstimateV1, MotionObservability,
    StableReferenceMotionEstimatorV1,
)


TRACKER_VERSION = "shape_motion_hypothesis_tracker_v1"


@dataclass(frozen=True)
class ShapeMotionHypothesisV1:
    geometry_type: str
    geometry_hypothesis: object
    motion: MotionEstimateV1
    transition_age: int
    prediction_only: bool
    runtime_gt_used: bool = False


@dataclass(frozen=True)
class TrackedShapeMotionV1:
    track_id: int
    generation: str
    timestamp: float
    hypotheses: Tuple[ShapeMotionHypothesisV1, ...]
    observability: MotionObservability
    prediction_only: bool
    missed_frames: int
    runtime_gt_used: bool = False


@dataclass
class _ShapeMemory:
    estimator: StableReferenceMotionEstimatorV1
    geometry: object
    transition_age: int
    last_direct_timestamp: float


class ShapeMotionHypothesisTrackerV1:
    version = TRACKER_VERSION

    def __init__(self, history_config, transition_config):
        self.history_config = dict(history_config)
        self.transition_config = dict(transition_config)
        self._tracks: Dict[Tuple[int, str], Dict[str, _ShapeMemory]] = {}
        self._missed: Dict[Tuple[int, str], int] = {}

    def reset(self):
        self._tracks.clear()
        self._missed.clear()

    def delete_missing(self, live_track_ids):
        live = {int(value) for value in live_track_ids}
        removed = [key for key in self._tracks if key[0] not in live]
        for key in removed:
            del self._tracks[key]
            self._missed.pop(key, None)
        return tuple(sorted(removed))

    def _estimator(self, mode):
        values = self.history_config
        return StableReferenceMotionEstimatorV1(
            mode, values["minimum_direct_frames"],
            values["maximum_direct_frames"],
            values["maximum_history_age_s"],
            values["huber_delta_m"], values["robust_iterations"],
            values["maximum_speed_mps"], values["minimum_dt_s"],
        )

    def update(self, track_id, generation, evaluation):
        if not isinstance(evaluation, ShapeGeometryEvaluationV1):
            raise TypeError("evaluation must be ShapeGeometryEvaluationV1")
        key = (int(track_id), str(generation))
        memory = self._tracks.setdefault(key, {})
        self._missed[key] = 0
        current_types = {
            hypothesis.geometry_type for hypothesis in evaluation.hypotheses
        }
        outputs = []
        for hypothesis in evaluation.hypotheses:
            mode = hypothesis.geometry_type
            item = memory.get(mode)
            if item is None:
                item = _ShapeMemory(
                    self._estimator(mode), hypothesis, 0,
                    evaluation.timestamp,
                )
                memory[mode] = item
            item.transition_age += 1
            item.geometry = hypothesis
            item.last_direct_timestamp = evaluation.timestamp
            quality = float(np.clip(hypothesis.evidence_score, .05, 1.))
            weak = (
                mode == "vertical_cylinder"
                and hypothesis.observability == "height_weakly_observable"
            )
            motion = item.estimator.update(
                evaluation.timestamp, hypothesis.center_world,
                geometry_uncertainty_m=max(
                    hypothesis.fit_residual_m,
                    float(np.sqrt(np.linalg.eigvalsh(
                        hypothesis.model_covariance_world
                    ).max())),
                ),
                quality=quality, weak_vertical=weak,
            )
            outputs.append(ShapeMotionHypothesisV1(
                mode, hypothesis, motion, item.transition_age, False,
            ))
        # Ambiguity never silently deletes an existing legal shape. A
        # determined direct observation may retire the other mode only after
        # the predeclared number of stable direct frames.
        stable_required = int(
            self.transition_config["stable_direct_frames_required"]
        )
        if len(current_types) == 1:
            selected = next(iter(current_types))
            if memory[selected].transition_age >= stable_required:
                for mode in tuple(memory):
                    if mode != selected:
                        del memory[mode]
        observability = self._observability(outputs, evaluation)
        return TrackedShapeMotionV1(
            key[0], key[1], evaluation.timestamp, tuple(outputs),
            observability, False, 0,
        )

    def predict_only(self, track_id, generation, timestamp):
        key = (int(track_id), str(generation))
        memory = self._tracks.get(key)
        if not memory:
            return None
        missed = self._missed.get(key, 0)+1
        self._missed[key] = missed
        if missed > int(self.transition_config["expire_after_frames"]):
            return TrackedShapeMotionV1(
                key[0], key[1], float(timestamp), (),
                MotionObservability.MOTION_EXPIRED, True, missed,
            )
        outputs = []
        for mode, item in memory.items():
            motion = item.estimator.last_estimate
            if motion is None:
                continue
            outputs.append(ShapeMotionHypothesisV1(
                mode, item.geometry, motion, item.transition_age, True,
            ))
        observability = (
            MotionObservability.MOTION_AMBIGUOUS
            if len(outputs) > 1 else (
                outputs[0].motion.observability if outputs
                else MotionObservability.MOTION_EXPIRED
            )
        )
        return TrackedShapeMotionV1(
            key[0], key[1], float(timestamp), tuple(outputs),
            observability, True, missed,
        )

    @staticmethod
    def _observability(outputs, evaluation):
        if not outputs:
            return MotionObservability.MOTION_EXPIRED
        if len(outputs) > 1:
            return MotionObservability.MOTION_AMBIGUOUS
        if evaluation.observability.value == "SUPPORT_ONLY":
            return MotionObservability.SUPPORT_PROPAGATION_ONLY
        return outputs[0].motion.observability


__all__ = [
    "TRACKER_VERSION", "ShapeMotionHypothesisV1",
    "TrackedShapeMotionV1", "ShapeMotionHypothesisTrackerV1",
]

