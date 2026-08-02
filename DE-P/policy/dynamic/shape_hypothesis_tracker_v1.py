"""Causal hysteresis for shadow sphere/cylinder geometry hypotheses."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Tuple

from .dynamic_object_geometry_model_v1 import (
    ShapeGeometryEvaluationV1, ShapeHypothesisV1, ShapeObservability,
)


TRACKER_VERSION = "shape_hypothesis_tracker_v1"


@dataclass(frozen=True)
class TrackedShapeHypothesesV1:
    track_id: int
    generation: str
    timestamp: float
    mode: str
    hypotheses: Tuple[ShapeHypothesisV1, ...]
    direct_evidence_frames: int
    ambiguous_age: int
    prediction_only: bool
    runtime_gt_used: bool = False


@dataclass
class _Memory:
    generation: str
    timestamp: float
    history: deque
    stable_mode: str
    hypotheses: Tuple[ShapeHypothesisV1, ...]
    direct_frames: int
    ambiguous_age: int
    missed: int


class ShapeHypothesisTrackerV1:
    """Does not associate or birth formal tracks; identity is an input."""

    version = TRACKER_VERSION

    def __init__(
        self, minimum_direct_evidence_frames=2,
        ambiguous_expiry_frames=2, maximum_missed_frames=2,
    ):
        self.minimum_direct_evidence_frames = int(
            minimum_direct_evidence_frames
        )
        self.ambiguous_expiry_frames = int(ambiguous_expiry_frames)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self._memory: Dict[int, _Memory] = {}

    def reset(self):
        self._memory.clear()

    def delete_missing(self, live_track_ids):
        live = {int(value) for value in live_track_ids}
        deleted = tuple(sorted(set(self._memory)-live))
        for track_id in deleted:
            del self._memory[track_id]
        return deleted

    def update(self, track_id, generation, evaluation):
        if not isinstance(evaluation, ShapeGeometryEvaluationV1):
            raise TypeError("evaluation must be ShapeGeometryEvaluationV1")
        track_id, generation = int(track_id), str(generation)
        memory = self._memory.get(track_id)
        if memory is not None and memory.generation != generation:
            del self._memory[track_id]
            memory = None
        if memory is None:
            memory = _Memory(
                generation, evaluation.timestamp, deque(maxlen=4),
                "ambiguous", (), 0, 0, 0,
            )
            self._memory[track_id] = memory
        label = evaluation.observability.value
        memory.history.append(label)
        memory.timestamp = evaluation.timestamp
        memory.direct_frames += 1
        memory.missed = 0
        if evaluation.observability in {
            ShapeObservability.SPHERE_OBSERVABLE,
            ShapeObservability.CYLINDER_OBSERVABLE,
        }:
            candidate = (
                "sphere" if evaluation.observability
                == ShapeObservability.SPHERE_OBSERVABLE
                else "vertical_cylinder"
            )
            count = Counter(memory.history)[label]
            if count >= self.minimum_direct_evidence_frames:
                memory.stable_mode = candidate
            memory.ambiguous_age = 0
            memory.hypotheses = evaluation.hypotheses
        elif evaluation.observability == ShapeObservability.SHAPE_AMBIGUOUS:
            memory.ambiguous_age += 1
            memory.hypotheses = evaluation.hypotheses
            if memory.ambiguous_age > self.ambiguous_expiry_frames:
                memory.stable_mode = "ambiguous"
        elif evaluation.observability == ShapeObservability.SUPPORT_ONLY:
            memory.ambiguous_age += 1
            memory.stable_mode = "support_only"
            memory.hypotheses = evaluation.hypotheses
        else:
            memory.ambiguous_age += 1
            if memory.ambiguous_age > self.ambiguous_expiry_frames:
                memory.stable_mode = "expired"
                memory.hypotheses = ()
        return self._snapshot(track_id, False)

    def predict_only(self, track_id, generation, timestamp):
        track_id, generation = int(track_id), str(generation)
        memory = self._memory.get(track_id)
        if memory is None or memory.generation != generation:
            return None
        memory.missed += 1
        memory.timestamp = float(timestamp)
        if memory.missed > self.maximum_missed_frames:
            memory.stable_mode = "expired"
            memory.hypotheses = ()
        return self._snapshot(track_id, True)

    def _snapshot(self, track_id, prediction_only):
        memory = self._memory[track_id]
        return TrackedShapeHypothesesV1(
            track_id, memory.generation, memory.timestamp,
            memory.stable_mode, memory.hypotheses,
            memory.direct_frames, memory.ambiguous_age,
            bool(prediction_only),
        )


__all__ = [
    "TRACKER_VERSION", "TrackedShapeHypothesesV1",
    "ShapeHypothesisTrackerV1",
]
