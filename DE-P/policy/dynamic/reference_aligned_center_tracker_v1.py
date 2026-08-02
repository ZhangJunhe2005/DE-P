"""Causal shadow state for reference-aligned centre evidence.

It is not an association system and cannot create a formal TrackManager track.
Identity and generation are supplied by the frozen TrackManager.  Velocity is
derived only from consecutive centre evidence, never copied from surface
velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .tracking_collision_reference_bridge_v1 import (
    ReferenceEvidenceV1, ReferenceObservability,
)


TRACKER_VERSION = "reference_aligned_center_tracker_v1"


@dataclass(frozen=True)
class CenterStateV1:
    track_id: int
    generation: str
    timestamp: float
    position_world: np.ndarray
    velocity_world: np.ndarray
    state_covariance: np.ndarray
    reference_observability: str
    last_direct_center_evidence_timestamp: float
    prediction_only: bool
    center_velocity_source: str


@dataclass
class _State:
    generation: str
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    last_direct_timestamp: float
    observability: str
    direct_count: int


class ReferenceAlignedCenterTrackerV1:
    """Bounded alpha-filter shadow state keyed by formal track generation."""

    version = TRACKER_VERSION

    def __init__(self, velocity_smoothing=.5, acceleration_noise_mps2=1.5):
        self.velocity_smoothing = float(velocity_smoothing)
        self.acceleration_noise_mps2 = float(acceleration_noise_mps2)
        if not 0 <= self.velocity_smoothing < 1:
            raise ValueError("velocity_smoothing must be in [0,1)")
        if self.acceleration_noise_mps2 <= 0:
            raise ValueError("acceleration noise must be positive")
        self._states: Dict[int, _State] = {}

    def reset(self):
        self._states.clear()

    def delete_missing(self, live_track_ids):
        live = {int(value) for value in live_track_ids}
        deleted = sorted(set(self._states)-live)
        for track_id in deleted:
            del self._states[track_id]
        return tuple(deleted)

    def update(self, track_id, generation, evidence: ReferenceEvidenceV1):
        track_id, generation = int(track_id), str(generation)
        if track_id < 0 or not generation:
            raise ValueError("track identity/generation is invalid")
        if not isinstance(evidence, ReferenceEvidenceV1):
            raise TypeError("evidence must be ReferenceEvidenceV1")
        if not evidence.valid or evidence.center_estimate_world is None:
            return self.predict(track_id, generation, evidence.timestamp)
        current = self._states.get(track_id)
        if current is not None and current.generation != generation:
            del self._states[track_id]
            current = None
        position = np.asarray(evidence.center_estimate_world, dtype=np.float64)
        if current is None:
            velocity = np.zeros(3, dtype=np.float64)
            covariance = np.block([
                [evidence.center_covariance_world, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3) * 1.0],
            ])
            direct_count = 1
        else:
            dt = float(evidence.timestamp-current.timestamp)
            if dt <= 0:
                raise ValueError("center evidence timestamps must increase")
            measured_velocity = (position-current.position)/dt
            velocity = (
                self.velocity_smoothing*current.velocity
                + (1-self.velocity_smoothing)*measured_velocity
            )
            position_variance = evidence.center_covariance_world
            velocity_variance = (
                current.covariance[:3, :3] + position_variance
            ) / dt**2
            covariance = np.block([
                [position_variance, np.zeros((3, 3))],
                [np.zeros((3, 3)), velocity_variance],
            ])
            direct_count = current.direct_count+1
        self._states[track_id] = _State(
            generation=generation,
            timestamp=float(evidence.timestamp),
            position=position.copy(),
            velocity=velocity.copy(),
            covariance=covariance.copy(),
            last_direct_timestamp=float(evidence.timestamp),
            observability=evidence.observability.value,
            direct_count=direct_count,
        )
        return self._snapshot(track_id, prediction_only=False)

    def predict(self, track_id, generation, timestamp):
        track_id, generation = int(track_id), str(generation)
        current = self._states.get(track_id)
        if current is None or current.generation != generation:
            return None
        dt = float(timestamp-current.timestamp)
        if dt < 0:
            raise ValueError("prediction timestamp precedes center state")
        if dt:
            transition = np.eye(6)
            transition[:3, 3:] = np.eye(3)*dt
            q = self.acceleration_noise_mps2**2
            process = np.block([
                [np.eye(3)*dt**4/4*q, np.eye(3)*dt**3/2*q],
                [np.eye(3)*dt**3/2*q, np.eye(3)*dt**2*q],
            ])
            state = np.concatenate((current.position, current.velocity))
            state = transition@state
            current.position = state[:3]
            current.velocity = state[3:]
            current.covariance = (
                transition@current.covariance@transition.T + process
            )
            current.timestamp = float(timestamp)
        return self._snapshot(track_id, prediction_only=True)

    def _snapshot(self, track_id, prediction_only):
        state = self._states[track_id]
        return CenterStateV1(
            track_id=track_id,
            generation=state.generation,
            timestamp=state.timestamp,
            position_world=state.position.copy(),
            velocity_world=state.velocity.copy(),
            state_covariance=state.covariance.copy(),
            reference_observability=state.observability,
            last_direct_center_evidence_timestamp=state.last_direct_timestamp,
            prediction_only=bool(prediction_only),
            center_velocity_source=(
                "causal_center_evidence_finite_difference"
                if state.direct_count > 1 else "uninitialized_zero"
            ),
        )


__all__ = [
    "TRACKER_VERSION", "CenterStateV1",
    "ReferenceAlignedCenterTrackerV1",
]
