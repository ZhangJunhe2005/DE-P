"""Read-only annotation of formal-track history for measurement review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


CONTRACT_VERSION = "historical_measurement_context_v1"


@dataclass(frozen=True)
class HistoricalMeasurementContextV1:
    track_exists: bool
    track_id: Optional[int]
    generation: Optional[str]
    confirmed: bool
    dynamic: bool
    attention_authorized: bool
    last_direct_measurement_time: Optional[float]
    prediction_only_age: int
    last_valid_geometry_bounds: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ]
    last_safety_evidence: Optional[str]
    current_measurement_outcome: str
    read_only: bool = True
    track_reactivated: bool = False
    confidence_modified: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("historical annotation cannot use runtime GT")
        if not self.read_only or self.track_reactivated:
            raise ValueError("historical context must remain read-only")
        if self.confidence_modified:
            raise ValueError("historical context cannot modify confidence")
        if self.track_exists:
            if self.track_id is None or self.generation is None:
                raise ValueError("existing track requires identity")
            if self.prediction_only_age < 0:
                raise ValueError("prediction age must be non-negative")
        elif any((
            self.track_id is not None, self.generation is not None,
            self.confirmed, self.dynamic, self.attention_authorized,
        )):
            raise ValueError("absent track cannot carry track state")
        if (
            self.last_direct_measurement_time is not None
            and not np.isfinite(self.last_direct_measurement_time)
        ):
            raise ValueError("last direct time must be finite")


def context_from_track(
    track,
    current_measurement_outcome: str,
    frame_period_s: float,
):
    if track is None:
        return HistoricalMeasurementContextV1(
            track_exists=False, track_id=None, generation=None,
            confirmed=False, dynamic=False, attention_authorized=False,
            last_direct_measurement_time=None, prediction_only_age=0,
            last_valid_geometry_bounds=None, last_safety_evidence=None,
            current_measurement_outcome=current_measurement_outcome,
        )
    position = np.asarray(track.position_world, dtype=np.float64)
    radius = .5
    direct_time = (
        float(track.timestamp)
        - float(track.prediction_only_age)*float(frame_period_s)
    )
    return HistoricalMeasurementContextV1(
        track_exists=True,
        track_id=int(track.track_id),
        generation=(
            f"{track.birth_frame}:{track.birth_observation_id}"
        ),
        confirmed=bool(track.is_confirmed),
        dynamic=bool(track.is_dynamic),
        attention_authorized=bool(track.attention_authorized),
        last_direct_measurement_time=direct_time,
        prediction_only_age=int(track.prediction_only_age),
        last_valid_geometry_bounds=(
            tuple(position-radius), tuple(position+radius)
        ),
        last_safety_evidence=(
            "DIRECT_MEASUREMENT_HISTORY"
            if track.ever_directly_observed else None
        ),
        current_measurement_outcome=current_measurement_outcome,
    )


__all__ = [
    "CONTRACT_VERSION", "HistoricalMeasurementContextV1",
    "context_from_track",
]
