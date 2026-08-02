"""Explicit per-hypothesis stale-geometry prediction time origins."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


TIME_CONTRACT_VERSION = "stale_geometry_time_contract_v1"


class TimeOriginMode(str, Enum):
    GEOMETRY_TIMESTAMP = "GEOMETRY_TIMESTAMP"
    CURRENT_PROPAGATED_STATE = "CURRENT_PROPAGATED_STATE"
    INVALID_MISSING_TIMESTAMP = "INVALID_MISSING_TIMESTAMP"


@dataclass(frozen=True)
class EffectiveHorizonV1:
    geometry_timestamp: float
    motion_state_timestamp: float
    current_query_timestamp: float
    candidate_relative_times_s: np.ndarray
    geometry_age_s: float
    state_age_s: float
    effective_prediction_horizons_s: np.ndarray
    time_origin_mode: TimeOriginMode
    double_age_guard: bool
    stale_age_included: bool
    runtime_gt_used: bool = False


def resolve_effective_horizon(
    *, geometry_timestamp, motion_state_timestamp,
    current_query_timestamp, candidate_relative_times_s,
    position_reference_timestamp,
):
    values = np.asarray(candidate_relative_times_s, dtype=np.float64)
    timestamps = np.asarray([
        geometry_timestamp, motion_state_timestamp,
        current_query_timestamp, position_reference_timestamp,
    ], dtype=np.float64)
    if (
        values.ndim != 1 or not np.isfinite(values).all()
        or not np.isfinite(timestamps).all()
    ):
        raise ValueError("time contract inputs must be finite")
    if np.any(values < 0) or current_query_timestamp < geometry_timestamp:
        raise ValueError("time contract is not causal")
    geometry_age = float(current_query_timestamp-geometry_timestamp)
    state_age = float(current_query_timestamp-motion_state_timestamp)
    if abs(position_reference_timestamp-current_query_timestamp) <= 1e-9:
        effective = values.copy()
        mode = TimeOriginMode.CURRENT_PROPAGATED_STATE
        included = False
    elif abs(position_reference_timestamp-geometry_timestamp) <= 1e-9:
        effective = values+geometry_age
        mode = TimeOriginMode.GEOMETRY_TIMESTAMP
        included = geometry_age > 1e-9
    else:
        raise ValueError("position reference timestamp is ambiguous")
    return EffectiveHorizonV1(
        float(geometry_timestamp), float(motion_state_timestamp),
        float(current_query_timestamp), values.copy(), geometry_age,
        state_age, effective, mode, True, included, False,
    )


__all__ = [
    "TIME_CONTRACT_VERSION", "TimeOriginMode", "EffectiveHorizonV1",
    "resolve_effective_horizon",
]

