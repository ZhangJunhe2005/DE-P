"""Non-dynamic, bounded near-horizon occupancy-conflict semantic."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CONTRACT_VERSION = "transient_unresolved_risk_v1"


@dataclass(frozen=True, slots=True)
class TransientUnresolvedRiskV1:
    risk_id: int
    source_outcome_ids: tuple[int, ...]
    birth_timestamp: float
    expiry_timestamp: float
    conflicting_candidate_ids: tuple[int, ...]
    support_bounds: tuple[float, ...]
    planner_semantic: str = "UNRESOLVED_TRANSIENT_RISK"
    dynamic_claim: bool = False
    position_world: None = None
    velocity_world: None = None
    shape: None = None
    formal_tracker_feed: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.risk_id < 0 or not self.conflicting_candidate_ids:
            raise ValueError("a transient risk requires a candidate conflict")
        if self.dynamic_claim or self.formal_tracker_feed or self.runtime_gt_used:
            raise ValueError("transient risk cannot claim dynamic/GT/formal state")
        if (
            not math.isfinite(self.birth_timestamp)
            or not math.isfinite(self.expiry_timestamp)
            or self.expiry_timestamp <= self.birth_timestamp
        ):
            raise ValueError("invalid transient lifetime")


def _point_aabb_clearance(point, bounds):
    point = np.asarray(point, np.float64)
    lower = np.asarray(bounds[:3], np.float64)
    upper = np.asarray(bounds[3:], np.float64)
    return float(np.linalg.norm(np.maximum(np.maximum(lower-point, 0.), point-upper)))


def build_transient_risk(
    pending, candidate_samples, *, known_static: bool,
    timestamp: float, expiry_timestamp: float, clearance_m: float,
):
    """Return risk only for a direct current-support/candidate conflict."""
    if known_static:
        return None
    conflicts = []
    for candidate_id, samples in candidate_samples:
        if any(
            _point_aabb_clearance(point, pending.support_bounds)
            <= float(clearance_m)
            for point in samples
        ):
            conflicts.append(int(candidate_id))
    if not conflicts:
        return None
    return TransientUnresolvedRiskV1(
        risk_id=pending.state_id,
        source_outcome_ids=pending.source_outcome_ids,
        birth_timestamp=float(timestamp),
        expiry_timestamp=float(expiry_timestamp),
        conflicting_candidate_ids=tuple(sorted(set(conflicts))),
        support_bounds=pending.support_bounds,
    )


__all__ = [
    "CONTRACT_VERSION", "TransientUnresolvedRiskV1",
    "build_transient_risk",
]
