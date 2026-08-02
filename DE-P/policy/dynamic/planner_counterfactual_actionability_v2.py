"""Planner-counterfactual actionability authority and learned-risk adapter.

The authority is offline-only.  It compares one frozen candidate set under
static-only and composite-dynamic safety masks.  It never creates, optimizes,
or re-scores candidate trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


AUTHORITY_VERSION = "planner_counterfactual_actionability_v2"
ADAPTER_VERSION = "learned_dynamic_conflict_adapter_v1"


class ActionabilityValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CounterfactualActionabilityV2:
    target: int
    valid: bool
    reasons: tuple[str, ...]
    static_safe_count: int
    composite_safe_count: int
    static_recommendation: int | None
    composite_recommendation: int | None
    minimum_dynamic_clearance_m: float | None
    authority_version: str = AUTHORITY_VERSION


def _recommend(safe, scores):
    ids = np.flatnonzero(safe)
    if not len(ids):
        return None
    return int(ids[np.argmin(np.asarray(scores)[ids])])


def planner_counterfactual_actionability(
    static_safe,
    composite_safe,
    candidate_scores,
    *,
    actor_future_complete,
    candidate_authority_complete=True,
    static_composite_comparable=True,
    mixed_authority_conflict=False,
    minimum_dynamic_clearance_m=None,
    clearance_boundary_crossed=False,
    ttc_boundary_crossed=False,
):
    """Return the frozen counterfactual label without scenario shortcuts."""
    static_safe = np.asarray(static_safe, dtype=np.bool_)
    composite_safe = np.asarray(composite_safe, dtype=np.bool_)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if (
        static_safe.ndim != 1 or composite_safe.shape != static_safe.shape
        or scores.shape != static_safe.shape or not np.isfinite(scores).all()
    ):
        raise ValueError("candidate masks/scores must be aligned finite vectors")
    invalid = []
    if not candidate_authority_complete:
        invalid.append("candidate_authority_missing")
    if not actor_future_complete:
        invalid.append("actor_future_authority_missing")
    if not static_composite_comparable:
        invalid.append("counterfactual_not_comparable")
    if mixed_authority_conflict:
        invalid.append("mixed_authority_conflict")
    static_recommendation = _recommend(static_safe, scores)
    composite_recommendation = _recommend(composite_safe, scores)
    if static_recommendation is None:
        invalid.append("static_authority_has_no_safe_candidate")
    if np.any(composite_safe & ~static_safe):
        invalid.append("composite_marked_candidate_safer_than_static")
    if invalid:
        return CounterfactualActionabilityV2(
            0, False, tuple(invalid), int(static_safe.sum()),
            int(composite_safe.sum()), static_recommendation,
            composite_recommendation, minimum_dynamic_clearance_m,
        )
    reasons = []
    if np.any(static_safe & ~composite_safe):
        reasons.append("safe_candidate_became_unsafe")
    if static_recommendation != composite_recommendation:
        reasons.append("router_recommendation_changed")
    if composite_safe.sum() < static_safe.sum():
        reasons.append("safe_candidate_count_reduced")
    if composite_recommendation is None:
        reasons.append("no_safe_candidate")
    if clearance_boundary_crossed:
        reasons.append("clearance_boundary_crossed")
    if ttc_boundary_crossed:
        reasons.append("ttc_boundary_crossed")
    return CounterfactualActionabilityV2(
        int(bool(reasons)), True, tuple(reasons), int(static_safe.sum()),
        int(composite_safe.sum()), static_recommendation,
        composite_recommendation, minimum_dynamic_clearance_m,
    )


class LearnedDynamicConflictAdapterV1:
    """Fail-closed mapping; it cannot command, track, or bypass BDRR1/BRIR1."""

    def __init__(self, conflict_threshold=0.5, confidence_threshold=0.6):
        self.conflict_threshold = float(conflict_threshold)
        self.confidence_threshold = float(confidence_threshold)
        if not 0 < self.conflict_threshold < 1:
            raise ValueError("conflict threshold must be in (0,1)")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in (0,1]")

    def map(self, conflict_probability, semantic_class, confidence):
        probability = float(conflict_probability)
        confidence = float(confidence)
        semantic = str(semantic_class)
        if (
            not np.isfinite((probability, confidence)).all()
            or not 0 <= probability <= 1 or not 0 <= confidence <= 1
        ):
            raise ValueError("probability/confidence must be finite in [0,1]")
        if confidence < self.confidence_threshold:
            status = "PENDING_OR_UNKNOWN_SUPPORT"
        elif probability >= self.conflict_threshold and semantic == "DYNAMIC_SUPPORT":
            status = "ACTIVE_LEARNED_DYNAMIC_RISK"
        elif probability >= self.conflict_threshold and semantic == "UNKNOWN_AMBIGUOUS":
            status = "UNRESOLVED_TRANSIENT_RISK"
        elif semantic in {"STATIC_SUPPORT", "SENSOR_OR_SEGMENTATION_ARTIFACT"}:
            status = "DIAGNOSTIC_ONLY"
        else:
            status = "PENDING_OR_UNKNOWN_SUPPORT"
        return {
            "adapter_version": ADAPTER_VERSION,
            "status": status,
            "formal_tracker_write": False,
            "planner_command": None,
            "bypass_bounded_reachability": False,
            "low_conflict_means_no_active": False,
        }


__all__ = [
    "AUTHORITY_VERSION", "ADAPTER_VERSION",
    "CounterfactualActionabilityV2",
    "planner_counterfactual_actionability",
    "LearnedDynamicConflictAdapterV1",
]
