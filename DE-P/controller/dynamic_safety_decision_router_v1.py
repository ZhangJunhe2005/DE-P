"""Fail-closed decision routing for development-only planner integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


ROUTER_VERSION = "dynamic_safety_decision_router_v1"


class FeatureMode(str, Enum):
    LEGACY_OFF = "LEGACY_OFF"
    SHADOW = "SHADOW"
    DEVELOPMENT_ACTIVE = "DEVELOPMENT_ACTIVE"


class ControlDisposition(str, Enum):
    LEGACY_BYPASS = "LEGACY_BYPASS"
    SHADOW_ONLY = "SHADOW_ONLY"
    SUBMIT_CANDIDATE = "SUBMIT_CANDIDATE"
    DEVELOPMENT_SAFE_ABORT = "DEVELOPMENT_SAFE_ABORT"


@dataclass(frozen=True)
class RoutedPlannerDecisionV1:
    feature_mode: FeatureMode
    decision_status: str
    original_candidate_id: int
    selected_candidate_id: int | None
    shadow_recommended_candidate_id: int | None
    disposition: ControlDisposition
    safe_abort: bool
    formal_command_modified: bool
    reason: str | None


def route_dynamic_safety_decision(
    *, mode, decision_status, original_candidate_id,
    recommended_candidate_id, safe_candidate_ids,
):
    mode = FeatureMode(mode)
    original = int(original_candidate_id)
    safe = {int(value) for value in safe_candidate_ids}
    recommended = (
        None if recommended_candidate_id is None
        else int(recommended_candidate_id)
    )
    if mode == FeatureMode.LEGACY_OFF:
        return RoutedPlannerDecisionV1(
            mode, "LEGACY_OFF_BYPASS", original, original, None,
            ControlDisposition.LEGACY_BYPASS, False, False, None,
        )
    if mode == FeatureMode.SHADOW:
        return RoutedPlannerDecisionV1(
            mode, str(decision_status), original, original, recommended,
            ControlDisposition.SHADOW_ONLY, False, False, None,
        )
    if decision_status in {"NO_ACTIVE_DYNAMIC_RISK", "KEEP_ORIGINAL"}:
        if decision_status == "KEEP_ORIGINAL" and original not in safe:
            decision_status = "INVALID_EVALUATION"
        else:
            return RoutedPlannerDecisionV1(
                mode, str(decision_status), original, original, recommended,
                ControlDisposition.SUBMIT_CANDIDATE, False, False, None,
            )
    if decision_status == "SWITCH_TO_SAFE_CANDIDATE":
        if recommended is not None and recommended in safe:
            return RoutedPlannerDecisionV1(
                mode, decision_status, original, recommended, recommended,
                ControlDisposition.SUBMIT_CANDIDATE, False, True, None,
            )
        decision_status = "INVALID_EVALUATION"
    if decision_status in {
        "NO_SAFE_CANDIDATE", "UNRESOLVED_DYNAMIC_RISK",
        "INVALID_EVALUATION",
    }:
        return RoutedPlannerDecisionV1(
            mode, str(decision_status), original, None, recommended,
            ControlDisposition.DEVELOPMENT_SAFE_ABORT, True, False,
            {
                "NO_SAFE_CANDIDATE": "all_candidates_vetoed",
                "UNRESOLVED_DYNAMIC_RISK": "bounded_occupancy_unavailable",
                "INVALID_EVALUATION": "snapshot_or_adapter_invalid",
            }[str(decision_status)],
        )
    return RoutedPlannerDecisionV1(
        mode, "INVALID_EVALUATION", original, None, None,
        ControlDisposition.DEVELOPMENT_SAFE_ABORT, True, False,
        "unknown_decision_status",
    )


__all__ = [
    "ROUTER_VERSION", "FeatureMode", "ControlDisposition",
    "RoutedPlannerDecisionV1", "route_dynamic_safety_decision",
]
