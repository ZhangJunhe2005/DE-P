"""Independent causal-evidence authorization for bounded safety supports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


CONTRACT_VERSION = "provisional_evidence_authorizer_v1"
FORBIDDEN_RUNTIME_KEYS = frozenset({
    "gt_actor_id", "gt_geometry", "owner_map", "future_frame",
    "future_information", "sequence_id", "scenario_id",
})


class EvidenceLevelV1(int, Enum):
    NO_EVIDENCE = 0
    SINGLE_WEAK_CUE = 1
    CORROBORATED_NO_HISTORY = 2
    RECENT_DYNAMIC_CONTEXT = 3
    FORMAL_OR_STRONG_PROVISIONAL_CONTEXT = 4


class SupportDiagnosticV1(str, Enum):
    AUTHORIZED_DYNAMIC_SUPPORT = "AUTHORIZED_DYNAMIC_SUPPORT"
    STATIC_SUPPORT_DIAGNOSTIC = "STATIC_SUPPORT_DIAGNOSTIC"
    UNKNOWN_SUPPORT_DIAGNOSTIC = "UNKNOWN_SUPPORT_DIAGNOSTIC"


@dataclass(frozen=True)
class ProvisionalEvidenceDecisionV1:
    authorized: bool
    evidence_level: EvidenceLevelV1
    reason_code: str
    diagnostic: SupportDiagnosticV1
    evidence_families: tuple[str, ...]
    unavailable_families: tuple[str, ...]
    negative_evidence: tuple[str, ...]
    history_level: str
    support_quality: str
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.authorized != (
            self.diagnostic
            == SupportDiagnosticV1.AUTHORIZED_DYNAMIC_SUPPORT
        ):
            raise ValueError("authorization/diagnostic mismatch")


def _finite_bounds(support):
    if not isinstance(support, Mapping):
        return None
    try:
        lower = np.asarray(support["position_set_min_world"], np.float64)
        upper = np.asarray(support["position_set_max_world"], np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        lower.shape != (3,) or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper < lower)
    ):
        return None
    return lower, upper


def _bbox_fill(fields):
    bbox = fields.get("pixel_bbox")
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return 0.0
    width = max(1, int(bbox[2])-int(bbox[0])+1)
    height = max(1, int(bbox[3])-int(bbox[1])+1)
    return float(fields.get("pixel_count", 0))/(width*height)


def _bounds_compatible(current, historical, margin):
    if current is None or historical is None:
        return False
    try:
        old_lower = np.asarray(historical[0], np.float64)
        old_upper = np.asarray(historical[1], np.float64)
    except (TypeError, ValueError):
        return False
    if (
        old_lower.shape != (3,) or old_upper.shape != (3,)
        or not np.isfinite(old_lower).all()
        or not np.isfinite(old_upper).all()
    ):
        return False
    lower, upper = current
    return bool(np.all(lower <= old_upper+margin)
                and np.all(old_lower <= upper+margin))


class ProvisionalEvidenceAuthorizerV1:
    """Authorize support birth without changing lifecycle or association."""

    families = (
        "HISTORICAL_DYNAMIC_CONTEXT", "TEMPORAL_PERSISTENCE",
        "MOTION_CONSISTENCY", "APPROACH_OR_SCALE_TREND",
        "CROSS_SOURCE_CORROBORATION",
    )

    def __init__(self, contract):
        self.contract = contract

    def _deny(
        self, reason, level, available, negative, history_level,
        quality, static=False,
    ):
        diagnostic = (
            SupportDiagnosticV1.STATIC_SUPPORT_DIAGNOSTIC
            if static else SupportDiagnosticV1.UNKNOWN_SUPPORT_DIAGNOSTIC
        )
        return ProvisionalEvidenceDecisionV1(
            False, level, reason, diagnostic, tuple(sorted(available)),
            tuple(sorted(set(self.families)-set(available))),
            tuple(sorted(negative)), history_level, quality,
        )

    def authorize(self, outcome: Mapping, causal_context=None):
        forbidden = FORBIDDEN_RUNTIME_KEYS.intersection(outcome)
        context = dict(causal_context or {})
        forbidden |= FORBIDDEN_RUNTIME_KEYS.intersection(context)
        if forbidden:
            raise ValueError(f"forbidden runtime keys: {sorted(forbidden)}")
        if outcome.get("status") != "BOUNDED_SAFETY_SUPPORT":
            return self._deny(
                "DENIED_NOT_BOUNDED_SUPPORT", EvidenceLevelV1.NO_EVIDENCE,
                (), ("NOT_BOUNDED_SUPPORT",), "NOT_APPLICABLE",
                "NOT_APPLICABLE",
            )
        support = outcome.get("support")
        bounds = _finite_bounds(support)
        quality_cfg = self.contract["support_quality"]
        if bounds is None:
            return self._deny(
                "DENIED_SUPPORT_QUALITY", EvidenceLevelV1.NO_EVIDENCE,
                (), ("NONFINITE_OR_UNBOUNDED_SUPPORT",), "NO_HISTORY",
                "INVALID",
            )
        point_count = int(support.get("valid_point_count", 0))
        diagonal = float(np.linalg.norm(bounds[1]-bounds[0]))
        if (
            point_count < int(quality_cfg["minimum_finite_points"])
            or diagonal
            > float(quality_cfg["maximum_position_set_diagonal_m"])
        ):
            return self._deny(
                "DENIED_SUPPORT_QUALITY", EvidenceLevelV1.NO_EVIDENCE,
                (), ("INSUFFICIENT_OR_UNBOUNDED_SUPPORT",),
                "NO_HISTORY", "INVALID",
            )

        fields = dict(context.get("fields") or {})
        history = dict(context.get("historical_context") or {})
        available = set()
        negative = set()
        if bool(fields.get("internal_contract_error", False)):
            negative.add("INTERNAL_CONTRACT_ERROR")
        if bool(fields.get("hard_invalid_no_evidence", False)):
            negative.add("HARD_INVALID_NO_EVIDENCE")
        if bool(fields.get("camera_motion_artifact", False)):
            negative.add("CAMERA_MOTION_ARTIFACT")
        if bool(fields.get("static_background_match", False)):
            negative.add("STATIC_BACKGROUND_MATCH")
        if bool(fields.get("association_conflict", False)):
            negative.add("ASSOCIATION_CONFLICT")
        if bool(fields.get("duplicate_source", False)):
            negative.add("DUPLICATE_SOURCE")
        hard = {
            "INTERNAL_CONTRACT_ERROR", "HARD_INVALID_NO_EVIDENCE",
            "CAMERA_MOTION_ARTIFACT", "STATIC_BACKGROUND_MATCH",
            "ASSOCIATION_CONFLICT", "DUPLICATE_SOURCE",
        }
        if negative.intersection(hard):
            return self._deny(
                "DENIED_NEGATIVE_VETO", EvidenceLevelV1.NO_EVIDENCE,
                available, negative, "NO_HISTORY", "VALID",
                static="STATIC_BACKGROUND_MATCH" in negative,
            )

        pixels = int(fields.get("pixel_count", point_count))
        tiny = pixels <= int(quality_cfg["tiny_support_max_points"])
        boundary = float(fields.get(
            "boundary_hazard_fraction", 1.0
        )) >= float(quality_cfg["high_boundary_hazard_fraction"])
        fov = float(fields.get(
            "fov_boundary_fraction", 0.0
        )) >= float(quality_cfg["high_fov_boundary_fraction"])
        provenance = float(fields.get(
            "temporal_provenance_invalid_fraction", 1.0
        )) >= float(quality_cfg["weak_provenance_fraction"])
        fragmented = _bbox_fill(fields) <= float(
            quality_cfg["low_component_fill_fraction"]
        )
        for flag, name in (
            (tiny, "TINY_SUPPORT"), (boundary, "HIGH_BOUNDARY_HAZARD"),
            (fov, "FOV_EDGE"), (provenance, "WEAK_DEPTH_PROVENANCE"),
            (fragmented, "HIGH_FRAGMENTATION"),
        ):
            if flag:
                negative.add(name)

        timestamp = float(outcome["timestamp"])
        history_cfg = self.contract["history"]
        history_exists = bool(history.get("track_exists", False))
        generation = history.get("generation")
        last_direct = history.get("last_direct_measurement_time")
        recent = (
            last_direct is not None
            and 0.0 <= timestamp-float(last_direct)
            <= float(history_cfg["maximum_direct_evidence_age_s"])
        )
        dynamic_basis = bool(
            history.get("dynamic", False)
            or (
                history.get("confirmed", False)
                and history.get("last_safety_evidence")
                in history_cfg["accepted_safety_evidence"]
            )
        )
        compatible = _bounds_compatible(
            bounds, history.get("last_valid_geometry_bounds"),
            float(history_cfg["maximum_position_gap_m"]),
        )
        history_authorized = bool(
            history_exists and generation is not None and recent
            and dynamic_basis and compatible
        )
        if history_exists and not history_authorized:
            history_level = "STALE_OR_INCOMPATIBLE_HISTORY"
            negative.add("STALE_OR_INCOMPATIBLE_HISTORY")
        elif history_authorized:
            history_level = "RECENT_DYNAMIC_TRACK_COMPATIBLE"
            available.add("HISTORICAL_DYNAMIC_CONTEXT")
        else:
            history_level = "NO_HISTORY"

        if history_authorized:
            return ProvisionalEvidenceDecisionV1(
                True, EvidenceLevelV1.RECENT_DYNAMIC_CONTEXT,
                "AUTHORIZED_RECENT_DYNAMIC_CONTEXT",
                SupportDiagnosticV1.AUTHORIZED_DYNAMIC_SUPPORT,
                tuple(sorted(available)),
                tuple(sorted(set(self.families)-available)),
                tuple(sorted(negative)), history_level, "FINITE_BOUNDED",
            )

        evidence_cfg = self.contract["evidence"]
        temporal = bool(
            int(fields.get("temporal_support", 0))
            >= int(evidence_cfg["minimum_temporal_support_frames"])
            and float(fields.get("stable_overlap_fraction", 0.0))
            >= float(evidence_cfg["minimum_stable_overlap_fraction"])
            and float(fields.get(
                "temporal_provenance_invalid_fraction", 1.0
            )) <= float(quality_cfg["weak_provenance_fraction"])
        )
        motion = bool(
            float(evidence_cfg["minimum_motion_speed_mps"])
            <= float(fields.get("world_speed_mps", 0.0))
            <= float(evidence_cfg["maximum_motion_speed_mps"])
            and float(fields.get("direction_consistency", -1.0))
            >= float(evidence_cfg["minimum_direction_consistency"])
        )
        approach = float(fields.get("closer_fraction", 0.0)) > float(
            evidence_cfg["minimum_approach_fraction"]
        )
        cross_source = bool(fields.get(
            "cross_source_dynamic_compatible", False
        ))
        for flag, family in (
            (temporal, "TEMPORAL_PERSISTENCE"),
            (motion, "MOTION_CONSISTENCY"),
            (approach, "APPROACH_OR_SCALE_TREND"),
            (cross_source, "CROSS_SOURCE_CORROBORATION"),
        ):
            if flag:
                available.add(family)

        level = (
            EvidenceLevelV1.NO_EVIDENCE if not available
            else EvidenceLevelV1.SINGLE_WEAK_CUE
        )
        if not temporal:
            return self._deny(
                "DENIED_INSUFFICIENT_INDEPENDENT_EVIDENCE", level,
                available, negative | {"NO_TEMPORAL_PERSISTENCE"},
                history_level, "FINITE_BOUNDED",
            )
        independent = {
            family for family in available
            if family != "HISTORICAL_DYNAMIC_CONTEXT"
        }
        quorum = int(
            evidence_cfg["no_history_minimum_independent_families"]
        )
        if len(independent) < quorum:
            reason = (
                "DENIED_SINGLE_WEAK_CUE"
                if independent == {"APPROACH_OR_SCALE_TREND"}
                else "DENIED_INSUFFICIENT_INDEPENDENT_EVIDENCE"
            )
            return self._deny(
                reason, EvidenceLevelV1.SINGLE_WEAK_CUE,
                available, negative, history_level, "FINITE_BOUNDED",
            )
        artifact_count = sum((
            tiny, boundary, fov, provenance, fragmented,
        ))
        combined_artifact_veto = bool(
            not cross_source
            and (
                (
                    tiny and boundary
                    and self.contract["no_history"][
                        "tiny_high_boundary_requires_history_or_cross_source"
                    ]
                )
                or (
                    artifact_count >= 2
                    and self.contract["no_history"][
                        "multiple_artifact_flags_require_cross_source"
                    ]
                )
                or (
                    artifact_count == 1
                    and len(independent) < int(
                        self.contract["no_history"][
                            "single_artifact_minimum_independent_families"
                        ]
                    )
                )
            )
        )
        if combined_artifact_veto:
            return self._deny(
                "DENIED_NEGATIVE_VETO",
                EvidenceLevelV1.CORROBORATED_NO_HISTORY,
                available,
                negative | {"TINY_HIGH_BOUNDARY_NO_CORROBORATION"},
                history_level, "CONDITIONAL_ARTIFACT_RISK",
            )
        return ProvisionalEvidenceDecisionV1(
            True, EvidenceLevelV1.CORROBORATED_NO_HISTORY,
            "AUTHORIZED_CORROBORATED_NO_HISTORY",
            SupportDiagnosticV1.AUTHORIZED_DYNAMIC_SUPPORT,
            tuple(sorted(available)),
            tuple(sorted(set(self.families)-available)),
            tuple(sorted(negative)), history_level, "FINITE_BOUNDED",
        )


__all__ = [
    "CONTRACT_VERSION", "FORBIDDEN_RUNTIME_KEYS", "EvidenceLevelV1",
    "SupportDiagnosticV1", "ProvisionalEvidenceDecisionV1",
    "ProvisionalEvidenceAuthorizerV1",
]
