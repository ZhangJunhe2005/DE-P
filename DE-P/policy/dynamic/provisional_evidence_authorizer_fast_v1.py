"""Allocation-reduced implementation of the frozen PECR1 decision."""

from __future__ import annotations

import math

from .provisional_evidence_context_v1 import ProvisionalEvidenceContextV1
from .provisional_evidence_authorizer_v1 import (
    EvidenceLevelV1, ProvisionalEvidenceDecisionV1,
    SupportDiagnosticV1,
)


CONTRACT_VERSION = "provisional_evidence_authorizer_fast_v1"

E_HISTORY, E_TEMPORAL, E_MOTION, E_APPROACH, E_CROSS = (
    1, 2, 4, 8, 16
)
EVIDENCE_NAMES = (
    (E_APPROACH, "APPROACH_OR_SCALE_TREND"),
    (E_CROSS, "CROSS_SOURCE_CORROBORATION"),
    (E_HISTORY, "HISTORICAL_DYNAMIC_CONTEXT"),
    (E_MOTION, "MOTION_CONSISTENCY"),
    (E_TEMPORAL, "TEMPORAL_PERSISTENCE"),
)
ALL_EVIDENCE = E_HISTORY | E_TEMPORAL | E_MOTION | E_APPROACH | E_CROSS

N_INTERNAL, N_HARD_INVALID, N_CAMERA, N_STATIC, N_ASSOC, N_DUP = (
    1, 2, 4, 8, 16, 32
)
N_TINY, N_BOUNDARY, N_FOV, N_PROVENANCE, N_FRAGMENTED, N_STALE = (
    64, 128, 256, 512, 1024, 2048
)
NEGATIVE_NAMES = (
    (N_ASSOC, "ASSOCIATION_CONFLICT"),
    (N_CAMERA, "CAMERA_MOTION_ARTIFACT"),
    (N_DUP, "DUPLICATE_SOURCE"),
    (N_FOV, "FOV_EDGE"),
    (N_FRAGMENTED, "HIGH_FRAGMENTATION"),
    (N_BOUNDARY, "HIGH_BOUNDARY_HAZARD"),
    (N_HARD_INVALID, "HARD_INVALID_NO_EVIDENCE"),
    (N_INTERNAL, "INTERNAL_CONTRACT_ERROR"),
    (N_PROVENANCE, "WEAK_DEPTH_PROVENANCE"),
    (N_STALE, "STALE_OR_INCOMPATIBLE_HISTORY"),
    (N_STATIC, "STATIC_BACKGROUND_MATCH"),
    (N_TINY, "TINY_SUPPORT"),
)
EXTRA_NEGATIVES = {
    "NOT_BOUNDED_SUPPORT": ("NOT_BOUNDED_SUPPORT",),
    "NONFINITE_OR_UNBOUNDED_SUPPORT":
        ("NONFINITE_OR_UNBOUNDED_SUPPORT",),
    "INSUFFICIENT_OR_UNBOUNDED_SUPPORT":
        ("INSUFFICIENT_OR_UNBOUNDED_SUPPORT",),
    "NO_TEMPORAL_PERSISTENCE": ("NO_TEMPORAL_PERSISTENCE",),
    "TINY_HIGH_BOUNDARY_NO_CORROBORATION":
        ("TINY_HIGH_BOUNDARY_NO_CORROBORATION",),
}


def _evidence(mask):
    return tuple(name for bit, name in EVIDENCE_NAMES if mask & bit)


def _unavailable(mask):
    return tuple(name for bit, name in EVIDENCE_NAMES if not mask & bit)


def _negative(mask, extra=None):
    values = [name for bit, name in NEGATIVE_NAMES if mask & bit]
    if extra:
        values.extend(EXTRA_NEGATIVES[extra])
    return tuple(sorted(values))


class ProvisionalEvidenceAuthorizerFastV1:
    families = tuple(name for _, name in EVIDENCE_NAMES)

    def __init__(self, contract, diagnostic_capacity=256):
        self.contract = contract
        self.diagnostic_capacity = max(1, int(diagnostic_capacity))
        self._diagnostics = [None] * self.diagnostic_capacity
        self._diagnostic_cursor = 0
        self.diagnostic_overflow_count = 0

    def _publish(self, decision):
        index = self._diagnostic_cursor % self.diagnostic_capacity
        if self._diagnostic_cursor >= self.diagnostic_capacity:
            self.diagnostic_overflow_count += 1
        self._diagnostics[index] = (
            decision.authorized, decision.reason_code,
            decision.evidence_level.value,
        )
        self._diagnostic_cursor += 1
        return decision

    def _decision(
        self, authorized, level, reason, evidence_mask, negative_mask,
        history, quality, static=False, extra=None,
    ):
        return self._publish(ProvisionalEvidenceDecisionV1(
            authorized, level, reason,
            (
                SupportDiagnosticV1.AUTHORIZED_DYNAMIC_SUPPORT
                if authorized else
                SupportDiagnosticV1.STATIC_SUPPORT_DIAGNOSTIC
                if static else SupportDiagnosticV1.UNKNOWN_SUPPORT_DIAGNOSTIC
            ),
            _evidence(evidence_mask), _unavailable(evidence_mask),
            _negative(negative_mask, extra), history, quality,
        ))

    @staticmethod
    def _compatible(current, historical, margin):
        if current is None or historical is None:
            return False
        return all(
            current[index] <= historical[index + 3] + margin
            and historical[index] <= current[index + 3] + margin
            for index in range(3)
        )

    def authorize(self, outcome, causal_context=None):
        context = (
            outcome if isinstance(outcome, ProvisionalEvidenceContextV1)
            else ProvisionalEvidenceContextV1.parse(
                outcome, causal_context
            )
        )
        if context.status != "BOUNDED_SAFETY_SUPPORT":
            return self._decision(
                False, EvidenceLevelV1.NO_EVIDENCE,
                "DENIED_NOT_BOUNDED_SUPPORT", 0, 0,
                "NOT_APPLICABLE", "NOT_APPLICABLE",
                extra="NOT_BOUNDED_SUPPORT",
            )
        bounds = context.support_bounds
        quality = self.contract["support_quality"]
        if bounds is None:
            return self._decision(
                False, EvidenceLevelV1.NO_EVIDENCE,
                "DENIED_SUPPORT_QUALITY", 0, 0, "NO_HISTORY",
                "INVALID", extra="NONFINITE_OR_UNBOUNDED_SUPPORT",
            )
        diagonal = math.sqrt(sum(
            (bounds[index + 3] - bounds[index]) ** 2
            for index in range(3)
        ))
        if (
            context.valid_point_count < int(quality["minimum_finite_points"])
            or diagonal > float(quality["maximum_position_set_diagonal_m"])
        ):
            return self._decision(
                False, EvidenceLevelV1.NO_EVIDENCE,
                "DENIED_SUPPORT_QUALITY", 0, 0, "NO_HISTORY",
                "INVALID", extra="INSUFFICIENT_OR_UNBOUNDED_SUPPORT",
            )
        negative = (
            N_INTERNAL if context.internal_contract_error else 0
        ) | (N_HARD_INVALID if context.hard_invalid_no_evidence else 0) | (
            N_CAMERA if context.camera_motion_artifact else 0
        ) | (N_STATIC if context.static_background_match else 0) | (
            N_ASSOC if context.association_conflict else 0
        ) | (N_DUP if context.duplicate_source else 0)
        if negative:
            return self._decision(
                False, EvidenceLevelV1.NO_EVIDENCE,
                "DENIED_NEGATIVE_VETO", 0, negative, "NO_HISTORY",
                "VALID", static=bool(negative & N_STATIC),
            )
        tiny = context.pixel_count <= int(quality["tiny_support_max_points"])
        boundary = context.boundary_hazard_fraction >= float(
            quality["high_boundary_hazard_fraction"]
        )
        fov = context.fov_boundary_fraction >= float(
            quality["high_fov_boundary_fraction"]
        )
        provenance = (
            context.temporal_provenance_invalid_fraction
            >= float(quality["weak_provenance_fraction"])
        )
        fragmented = context.bbox_fill <= float(
            quality["low_component_fill_fraction"]
        )
        negative = (
            (N_TINY if tiny else 0) | (N_BOUNDARY if boundary else 0)
            | (N_FOV if fov else 0) | (N_PROVENANCE if provenance else 0)
            | (N_FRAGMENTED if fragmented else 0)
        )
        history = self.contract["history"]
        last = context.last_direct_measurement_time
        recent = (
            last is not None and
            0.0 <= context.timestamp-float(last)
            <= float(history["maximum_direct_evidence_age_s"])
        )
        dynamic_basis = (
            context.dynamic or (
                context.confirmed and context.last_safety_evidence
                in history["accepted_safety_evidence"]
            )
        )
        compatible = self._compatible(
            bounds, context.historical_bounds,
            float(history["maximum_position_gap_m"]),
        )
        authorized_history = (
            context.track_exists and context.generation is not None
            and recent and dynamic_basis and compatible
        )
        evidence = E_HISTORY if authorized_history else 0
        if context.track_exists and not authorized_history:
            history_level = "STALE_OR_INCOMPATIBLE_HISTORY"
            negative |= N_STALE
        elif authorized_history:
            history_level = "RECENT_DYNAMIC_TRACK_COMPATIBLE"
        else:
            history_level = "NO_HISTORY"
        if authorized_history:
            return self._decision(
                True, EvidenceLevelV1.RECENT_DYNAMIC_CONTEXT,
                "AUTHORIZED_RECENT_DYNAMIC_CONTEXT", evidence, negative,
                history_level, "FINITE_BOUNDED",
            )
        config = self.contract["evidence"]
        temporal = (
            context.temporal_support
            >= int(config["minimum_temporal_support_frames"])
            and context.stable_overlap_fraction
            >= float(config["minimum_stable_overlap_fraction"])
            and context.temporal_provenance_invalid_fraction
            <= float(quality["weak_provenance_fraction"])
        )
        motion = (
            float(config["minimum_motion_speed_mps"])
            <= context.world_speed_mps
            <= float(config["maximum_motion_speed_mps"])
            and context.direction_consistency
            >= float(config["minimum_direction_consistency"])
        )
        approach = context.closer_fraction > float(
            config["minimum_approach_fraction"]
        )
        cross = context.cross_source_dynamic_compatible
        evidence |= (
            (E_TEMPORAL if temporal else 0)
            | (E_MOTION if motion else 0)
            | (E_APPROACH if approach else 0)
            | (E_CROSS if cross else 0)
        )
        level = (
            EvidenceLevelV1.NO_EVIDENCE
            if evidence == 0 else EvidenceLevelV1.SINGLE_WEAK_CUE
        )
        if not temporal:
            return self._decision(
                False, level, "DENIED_INSUFFICIENT_INDEPENDENT_EVIDENCE",
                evidence, negative, history_level, "FINITE_BOUNDED",
                extra="NO_TEMPORAL_PERSISTENCE",
            )
        count = int(bool(evidence & E_TEMPORAL)) + int(
            bool(evidence & E_MOTION)
        ) + int(bool(evidence & E_APPROACH)) + int(bool(evidence & E_CROSS))
        if count < int(config["no_history_minimum_independent_families"]):
            reason = (
                "DENIED_SINGLE_WEAK_CUE"
                if evidence == E_APPROACH
                else "DENIED_INSUFFICIENT_INDEPENDENT_EVIDENCE"
            )
            return self._decision(
                False, EvidenceLevelV1.SINGLE_WEAK_CUE, reason, evidence,
                negative, history_level, "FINITE_BOUNDED",
            )
        artifacts = sum((tiny, boundary, fov, provenance, fragmented))
        no_history = self.contract["no_history"]
        veto = (
            not cross and (
                tiny and boundary and no_history[
                    "tiny_high_boundary_requires_history_or_cross_source"
                ] or artifacts >= 2 and no_history[
                    "multiple_artifact_flags_require_cross_source"
                ] or artifacts == 1 and count < int(no_history[
                    "single_artifact_minimum_independent_families"
                ])
            )
        )
        if veto:
            return self._decision(
                False, EvidenceLevelV1.CORROBORATED_NO_HISTORY,
                "DENIED_NEGATIVE_VETO", evidence, negative, history_level,
                "CONDITIONAL_ARTIFACT_RISK",
                extra="TINY_HIGH_BOUNDARY_NO_CORROBORATION",
            )
        return self._decision(
            True, EvidenceLevelV1.CORROBORATED_NO_HISTORY,
            "AUTHORIZED_CORROBORATED_NO_HISTORY", evidence, negative,
            history_level, "FINITE_BOUNDED",
        )

    def authorize_batch(self, rows):
        return tuple(
            self.authorize(outcome, context)
            for outcome, context in rows
        )


__all__ = [
    "CONTRACT_VERSION", "ProvisionalEvidenceAuthorizerFastV1",
    "EVIDENCE_NAMES", "NEGATIVE_NAMES",
]
