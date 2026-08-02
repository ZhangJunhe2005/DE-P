"""Offline-only four-class label authority with explicit ambiguity."""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "dynamic_evidence_label_authority_v1"
LABELS = (
    "DYNAMIC_SUPPORT", "STATIC_SUPPORT",
    "SENSOR_OR_SEGMENTATION_ARTIFACT", "UNKNOWN_AMBIGUOUS",
)


@dataclass(frozen=True, slots=True)
class LabelAuthorityResultV1:
    label: str
    authority: tuple[str, ...]
    confidence: str
    ambiguity_reasons: tuple[str, ...]
    source_actor_count: int
    static_overlap: bool
    dynamic_overlap: bool
    invalid_fraction: float
    mixed_ownership: bool


def assign_offline_label(
    *, dynamic_overlap: bool, static_overlap: bool,
    actor_count: int, renderer_valid: bool,
    segmentation_artifact: bool, authority_conflict: bool = False,
    invalid_fraction: float = 0.0,
):
    reasons = []
    mixed = dynamic_overlap and static_overlap
    if mixed:
        reasons.append("MIXED_STATIC_DYNAMIC_OWNERSHIP")
    if actor_count > 1:
        reasons.append("MULTI_ACTOR_SUPPORT")
    if authority_conflict:
        reasons.append("AUTHORITY_CONFLICT")
    if not renderer_valid:
        reasons.append("RENDERER_OR_DEPTH_INVALID")
    if reasons:
        label, confidence = "UNKNOWN_AMBIGUOUS", "AMBIGUOUS"
    elif dynamic_overlap and actor_count == 1:
        label, confidence = "DYNAMIC_SUPPORT", "AUTHORITATIVE"
    elif static_overlap:
        label, confidence = "STATIC_SUPPORT", "AUTHORITATIVE"
    elif segmentation_artifact:
        label, confidence = (
            "SENSOR_OR_SEGMENTATION_ARTIFACT", "AUTHORITATIVE"
        )
    else:
        label, confidence = "UNKNOWN_AMBIGUOUS", "INSUFFICIENT"
        reasons.append("NO_EXCLUSIVE_AUTHORITY")
    return LabelAuthorityResultV1(
        label, (
            "DYNAMIC_ACTOR_RENDERING_AUTHORITY",
            "STATIC_MAP_AUTHORITY", "SENSOR_RENDERER_CONTRACT",
        ), confidence, tuple(reasons), int(actor_count),
        bool(static_overlap), bool(dynamic_overlap),
        float(invalid_fraction), mixed,
    )


__all__ = [
    "CONTRACT_VERSION", "LABELS", "LabelAuthorityResultV1",
    "assign_offline_label",
]
