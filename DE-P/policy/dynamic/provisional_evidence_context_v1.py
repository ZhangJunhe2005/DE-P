"""Immutable, scalar-only input for the PECR1 evidence decision."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .provisional_evidence_authorizer_v1 import FORBIDDEN_RUNTIME_KEYS


CONTRACT_VERSION = "provisional_evidence_context_v1"


def _bounds(value):
    if not isinstance(value, Mapping):
        return None
    try:
        lower = value["position_set_min_world"]
        upper = value["position_set_max_world"]
        result = tuple(float(item) for item in (*lower, *upper))
    except (KeyError, TypeError, ValueError):
        return None
    if len(result) != 6 or not all(math.isfinite(item) for item in result):
        return None
    if any(result[index + 3] < result[index] for index in range(3)):
        return None
    return result


def _historical_bounds(value):
    try:
        lower, upper = value
        result = tuple(float(item) for item in (*lower, *upper))
    except (TypeError, ValueError):
        return None
    if len(result) != 6 or not all(math.isfinite(item) for item in result):
        return None
    return result


@dataclass(frozen=True, slots=True)
class ProvisionalEvidenceContextV1:
    status: object
    timestamp: float
    support_bounds: tuple[float, ...] | None
    valid_point_count: int
    pixel_count: int
    bbox_fill: float
    internal_contract_error: bool
    hard_invalid_no_evidence: bool
    camera_motion_artifact: bool
    static_background_match: bool
    association_conflict: bool
    duplicate_source: bool
    boundary_hazard_fraction: float
    fov_boundary_fraction: float
    temporal_provenance_invalid_fraction: float
    temporal_support: int
    stable_overlap_fraction: float
    world_speed_mps: float
    direction_consistency: float
    closer_fraction: float
    cross_source_dynamic_compatible: bool
    track_exists: bool
    generation: object
    dynamic: bool
    confirmed: bool
    last_direct_measurement_time: object
    last_safety_evidence: object
    historical_bounds: tuple[float, ...] | None

    @classmethod
    def parse(cls, outcome: Mapping, causal_context=None):
        context = causal_context or {}
        if not isinstance(context, Mapping):
            context = dict(context)
        forbidden = (
            FORBIDDEN_RUNTIME_KEYS.intersection(outcome)
            | FORBIDDEN_RUNTIME_KEYS.intersection(context)
        )
        if forbidden:
            raise ValueError(f"forbidden runtime keys: {sorted(forbidden)}")
        fields = context.get("fields") or {}
        history = context.get("historical_context") or {}
        support = outcome.get("support")
        bounds = _bounds(support)
        points = int(support.get("valid_point_count", 0)) if isinstance(
            support, Mapping
        ) else 0
        bbox = fields.get("pixel_bbox")
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            width = max(1, int(bbox[2])-int(bbox[0])+1)
            height = max(1, int(bbox[3])-int(bbox[1])+1)
            fill = float(fields.get("pixel_count", 0))/(width*height)
        else:
            fill = 0.0
        return cls(
            outcome.get("status"), float(outcome.get("timestamp", 0.0)),
            bounds, points,
            int(fields.get("pixel_count", points)), fill,
            bool(fields.get("internal_contract_error", False)),
            bool(fields.get("hard_invalid_no_evidence", False)),
            bool(fields.get("camera_motion_artifact", False)),
            bool(fields.get("static_background_match", False)),
            bool(fields.get("association_conflict", False)),
            bool(fields.get("duplicate_source", False)),
            float(fields.get("boundary_hazard_fraction", 1.0)),
            float(fields.get("fov_boundary_fraction", 0.0)),
            float(fields.get("temporal_provenance_invalid_fraction", 1.0)),
            int(fields.get("temporal_support", 0)),
            float(fields.get("stable_overlap_fraction", 0.0)),
            float(fields.get("world_speed_mps", 0.0)),
            float(fields.get("direction_consistency", -1.0)),
            float(fields.get("closer_fraction", 0.0)),
            bool(fields.get("cross_source_dynamic_compatible", False)),
            bool(history.get("track_exists", False)),
            history.get("generation"), bool(history.get("dynamic", False)),
            bool(history.get("confirmed", False)),
            history.get("last_direct_measurement_time"),
            history.get("last_safety_evidence"),
            _historical_bounds(history.get("last_valid_geometry_bounds")),
        )


__all__ = ["CONTRACT_VERSION", "ProvisionalEvidenceContextV1"]
