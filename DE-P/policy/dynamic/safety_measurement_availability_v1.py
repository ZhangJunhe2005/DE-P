"""Development-only strict/safety measurement separation for MAR1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple

import numpy as np


CONTRACT_VERSION = "safety_measurement_availability_v1"


class EvidenceStateV1(str, Enum):
    AVAILABLE_TRUE = "AVAILABLE_TRUE"
    AVAILABLE_FALSE = "AVAILABLE_FALSE"
    UNAVAILABLE = "UNAVAILABLE"


class AvailabilityModeV1(str, Enum):
    MEASUREMENT_INITIALIZING = "MEASUREMENT_INITIALIZING"
    CAUSAL_MOTION_SUPPORT = "CAUSAL_MOTION_SUPPORT"
    CAUSAL_APPROACH_SUPPORT = "CAUSAL_APPROACH_SUPPORT"
    FRAGMENTED_SAFETY_SUPPORT = "FRAGMENTED_SAFETY_SUPPORT"
    SUPPORT_ONLY = "SUPPORT_ONLY"
    UNRESOLVED_MEASUREMENT_RISK = "UNRESOLVED_MEASUREMENT_RISK"


@dataclass(frozen=True)
class SafetyMeasurementEvidenceV1:
    finite_geometry: EvidenceStateV1
    valid_depth_support: EvidenceStateV1
    temporal_persistence: EvidenceStateV1
    stable_pixel_overlap: EvidenceStateV1
    consistent_world_motion: EvidenceStateV1
    depth_approach_trend: EvidenceStateV1
    direction_consistency: EvidenceStateV1
    boundary_safe: EvidenceStateV1
    fov_safe: EvidenceStateV1
    static_conflict_absent: EvidenceStateV1

    def true_count(self):
        return sum(
            value == EvidenceStateV1.AVAILABLE_TRUE
            for value in self.__dict__.values()
        )


def _finite(name, value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape {shape}")
    output = array.copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class SafetyWeakMeasurementV1:
    safety_measurement_id: int
    frame_index: int
    timestamp: float
    source_component_ids: Tuple[int, ...]
    source_component_provenance: str
    strict_rejection_reasons: Tuple[str, ...]
    availability_mode: AvailabilityModeV1
    position_world: np.ndarray
    covariance_world: np.ndarray
    support_radius_m: float
    extent_interval_m: Tuple[float, float]
    velocity_center_mps: np.ndarray
    velocity_uncertainty_mps: float
    acceleration_bound_mps2: float
    pixel_bbox: Tuple[int, int, int, int]
    point_count: int
    boundary_hazard_fraction: float
    evidence: SafetyMeasurementEvidenceV1
    expiry_timestamp: float
    formal_eligible: bool = False
    provisional_eligible: bool = True
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.safety_measurement_id < 0 or self.frame_index < 0:
            raise ValueError("measurement/frame IDs must be non-negative")
        if not self.source_component_ids:
            raise ValueError("component provenance is required")
        if not self.strict_rejection_reasons:
            raise ValueError("strict rejection reasons are required")
        if self.formal_eligible:
            raise ValueError("weak measurement cannot be formal eligible")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if (
            not np.isfinite((self.timestamp, self.expiry_timestamp))
            .all()
            or self.expiry_timestamp <= self.timestamp
        ):
            raise ValueError("weak lifetime must be finite and positive")
        if self.support_radius_m <= 0 or self.point_count <= 0:
            raise ValueError("bounded support must be positive")
        if (
            len(self.extent_interval_m) != 2
            or self.extent_interval_m[0] < 0
            or self.extent_interval_m[1]
            < self.extent_interval_m[0]
        ):
            raise ValueError("extent interval is invalid")
        if not 0 <= self.boundary_hazard_fraction <= 1:
            raise ValueError("boundary hazard must be in [0,1]")
        object.__setattr__(
            self, "position_world",
            _finite("position_world", self.position_world, (3,)),
        )
        covariance = _finite(
            "covariance_world", self.covariance_world, (3, 3)
        )
        if not np.allclose(covariance, covariance.T, atol=1e-9):
            raise ValueError("covariance must be symmetric")
        object.__setattr__(self, "covariance_world", covariance)
        object.__setattr__(
            self, "velocity_center_mps",
            _finite("velocity_center_mps",
                    self.velocity_center_mps, (3,)),
        )
        object.__setattr__(
            self, "source_component_ids",
            tuple(int(value) for value in self.source_component_ids),
        )
        object.__setattr__(
            self, "strict_rejection_reasons",
            tuple(str(value) for value in self.strict_rejection_reasons),
        )
        object.__setattr__(
            self, "pixel_bbox",
            tuple(int(value) for value in self.pixel_bbox),
        )


@dataclass(frozen=True)
class SafetyMeasurementDecisionV1:
    classification: str
    reasons: Tuple[str, ...]
    measurement: SafetyWeakMeasurementV1 | None
    evidence: SafetyMeasurementEvidenceV1
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.classification not in {
            "CONDITIONAL_SALVAGE", "HARD_REJECT", "AUDIT_REQUIRED",
        }:
            raise ValueError("unknown salvage classification")


__all__ = [
    "CONTRACT_VERSION", "EvidenceStateV1", "AvailabilityModeV1",
    "SafetyMeasurementEvidenceV1", "SafetyWeakMeasurementV1",
    "SafetyMeasurementDecisionV1",
]
