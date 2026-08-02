"""Explicit separation of measurement, bounded support, and unresolved risk."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np

from .near_field_safety_support_v1 import NearFieldSafetySupportV1


CONTRACT_VERSION = "dynamic_measurement_outcome_v1"


class DynamicMeasurementStatusV1(str, Enum):
    VALID_STRICT_MEASUREMENT = "VALID_STRICT_MEASUREMENT"
    VALID_SAFETY_WEAK_MEASUREMENT = "VALID_SAFETY_WEAK_MEASUREMENT"
    BOUNDED_SAFETY_SUPPORT = "BOUNDED_SAFETY_SUPPORT"
    UNRESOLVED_MEASUREMENT_RISK = "UNRESOLVED_MEASUREMENT_RISK"
    HARD_INVALID_NO_EVIDENCE = "HARD_INVALID_NO_EVIDENCE"
    SENSOR_CONTRACT_LIMIT = "SENSOR_CONTRACT_LIMIT"
    INTERNAL_CONTRACT_ERROR = "INTERNAL_CONTRACT_ERROR"


@dataclass(frozen=True)
class UnresolvedMeasurementRiskV1:
    source_evidence: Tuple[str, ...]
    first_timestamp: float
    last_timestamp: float
    track_context: Any
    depth_contract_state: str
    reason_code: str
    expiry_timestamp: float
    recommended_planner_semantic: str = (
        "FAIL_CLOSED_REVIEW_REQUIRED"
    )
    runtime_gt_used: bool = False

    def __post_init__(self):
        if not self.source_evidence:
            raise ValueError("unresolved risk requires causal evidence")
        if (
            self.runtime_gt_used
            or not np.isfinite((
                self.first_timestamp, self.last_timestamp,
                self.expiry_timestamp,
            )).all()
            or self.last_timestamp < self.first_timestamp
            or self.expiry_timestamp <= self.last_timestamp
        ):
            raise ValueError("unresolved risk lifetime is invalid")
        if self.recommended_planner_semantic != (
            "FAIL_CLOSED_REVIEW_REQUIRED"
        ):
            raise ValueError("unresolved risk must fail closed")


@dataclass(frozen=True)
class DynamicMeasurementOutcomeV1:
    outcome_id: int
    frame_index: int
    timestamp: float
    status: DynamicMeasurementStatusV1
    resolution_status: str
    source_component_ids: Tuple[int, ...] = ()
    strict_rejection_reasons: Tuple[str, ...] = ()
    position_reference: str = "NONE"
    position_world: Optional[np.ndarray] = None
    covariance_world: Optional[np.ndarray] = None
    velocity_center_mps: Optional[np.ndarray] = None
    shape: Any = None
    support: Optional[NearFieldSafetySupportV1] = None
    unresolved_risk: Optional[UnresolvedMeasurementRiskV1] = None
    position_valid: bool = False
    velocity_valid: bool = False
    shape_valid: bool = False
    support_valid: bool = False
    measurement_valid: bool = False
    risk_present: bool = False
    formal_eligible: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.outcome_id < 0 or self.frame_index < 0:
            raise ValueError("outcome/frame IDs must be non-negative")
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        measurement_states = {
            DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT,
            DynamicMeasurementStatusV1.VALID_SAFETY_WEAK_MEASUREMENT,
        }
        non_center_states = {
            DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT,
            DynamicMeasurementStatusV1.UNRESOLVED_MEASUREMENT_RISK,
            DynamicMeasurementStatusV1.SENSOR_CONTRACT_LIMIT,
            DynamicMeasurementStatusV1.HARD_INVALID_NO_EVIDENCE,
            DynamicMeasurementStatusV1.INTERNAL_CONTRACT_ERROR,
        }
        if self.status in measurement_states:
            if not self.measurement_valid or not self.position_valid:
                raise ValueError("measurement state requires valid position")
            if self.position_world is None or self.covariance_world is None:
                raise ValueError("measurement geometry is required")
        elif self.measurement_valid:
            raise ValueError("non-measurement state cannot be valid measurement")
        if self.status in non_center_states and any((
            self.position_valid, self.velocity_valid, self.shape_valid,
            self.position_world is not None,
            self.covariance_world is not None,
            self.velocity_center_mps is not None,
            self.shape is not None,
        )):
            raise ValueError("support/risk states cannot fake geometry")
        if self.status == DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
            if not self.support_valid or self.support is None:
                raise ValueError("bounded support state requires support")
        elif self.support_valid or self.support is not None:
            raise ValueError("support is exclusive to bounded support state")
        if self.status == (
            DynamicMeasurementStatusV1.UNRESOLVED_MEASUREMENT_RISK
        ):
            if self.unresolved_risk is None or not self.risk_present:
                raise ValueError("unresolved state requires explicit risk")
        elif self.unresolved_risk is not None:
            raise ValueError("unresolved payload has wrong state")
        if self.status == (
            DynamicMeasurementStatusV1.HARD_INVALID_NO_EVIDENCE
        ) and self.risk_present:
            raise ValueError("no-evidence state cannot assert risk")
        if self.formal_eligible and self.status != (
            DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT
        ):
            raise ValueError("only strict measurement may be formal eligible")
        if self.position_world is not None:
            position = np.asarray(self.position_world, dtype=np.float64)
            covariance = np.asarray(
                self.covariance_world, dtype=np.float64
            )
            if (
                position.shape != (3,) or covariance.shape != (3, 3)
                or not np.isfinite(position).all()
                or not np.isfinite(covariance).all()
            ):
                raise ValueError("measurement geometry is invalid")
            position = position.copy()
            covariance = covariance.copy()
            position.setflags(write=False)
            covariance.setflags(write=False)
            object.__setattr__(self, "position_world", position)
            object.__setattr__(self, "covariance_world", covariance)
        if self.velocity_center_mps is not None:
            velocity = np.asarray(
                self.velocity_center_mps, dtype=np.float64
            )
            if velocity.shape != (3,) or not np.isfinite(velocity).all():
                raise ValueError("velocity is invalid")
            velocity = velocity.copy()
            velocity.setflags(write=False)
            object.__setattr__(self, "velocity_center_mps", velocity)

    @property
    def support_bounds(self):
        return None if self.support is None else self.support.support_bounds


__all__ = [
    "CONTRACT_VERSION", "DynamicMeasurementStatusV1",
    "DynamicMeasurementOutcomeV1", "UnresolvedMeasurementRiskV1",
]
