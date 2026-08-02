"""Development-only consumer proving that non-measurement risk stays explicit."""

from __future__ import annotations

from dataclasses import dataclass

from .dynamic_measurement_outcome_v1 import (
    DynamicMeasurementOutcomeV1, DynamicMeasurementStatusV1,
)


CONTRACT_VERSION = "measurement_contract_shadow_consumer_v1"


@dataclass(frozen=True)
class ShadowRiskDecisionV1:
    semantic: str
    risk_present: bool
    support_bounds: object = None
    formal_router_called: bool = False


def consume_outcome(outcome: DynamicMeasurementOutcomeV1):
    if not isinstance(outcome, DynamicMeasurementOutcomeV1):
        raise TypeError("expected DynamicMeasurementOutcomeV1")
    status = outcome.status
    if status in {
        DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT,
        DynamicMeasurementStatusV1.VALID_SAFETY_WEAK_MEASUREMENT,
    }:
        semantic = "ACTIVE_MEASUREMENT_RISK"
    elif status == DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
        semantic = "ACTIVE_BOUNDED_SUPPORT"
    elif status in {
        DynamicMeasurementStatusV1.UNRESOLVED_MEASUREMENT_RISK,
        DynamicMeasurementStatusV1.INTERNAL_CONTRACT_ERROR,
    }:
        semantic = "FAIL_CLOSED_REVIEW_REQUIRED"
    elif status == DynamicMeasurementStatusV1.SENSOR_CONTRACT_LIMIT:
        semantic = "SENSOR_LIMIT_REVIEW_REQUIRED"
    elif status == DynamicMeasurementStatusV1.HARD_INVALID_NO_EVIDENCE:
        semantic = "NO_CURRENT_MEASUREMENT_RISK"
    else:
        raise RuntimeError("unhandled measurement outcome")
    return ShadowRiskDecisionV1(
        semantic=semantic,
        risk_present=outcome.risk_present,
        support_bounds=outcome.support_bounds,
    )


__all__ = [
    "CONTRACT_VERSION", "ShadowRiskDecisionV1", "consume_outcome",
]
