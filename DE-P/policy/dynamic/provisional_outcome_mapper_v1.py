"""Frozen outcome-to-provisional-state mapping for development shadow use."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .dynamic_measurement_outcome_v1 import DynamicMeasurementStatusV1
from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateTypeV1,
)


CONTRACT_VERSION = "provisional_outcome_mapper_v1"
FORBIDDEN_KEYS = frozenset({
    "gt_actor_id", "owner_map", "future_frame", "future_information",
})


class ProvisionalOutcomeMapperV1:
    def __init__(self, contract):
        self.contract = contract

    def map(
        self, outcome: Mapping, state_id, generation=0,
        causal_context: Mapping | None = None,
    ):
        forbidden = FORBIDDEN_KEYS.intersection(outcome)
        if forbidden:
            raise ValueError(f"forbidden runtime keys: {sorted(forbidden)}")
        status = DynamicMeasurementStatusV1(outcome["status"])
        timestamp = float(outcome["timestamp"])
        lifecycle = self.contract["lifecycle"]
        common = {
            "state_id": int(state_id), "generation": int(generation),
            "source_outcome_ids": (int(outcome["outcome_id"]),),
            "source_component_ids": tuple(
                int(value)
                for value in outcome.get("source_component_ids", ())
            ),
            "source_observation_ids": (),
            "birth_timestamp": timestamp,
            "last_evidence_timestamp": timestamp,
            "missed_frames": 0,
            "maximum_missed_frames":
                lifecycle["maximum_missed_frames"],
            "evidence_frames": 1,
            "formal_tracker_feed": False,
            "runtime_gt_used": False,
        }
        if status == DynamicMeasurementStatusV1.HARD_INVALID_NO_EVIDENCE:
            return None
        if status == DynamicMeasurementStatusV1.VALID_STRICT_MEASUREMENT:
            # Confirmed strict measurements remain on the formal path. An
            # unconfirmed strict measurement must use the explicit bridge.
            return None
        if status == (
            DynamicMeasurementStatusV1
            .VALID_SAFETY_WEAK_MEASUREMENT
        ):
            velocity = outcome.get("velocity_center_mps")
            uncertainty = max(
                .25, float(self.contract["motion"]["maximum_speed_mps"])
                if velocity is None else .25
            )
            return ProvisionalDynamicSafetyStateV1(
                **common,
                state_type=(
                    ProvisionalSafetyStateTypeV1
                    .PROVISIONAL_MEASUREMENT_STATE
                ),
                expiry_timestamp=(
                    timestamp+lifecycle["measurement_max_age_s"]
                ),
                position_reference=outcome["position_reference"],
                position_world=np.asarray(
                    outcome["position_world"], np.float64
                ),
                covariance_world=np.asarray(
                    outcome["covariance_world"], np.float64
                ),
                velocity_center_mps=(
                    np.zeros(3) if velocity is None
                    else np.asarray(velocity, np.float64)
                ),
                velocity_uncertainty_mps=uncertainty,
                acceleration_bound_mps2=self.contract[
                    "motion"]["maximum_acceleration_mps2"],
                planner_semantic="ACTIVE_PROVISIONAL_RISK",
            )
        if status == DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
            context = dict(causal_context or {})
            fields = dict(context.get("fields") or {})
            history = dict(context.get("historical_context") or {})
            authorized = bool(
                float(fields.get("closer_fraction", 0.0)) > 0.0
                or history.get("track_exists", False)
            )
            if (
                self.contract["support_birth"][
                    "require_dynamic_safety_context"
                ]
                and not authorized
            ):
                return None
            support = outcome["support"]
            return ProvisionalDynamicSafetyStateV1(
                **common,
                state_type=(
                    ProvisionalSafetyStateTypeV1
                    .PROVISIONAL_SUPPORT_STATE
                ),
                expiry_timestamp=(
                    timestamp+lifecycle["support_max_age_s"]
                ),
                position_reference="FINITE_WORLD_POSITION_SET",
                support_bounds={
                    "min_world": tuple(
                        support["position_set_min_world"]
                    ),
                    "max_world": tuple(
                        support["position_set_max_world"]
                    ),
                },
                planner_semantic="ACTIVE_SUPPORT_RISK",
            )
        if status in {
            DynamicMeasurementStatusV1.UNRESOLVED_MEASUREMENT_RISK,
            DynamicMeasurementStatusV1.SENSOR_CONTRACT_LIMIT,
        }:
            evidence = outcome.get("unresolved_risk") or {}
            source = tuple(evidence.get(
                "source_evidence", ("SENSOR_CONTRACT_LIMIT",)
            ))
            if not outcome.get("risk_present", False):
                return None
            return ProvisionalDynamicSafetyStateV1(
                **common,
                state_type=(
                    ProvisionalSafetyStateTypeV1
                    .PROVISIONAL_UNRESOLVED_STATE
                ),
                expiry_timestamp=(
                    timestamp+lifecycle["unresolved_max_age_s"]
                ),
                position_reference="NONE",
                unresolved_evidence=source,
                planner_semantic="UNRESOLVED_DYNAMIC_RISK",
            )
        if status == DynamicMeasurementStatusV1.INTERNAL_CONTRACT_ERROR:
            return ProvisionalDynamicSafetyStateV1(
                **common,
                state_type=ProvisionalSafetyStateTypeV1.INVALID,
                expiry_timestamp=(
                    timestamp+lifecycle["unresolved_max_age_s"]
                ),
                position_reference="NONE",
                planner_semantic="INVALID_EVALUATION",
            )
        raise RuntimeError("unhandled outcome status")


__all__ = [
    "CONTRACT_VERSION", "FORBIDDEN_KEYS",
    "ProvisionalOutcomeMapperV1",
]
