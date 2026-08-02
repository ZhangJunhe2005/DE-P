"""Evidence-authorized wrapper around the frozen PDSCR1 outcome mapper."""

from __future__ import annotations

from typing import Mapping

from .dynamic_measurement_outcome_v1 import DynamicMeasurementStatusV1
from .provisional_evidence_authorizer_v1 import (
    ProvisionalEvidenceAuthorizerV1,
)
from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateTypeV1,
)
from .provisional_outcome_mapper_v1 import ProvisionalOutcomeMapperV1
from .provisional_support_birth_contract_v1 import support_birth_contract


CONTRACT_VERSION = "provisional_outcome_mapper_v3"


class ProvisionalOutcomeMapperV3:
    """Change only bounded-support admission; preserve every other mapping."""

    def __init__(self, pds_contract, evidence_contract):
        self.frozen_mapper = ProvisionalOutcomeMapperV1(pds_contract)
        self.authorizer = ProvisionalEvidenceAuthorizerV1(
            evidence_contract
        )
        self.last_authorization = None
        self.last_birth_contract = None

    def map(
        self, outcome: Mapping, state_id, generation=0,
        causal_context: Mapping | None = None,
    ):
        status = DynamicMeasurementStatusV1(outcome["status"])
        if status != DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
            self.last_authorization = None
            self.last_birth_contract = None
            return self.frozen_mapper.map(
                outcome, state_id, generation, causal_context
            )
        decision = self.authorizer.authorize(
            outcome, causal_context=causal_context
        )
        contract = support_birth_contract(decision)
        self.last_authorization = decision
        self.last_birth_contract = contract
        if not contract.state_birth_allowed:
            return None
        timestamp = float(outcome["timestamp"])
        support = outcome["support"]
        lifecycle = self.frozen_mapper.contract["lifecycle"]
        return ProvisionalDynamicSafetyStateV1(
            state_id=int(state_id), generation=int(generation),
            state_type=(
                ProvisionalSafetyStateTypeV1.PROVISIONAL_SUPPORT_STATE
            ),
            source_outcome_ids=(int(outcome["outcome_id"]),),
            source_component_ids=tuple(
                int(value)
                for value in outcome.get("source_component_ids", ())
            ),
            source_observation_ids=(),
            birth_timestamp=timestamp,
            last_evidence_timestamp=timestamp,
            expiry_timestamp=(
                timestamp+lifecycle["support_max_age_s"]
            ),
            missed_frames=0,
            maximum_missed_frames=lifecycle["maximum_missed_frames"],
            evidence_frames=1,
            position_reference="FINITE_WORLD_POSITION_SET",
            support_bounds={
                "min_world": tuple(support["position_set_min_world"]),
                "max_world": tuple(support["position_set_max_world"]),
            },
            planner_semantic="ACTIVE_SUPPORT_RISK",
            formal_tracker_feed=False,
            runtime_gt_used=False,
        )


__all__ = ["CONTRACT_VERSION", "ProvisionalOutcomeMapperV3"]
