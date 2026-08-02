"""PEPCR1 Strategy A: only compatible recent dynamic history is active."""

from __future__ import annotations

from .dynamic_measurement_outcome_v1 import DynamicMeasurementStatusV1
from .pending_support_state_v1 import (
    PendingSupportStateV1, bounds_from_outcome,
)
from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateTypeV1,
)
from .provisional_evidence_authorizer_fast_v1 import (
    ProvisionalEvidenceAuthorizerFastV1,
)
from .provisional_outcome_mapper_v1 import ProvisionalOutcomeMapperV1


CONTRACT_VERSION = "history_backed_dynamic_only_v1"


def _active_support(outcome, state_id, generation, lifecycle):
    timestamp = float(outcome["timestamp"])
    support = outcome["support"]
    return ProvisionalDynamicSafetyStateV1(
        state_id=int(state_id), generation=int(generation),
        state_type=ProvisionalSafetyStateTypeV1.PROVISIONAL_SUPPORT_STATE,
        source_outcome_ids=(int(outcome["outcome_id"]),),
        source_component_ids=tuple(
            int(value) for value in outcome.get("source_component_ids", ())
        ),
        source_observation_ids=(), birth_timestamp=timestamp,
        last_evidence_timestamp=timestamp,
        expiry_timestamp=timestamp+lifecycle["support_max_age_s"],
        missed_frames=0,
        maximum_missed_frames=lifecycle["maximum_missed_frames"],
        evidence_frames=1, position_reference="FINITE_WORLD_POSITION_SET",
        support_bounds={
            "min_world": tuple(support["position_set_min_world"]),
            "max_world": tuple(support["position_set_max_world"]),
        },
        planner_semantic="ACTIVE_SUPPORT_RISK",
        formal_tracker_feed=False, runtime_gt_used=False,
    )


class HistoryBackedSupportPolicyV1:
    strategy_id = "A_HISTORY_BACKED_DYNAMIC_ONLY"

    def __init__(self, pds_contract, evidence_contract, policy_contract):
        self.frozen_mapper = ProvisionalOutcomeMapperV1(pds_contract)
        self.authorizer = ProvisionalEvidenceAuthorizerFastV1(
            evidence_contract
        )
        self.lifecycle = pds_contract["lifecycle"]
        self.policy_contract = policy_contract
        self.last_authorization = None
        self.last_pending = None
        self.last_semantic = "NO_ACTIVE_DYNAMIC_RISK"

    def map(self, outcome, state_id, generation=0, causal_context=None):
        status = DynamicMeasurementStatusV1(outcome["status"])
        self.last_pending = None
        if status != DynamicMeasurementStatusV1.BOUNDED_SAFETY_SUPPORT:
            self.last_authorization = None
            state = self.frozen_mapper.map(
                outcome, state_id, generation, causal_context
            )
            self.last_semantic = (
                "NO_ACTIVE_DYNAMIC_RISK"
                if state is None else state.planner_semantic
            )
            return state
        decision = self.authorizer.authorize(outcome, causal_context)
        self.last_authorization = decision
        if (
            decision.authorized
            and decision.history_level == "RECENT_DYNAMIC_TRACK_COMPATIBLE"
        ):
            state = _active_support(
                outcome, state_id, generation, self.lifecycle
            )
            self.last_semantic = state.planner_semantic
            return state
        timestamp = float(outcome["timestamp"])
        self.last_pending = PendingSupportStateV1(
            state_id=int(state_id), generation=int(generation),
            source_outcome_ids=(int(outcome["outcome_id"]),),
            source_component_ids=tuple(
                int(value)
                for value in outcome.get("source_component_ids", ())
            ),
            birth_frame=int(outcome.get("frame_index", 0)),
            birth_timestamp=timestamp, last_evidence_timestamp=timestamp,
            expiry_timestamp=timestamp+float(
                self.policy_contract["pending"]["maximum_age_s"]
            ),
            support_bounds=bounds_from_outcome(outcome),
            provenance_signature=("FINITE_WORLD_POSITION_SET",),
            planner_semantic="PENDING_OR_UNKNOWN_SUPPORT",
        )
        self.last_semantic = "PENDING_OR_UNKNOWN_SUPPORT"
        return None


__all__ = ["CONTRACT_VERSION", "HistoryBackedSupportPolicyV1"]
