"""Shadow planner consumer with formal precedence and explicit unresolved risk."""

from __future__ import annotations

from dataclasses import dataclass

from .provisional_dynamic_safety_state_v1 import (
    ProvisionalDynamicSafetyStateV1, ProvisionalSafetyStateTypeV1,
)


CONTRACT_VERSION = "provisional_risk_consumer_v1"


@dataclass(frozen=True)
class ProvisionalRiskDecisionV1:
    semantic: str
    consumed_state_ids: tuple[int, ...]
    suppressed_state_ids: tuple[int, ...]
    formal_router_called: bool = False
    runtime_gt_used: bool = False


class ProvisionalRiskConsumerV1:
    priority = {
        "ACTIVE_FORMAL_RISK": 0,
        ProvisionalSafetyStateTypeV1
        .FORMAL_UNCONFIRMED_SAFETY_STATE: 1,
        ProvisionalSafetyStateTypeV1
        .PROVISIONAL_MEASUREMENT_STATE: 2,
        ProvisionalSafetyStateTypeV1
        .PROVISIONAL_SUPPORT_STATE: 3,
        ProvisionalSafetyStateTypeV1
        .PROVISIONAL_UNRESOLVED_STATE: 4,
        ProvisionalSafetyStateTypeV1.INVALID: 5,
    }

    @staticmethod
    def _key(state):
        provenance = (
            state.source_observation_ids
            or state.source_component_ids
            or state.source_outcome_ids
        )
        return (
            ("formal_generation", state.formal_generation)
            if state.formal_generation is not None
            else tuple(provenance)
        )

    def consume(self, states, formal_risks=()):
        selected = {}
        suppressed = []
        for risk in formal_risks:
            selected[("formal_generation", str(risk["identity"]))] = (
                0, None, "ACTIVE_FORMAL_RISK"
            )
        for state in states:
            if state.state_type in {
                ProvisionalSafetyStateTypeV1.EXPIRED,
                ProvisionalSafetyStateTypeV1.PROMOTED,
            }:
                continue
            key = self._key(state)
            rank = self.priority[state.state_type]
            previous = selected.get(key)
            if previous is None or rank < previous[0]:
                if previous is not None and previous[1] is not None:
                    suppressed.append(previous[1].state_id)
                selected[key] = (rank, state, state.planner_semantic)
            else:
                suppressed.append(state.state_id)
        rows = sorted(selected.values(), key=lambda row: row[0])
        states_used = tuple(
            row[1].state_id for row in rows if row[1] is not None
        )
        semantics = [row[2] for row in rows]
        if "INVALID_EVALUATION" in semantics:
            semantic = "INVALID_EVALUATION"
        elif "ACTIVE_FORMAL_RISK" in semantics:
            semantic = "ACTIVE_FORMAL_RISK"
        elif "ACTIVE_PROVISIONAL_RISK" in semantics:
            semantic = "ACTIVE_PROVISIONAL_RISK"
        elif "ACTIVE_SUPPORT_RISK" in semantics:
            semantic = "ACTIVE_SUPPORT_RISK"
        elif "UNRESOLVED_DYNAMIC_RISK" in semantics:
            semantic = "UNRESOLVED_DYNAMIC_RISK"
        else:
            semantic = "NO_ACTIVE_DYNAMIC_RISK"
        return ProvisionalRiskDecisionV1(
            semantic=semantic,
            consumed_state_ids=states_used,
            suppressed_state_ids=tuple(sorted(set(suppressed))),
        )


__all__ = [
    "CONTRACT_VERSION", "ProvisionalRiskDecisionV1",
    "ProvisionalRiskConsumerV1",
]
