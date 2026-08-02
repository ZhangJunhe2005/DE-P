"""Boundary object between evidence authorization and frozen state birth."""

from __future__ import annotations

from dataclasses import dataclass

from .provisional_evidence_authorizer_v1 import (
    ProvisionalEvidenceDecisionV1,
)


CONTRACT_VERSION = "provisional_support_birth_contract_v1"


@dataclass(frozen=True)
class ProvisionalSupportBirthContractV1:
    decision: ProvisionalEvidenceDecisionV1
    state_birth_allowed: bool
    dynamic_consumer_allowed: bool
    unresolved_conversion_allowed: bool = False
    formal_tracker_feed: bool = False
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used or self.formal_tracker_feed:
            raise ValueError("GT/formal feed is forbidden")
        if self.unresolved_conversion_allowed:
            raise ValueError("support denial cannot become unresolved risk")
        if self.state_birth_allowed != self.decision.authorized:
            raise ValueError("birth must follow evidence authorization")
        if self.dynamic_consumer_allowed != self.state_birth_allowed:
            raise ValueError("diagnostics cannot enter dynamic consumer")


def support_birth_contract(decision):
    return ProvisionalSupportBirthContractV1(
        decision=decision,
        state_birth_allowed=decision.authorized,
        dynamic_consumer_allowed=decision.authorized,
    )


__all__ = [
    "CONTRACT_VERSION", "ProvisionalSupportBirthContractV1",
    "support_birth_contract",
]
