"""Route-A hard boundary: only strict measurements or legal history may activate risk."""

from __future__ import annotations

from dataclasses import dataclass

CONTRACT_VERSION = "deterministic_dynamic_guard_strict_history_only_v1"


@dataclass(frozen=True)
class DeterministicGuardStrictHistoryOnlyConfigV1:
    enabled: bool = True
    weak_provisional_enabled: bool = False
    pdscr1_runtime_reachable: bool = False
    learned_adapter_runtime_reachable: bool = False
    formal_tracker_accepts_weak: bool = False
    unknown_disposition: str = "UNRESOLVED_CONSERVATIVE_VETO"
    out_of_odd_disposition: str = "SAFE_ABORT"

    def validate(self):
        if not self.enabled:
            raise ValueError("Route A deterministic guard must be enabled")
        forbidden = (
            self.weak_provisional_enabled, self.pdscr1_runtime_reachable,
            self.learned_adapter_runtime_reachable, self.formal_tracker_accepts_weak,
        )
        if any(forbidden):
            raise ValueError("weak/provisional/learned runtime paths are forbidden")


@dataclass(frozen=True)
class DeterministicDynamicGuardInputV1:
    evidence_kind: str
    in_odd: bool
    measurement_valid: bool
    history_legal: bool = False
    runtime_gt_present: bool = False
    owner_map_present: bool = False
    future_actor_state_present: bool = False


class DeterministicDynamicGuardStrictHistoryOnlyV1:
    ACCEPTED = {"STRICT_MEASUREMENT", "HISTORY_BACKED_STATE"}
    REJECTED_WEAK = {"WEAK", "PROVISIONAL", "PDSCR1", "LEARNED_ADAPTER"}

    def __init__(self, config=None):
        self.config = config or DeterministicGuardStrictHistoryOnlyConfigV1()
        self.config.validate()

    def evaluate(self, value: DeterministicDynamicGuardInputV1):
        if value.runtime_gt_present or value.owner_map_present or value.future_actor_state_present:
            raise ValueError("runtime GT/owner/future actor state is forbidden")
        if not value.in_odd:
            return {"active_risk": False, "decision": self.config.out_of_odd_disposition,
                    "reason": "OUT_OF_ODD"}
        if value.evidence_kind in self.REJECTED_WEAK:
            return {"active_risk": False, "decision": self.config.unknown_disposition,
                    "reason": "WEAK_PATH_HARD_DISABLED"}
        strict = value.evidence_kind == "STRICT_MEASUREMENT" and value.measurement_valid
        history = value.evidence_kind == "HISTORY_BACKED_STATE" and value.history_legal
        if strict or history:
            return {"active_risk": True, "decision": "BDRR1_BRIR1_INTERVENTION",
                    "reason": value.evidence_kind}
        return {"active_risk": False, "decision": self.config.unknown_disposition,
                "reason": "NO_LEGAL_STRICT_OR_HISTORY_EVIDENCE"}

