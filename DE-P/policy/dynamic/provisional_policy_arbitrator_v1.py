"""One-shot A/B/C terminal selection for PEPCR1."""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "provisional_policy_arbitrator_v1"
ALLOWED_STRATEGIES = (
    "A_HISTORY_BACKED_DYNAMIC_ONLY",
    "B_TWO_STAGE_NO_HISTORY_BOOTSTRAP",
    "C_HANDCRAFTED_EVIDENCE_INSUFFICIENT",
)


@dataclass(frozen=True, slots=True)
class PolicySelectionV1:
    route: str
    selected_policy: str | None
    reason: str
    handcrafted_policy_selected: bool
    model_data_contract_required: bool


def select_terminal_policy(strategy_a, strategy_b):
    """Apply the predeclared safety-first, simplicity-biased decision."""
    def hard_pass(row):
        return all((
            row["runtime_pass"],
            row["unsafe_recommendation_increase"] == 0,
            row["negative_intervention_gate_pass"],
            row["unresolved_downgrade_count"] == 0,
            row["critical_regressions_pass"],
        ))

    a_ok, b_ok = hard_pass(strategy_a), hard_pass(strategy_b)
    if b_ok and (
        strategy_b["causally_observable_unsafe_proxy"]
        < strategy_a["causally_observable_unsafe_proxy"]
        and strategy_b["reaction_time_margin_frames"]
        > strategy_a["reaction_time_margin_frames"]
        and strategy_b["candidate_availability"]
        >= strategy_a["candidate_availability"]
        and strategy_b["negative_interventions"]
        <= strategy_a["negative_interventions"]
        and strategy_b["safe_false_veto"]
        <= strategy_a["safe_false_veto"]
    ):
        return PolicySelectionV1(
            "B", "two_stage_no_history_bootstrap_v1",
            "B materially improves planner-level no-history response "
            "without a frozen-Gate regression", True, False,
        )
    if a_ok:
        return PolicySelectionV1(
            "A", "history_backed_dynamic_only_v1",
            "B has no proven planner-level advantage; select simpler A",
            True, False,
        )
    if b_ok:
        return PolicySelectionV1(
            "B", "two_stage_no_history_bootstrap_v1",
            "A fails a hard Gate while B passes", True, False,
        )
    return PolicySelectionV1(
        "C", None,
        "Neither handcrafted policy satisfies all frozen planner-level Gates",
        False, True,
    )


__all__ = [
    "CONTRACT_VERSION", "ALLOWED_STRATEGIES",
    "PolicySelectionV1", "select_terminal_policy",
]
