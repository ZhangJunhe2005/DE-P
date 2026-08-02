#!/usr/bin/env python3
"""Finalize Phase 8J-R at the mandatory pre-training capacity Gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jr"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def not_executed(reason, **extra):
    return {
        "status": "NOT_EXECUTED",
        "reason": reason,
        "production_test_used": False,
        **extra,
    }


def main():
    entry = json.loads((REPORTS / "phase8jr_entry_gate.json").read_text())
    baseline = json.loads(
        (REPORTS / "phase8jr_baseline_reproduction.json").read_text()
    )
    capacity = json.loads(
        (REPORTS / "phase8jr_capacity_oracle.json").read_text()
    )
    if entry.get("status") != "PASS" or baseline.get("status") != "PASS":
        raise RuntimeError("Phase 8J-R entry/baseline Gate is not PASS")
    if not capacity.get("capacity_audit_complete"):
        raise RuntimeError("formal capacity audit is incomplete")

    estimated = capacity["suites"]["valid_estimated"]
    gt = capacity["suites"]["valid_gt"]
    for suite in (estimated, gt):
        unresolved = suite["taxonomy"]["counts"].get(
            "parameterization_limited", 0
        )
        suite["taxonomy"].update({
            "intrinsically_infeasible_fraction": None,
            "intrinsically_infeasible_fraction_bounds": [
                0.0, unresolved / suite["window_count"],
            ],
            "intrinsic_classification_status": (
                "not identifiable with the current O0-O4 parameterization; "
                "the unresolved set is an upper bound, not a claim that every "
                "window is physically infeasible"
            ),
        })
    atomic_json(REPORTS / "phase8jr_capacity_oracle.json", capacity)
    estimated_o1 = estimated["o1_direct_optimization"][
        "joint_coverage_failure_fraction"
    ]
    estimated_o2 = estimated["o2_dense_same_bound"][
        "joint_coverage_failure_fraction_by_count"
    ]["512"]
    estimated_o3 = estimated["o3_temporal_bank"][
        "joint_coverage_failure_fraction"
    ]
    gt_o1 = gt["o1_direct_optimization"][
        "joint_coverage_failure_fraction"
    ]
    gt_o2 = gt["o2_dense_same_bound"][
        "joint_coverage_failure_fraction_by_count"
    ]["512"]
    gt_o3 = gt["o3_temporal_bank"]["joint_coverage_failure_fraction"]
    capacity_sufficient = min(estimated_o1, estimated_o2, estimated_o3) <= 0.05
    checks = {
        "entry_gate_pass": True,
        "baseline_reproduced": True,
        "valid_estimated_complete_2052": estimated["window_count"] == 2052,
        "valid_gt_complete_2052": gt["window_count"] == 2052,
        "o1_direct_optimization_le_005": estimated_o1 <= 0.05,
        "o2_dense_same_bound_le_005": estimated_o2 <= 0.05,
        "o3_temporal_bank_le_005": estimated_o3 <= 0.05,
        "gt_only_improvement_not_used": (
            abs(estimated_o3 - gt_o3) <= 0.01
        ),
        "production_test_unused": True,
        "network_weights_unmodified": True,
        "score_stage_unexecuted": True,
    }
    capacity_gate = {
        "status": "PASS" if capacity_sufficient else "FAIL",
        "capacity_sufficient": capacity_sufficient,
        "training_executed": False,
        "threshold": 0.05,
        "valid_estimated": {
            "o0_current_15": estimated["o0_current_15"][
                "joint_coverage_failure_fraction"
            ],
            "o1_direct_optimization": estimated_o1,
            "o2_dense_512": estimated_o2,
            "o3_temporal_bank": estimated_o3,
            "o4_extended_diagnostic": estimated[
                "o4_extended_horizon_diagnostic"
            ]["joint_coverage_failure_fraction"],
        },
        "valid_gt": {
            "o0_current_15": gt["o0_current_15"][
                "joint_coverage_failure_fraction"
            ],
            "o1_direct_optimization": gt_o1,
            "o2_dense_512": gt_o2,
            "o3_temporal_bank": gt_o3,
            "o4_extended_diagnostic": gt[
                "o4_extended_horizon_diagnostic"
            ]["joint_coverage_failure_fraction"],
        },
        "intrinsic_infeasibility": estimated["taxonomy"][
            "intrinsic_classification_status"
        ],
        "intrinsic_fraction_bounds": estimated["taxonomy"][
            "intrinsically_infeasible_fraction_bounds"
        ],
        "checks": checks,
        "failed_checks": [
            key for key, passed in checks.items() if not passed
        ],
        "stop_reason": (
            "All Gate-eligible learnable/expressible O1-O3 oracles remain "
            "above 5%; Phase 8J-R section 6 requires immediate stop."
        ),
        "production_test_used": False,
    }
    atomic_json(REPORTS / "phase8jr_capacity_gate.json", capacity_gate)

    taxonomy_names = (
        "optimization_limited", "count_limited", "temporal_limited",
        "horizon_limited", "parameterization_limited",
        "intrinsically_infeasible",
    )
    records = estimated["failure_records"]
    for taxonomy in taxonomy_names:
        selected = [
            row for row in records if row["taxonomy"] == taxonomy
        ]
        payload = {
            "status": (
                "NOT_ASSERTED" if taxonomy == "intrinsically_infeasible"
                else "PASS"
            ),
            "taxonomy": taxonomy,
            "count": len(selected),
            "records": selected,
            "note": (
                "O1-O4 cannot distinguish physical intrinsic infeasibility "
                "from missing trajectory parameterization."
                if taxonomy == "intrinsically_infeasible" else None
            ),
            "production_test_used": False,
        }
        atomic_json(DIAGNOSTICS / f"{taxonomy}.json", payload)
    remaining = [
        row for row in records
        if not any(row["o3_temporal_success"].values())
    ]
    atomic_json(DIAGNOSTICS / "remaining_coverage_failures.json", {
        "status": "PASS",
        "count": len(remaining),
        "fraction_of_full_valid": len(remaining) / 2052.0,
        "records": remaining,
        "production_test_used": False,
    })

    profile_counts = Counter()
    scenario_profile_counts = defaultdict(Counter)
    for row in records:
        for profile, succeeded in row["o3_temporal_success"].items():
            if succeeded:
                profile_counts[profile] += 1
                scenario_profile_counts[row["scenario"]][profile] += 1
    atomic_json(REPORTS / "phase8jr_temporal_candidate_analysis.json", {
        "status": "PASS",
        "gate_eligible": True,
        "profiles_frozen_before_audit": capacity["temporal_profiles"],
        "valid_estimated_recovered_count_by_profile": dict(profile_counts),
        "by_scenario": {
            key: dict(value)
            for key, value in sorted(scenario_profile_counts.items())
        },
        "union_recovered_count": estimated["o3_temporal_bank"][
            "recovered_count"
        ],
        "joint_coverage_failure_fraction": estimated_o3,
        "conclusion": (
            "Temporal profiles recover additional windows but plateau at "
            f"{estimated_o3:.4%}, above the 5% continuation Gate."
        ),
        "production_test_used": False,
    })

    stop_reason = capacity_gate["stop_reason"]
    atomic_json(
        REPORTS / "phase8jr_coverage_gradient_audit.json",
        not_executed(
            stop_reason,
            stage_order="section 7 is after the failed section 6 capacity Gate",
        ),
    )
    atomic_json(
        REPORTS / "phase8jr_candidate_count_latency.json",
        not_executed(
            stop_reason,
            note=(
                "No deployable N=30/45 proposal architecture was created; "
                "offline oracle elapsed times are retained in the capacity report."
            ),
        ),
    )
    atomic_json(
        REPORTS / "phase8jr_sampling_curriculum.json",
        not_executed(stop_reason),
    )
    atomic_json(
        REPORTS / "phase8jr_ablation_summary.json",
        not_executed(stop_reason),
    )
    atomic_json(
        REPORTS / "phase8jr_three_seed_summary.json",
        not_executed(stop_reason, seed_count=0),
    )
    atomic_json(
        REPORTS / "phase8jr_coverage_gate.json",
        not_executed(
            stop_reason,
            coverage_ready=False,
            capacity_gate=str(
                (REPORTS / "phase8jr_capacity_gate.json").resolve()
            ),
        ),
    )
    atomic_json(
        REPORTS / "phase8jr_candidate_generator_freeze.json",
        not_executed(
            stop_reason,
            candidate_generator_frozen=False,
            generator_parameter_hash=None,
        ),
    )

    proposal_design = f"""# Phase 8J-R proposal design decision

Status: **NOT EXECUTED**

The mandatory capacity Gate failed before proposal redesign:

- O1 direct optimization: {estimated_o1:.6f}
- O2 dense same-bound bank (512): {estimated_o2:.6f}
- O3 bounded temporal bank: {estimated_o3:.6f}
- required joint coverage failure: <= 0.050000

The unresolved O1–O4 set contains
{estimated['taxonomy']['counts'].get('parameterization_limited', 0)} windows.
It is an upper bound on intrinsic infeasibility, not proof that those windows
are physically impossible. A richer-parameterization oracle (for example an
intermediate waypoint or piecewise trajectory) is required before choosing
N=30/45 or a temporal proposal architecture.

No score, perception, production-test, blind, ROS, or long-training path was
modified or executed.
"""
    (REPORTS / "phase8jr_proposal_design.md").write_text(
        proposal_design, encoding="utf-8"
    )

    final = {
        "status": "FAIL",
        "capacity_audit_complete": True,
        "capacity_sufficient": False,
        "training_executed": False,
        "coverage_ready": False,
        "score_stage_executed": False,
        "candidate_generator_frozen": False,
        "perception_frozen": True,
        "production_test_used": False,
        "long_training_started": False,
        "next_allowed_phase": None,
        "capacity_gate": str(
            (REPORTS / "phase8jr_capacity_gate.json").resolve()
        ),
        "failed_checks": capacity_gate["failed_checks"],
    }
    atomic_json(REPORTS / "phase8jr_final_result.json", final)
    readiness = f"""# Phase 8J-R final readiness

**FAIL — stopped at the pre-training capacity Gate.**

- Current-15 estimated joint failure: {estimated['o0_current_15']['joint_coverage_failure_fraction']:.4%}
- O1 direct optimization: {estimated_o1:.4%}
- O2 dense-512: {estimated_o2:.4%}
- O3 temporal bank: {estimated_o3:.4%}
- Best Gate-eligible oracle remains above 5%.
- Unresolved parameterization/intrinsic upper bound:
  {estimated['taxonomy']['intrinsically_infeasible_fraction_bounds'][1]:.4%}
- Proposal redesign, coverage training, score training and generator freeze
  were not executed.
- Production test and Phase 8H blind were not used.

The next valid action is a new, explicitly authorized richer-parameterization
oracle. Phase 8J-B and Phase 8K remain blocked.
"""
    (REPORTS / "phase8jr_final_readiness.md").write_text(
        readiness, encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
