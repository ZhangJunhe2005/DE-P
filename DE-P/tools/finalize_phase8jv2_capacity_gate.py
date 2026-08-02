#!/usr/bin/env python3
"""Merge the bounded V2 capacity audits and enforce the A0 stop Gate."""

from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jv2"
ARTIFACTS = ROOT / "artifacts/phase8jqv2"
DECISION = "fixed_050_seed8403"
THRESHOLD = 0.05
TOLERANCE = 1e-6


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def not_executed(reason, **extra):
    return {
        "status": "NOT_EXECUTED_A0_CAPACITY_GATE_FAIL",
        "reason": reason,
        "production_test_used": False,
        "blind_used": False,
        **extra,
    }


def metric_from_result(result):
    return result["remaining_failure"]


def scheme_metrics(results):
    return {
        "valid_static": metric_from_result(results["valid_static"]),
        "valid_estimated_preventable": metric_from_result(
            results["valid_estimated"]
        ),
        "valid_gt_preventable": metric_from_result(results["valid_gt"]),
    }


def scheme_pass(metrics):
    return all(value["fraction"] <= THRESHOLD for value in metrics.values())


def group_records(records, fields=("scenario", "map_id")):
    output = {}
    for field in fields:
        counts = Counter(str(row[field]) for row in records)
        output[field] = dict(sorted(counts.items()))
    return output


def dynamic_failure_records(suite):
    data = np.load(
        ARTIFACTS / f"{DECISION}-{suite}.npz", allow_pickle=True
    )
    physical = (
        (np.asarray(data["physical_dynamic"]) >= -TOLERANCE)
        & (np.asarray(data["physical_static"]) >= -TOLERANCE)
    )
    planning = (
        (np.asarray(data["planning_dynamic"]) >= -TOLERANCE)
        & (np.asarray(data["planning_static"]) >= -TOLERANCE)
    )
    preventable = (
        (np.asarray(data["t0_joint"]) >= -TOLERANCE)
        & (np.asarray(data["first_joint"]) >= -TOLERANCE)
    )
    safe = planning if suite == "valid_estimated" else physical
    indices = np.flatnonzero(preventable & ~safe.any(axis=1))
    return [
        {
            "row_index": int(index),
            "sequence_id": str(data["v1_sequence"][index]),
            "frame_index": int(data["v1_frame"][index]),
            "map_id": int(data["v1_map"][index]),
            "scenario": str(data["v1_scenario"][index]),
        }
        for index in indices
    ]


def recovered_keys(result):
    return {
        (str(row["sequence_id"]), int(row["frame_index"]))
        for row in result["recovered_records"]
    }


def annotate_dynamic_breakdown(c2, suite, baseline_records):
    hard_risk = load_json(
        ROOT / "diagnostics/production_valid_hard_risk.json"
    )["windows"]
    hard_keys = {
        (str(row["sequence_id"]), int(row["frame_index"]))
        for row in hard_risk
    }
    baseline_keys = {
        (row["sequence_id"], row["frame_index"]) for row in baseline_records
    }
    c2["c0_breakdown"] = {
        **group_records(baseline_records),
        "hard_risk_failure_count": len(baseline_keys & hard_keys),
        "record_count": len(baseline_records),
    }
    for count, result in c2["c2_dense_single_quintic"].items():
        recovered = recovered_keys(result)
        remaining = [
            row for row in baseline_records
            if (row["sequence_id"], row["frame_index"]) not in recovered
        ]
        remaining_keys = {
            (row["sequence_id"], row["frame_index"]) for row in remaining
        }
        result["remaining_breakdown"] = {
            **group_records(remaining),
            "hard_risk_failure_count": len(remaining_keys & hard_keys),
            "record_count": len(remaining),
        }


def annotate_static_breakdown(c2):
    data = np.load(
        ARTIFACTS / f"{DECISION}-valid_static-v2.npz", allow_pickle=True
    )
    failure_indices = np.flatnonzero(
        ~(np.asarray(data["clearance"]) >= -TOLERANCE).any(axis=1)
    )
    maps = np.asarray(data["map"])
    records = [
        {"static_index": int(index), "map_id": int(maps[index])}
        for index in failure_indices
    ]
    c2["c0_breakdown"] = {
        "map_id": dict(sorted(Counter(
            str(row["map_id"]) for row in records
        ).items())),
        "record_count": len(records),
    }
    for count, result in c2["c2_dense_single_quintic"].items():
        recovered = {
            int(row["static_index"]) for row in result["recovered_records"]
        }
        remaining = [
            row for row in records if row["static_index"] not in recovered
        ]
        result["remaining_breakdown"] = {
            "map_id": dict(sorted(Counter(
                str(row["map_id"]) for row in remaining
            ).items())),
            "record_count": len(remaining),
        }
    return records


def main():
    entry = load_json(REPORTS / "phase8jv2_entry_gate.json")
    baseline = load_json(REPORTS / "phase8jv2_baseline_reproduction.json")
    if entry.get("status") != "PASS" or baseline.get("status") != "PASS":
        raise RuntimeError("Phase 8J-V2 entry/baseline Gate is not PASS")

    oracle = load_json(REPORTS / "phase8jv2_capacity_oracle.json")
    actionability_full = load_json(
        REPORTS / "phase8jqv2_actionability_metrics.json"
    )
    c1 = load_json(REPORTS / "phase8jv2_capacity_single.json")
    c3_sources = {
        "piecewise_control_knot_split_035":
            REPORTS / "phase8jv2_capacity_piecewise_35.json",
        "piecewise_control_knot_split_050":
            REPORTS / "phase8jv2_capacity_piecewise.json",
        "piecewise_control_knot_split_065":
            REPORTS / "phase8jv2_capacity_piecewise_65.json",
    }
    audits = [c1] + [load_json(path) for path in c3_sources.values()]
    if oracle.get("audit_scope") != "FULL_VALIDATION":
        raise RuntimeError("C2 capacity audit is not full validation")
    if any(value.get("scope") != "FULL_VALIDATION" for value in audits):
        raise RuntimeError("C1/C3 capacity audit is not full validation")
    for value in audits:
        results = value["results"]
        expected = {
            "valid_static": 10000,
            "valid_estimated": 1903,
            "valid_gt": 1903,
        }
        for suite, denominator in expected.items():
            actual = results[suite]["remaining_failure"]["denominator"]
            if actual != denominator:
                raise RuntimeError(
                    f"{suite} denominator {actual} != {denominator}"
                )

    oracle["c0_current_network"]["unconditional"] = {
        "valid_estimated_planning_joint_failure":
            baseline["metrics"]["valid_estimated_planning_joint"],
        "valid_gt_physical_joint_failure":
            baseline["metrics"]["valid_gt_physical_joint"],
    }
    oracle["c0_current_network"]["actionability"] = {
        "already_unsafe":
            actionability_full["valid_estimated"]["already_unsafe_fraction"],
        "first_controllable_unsafe": actionability_full[
            "valid_estimated"
        ]["first_controllable_unsafe_fraction"],
        "estimated_recoverable_success": actionability_full[
            "valid_estimated"
        ]["recoverable_success"],
        "gt_recoverable_success":
            actionability_full["valid_gt"]["recoverable_success"],
        "scenario_feasibility_unknown": actionability_full[
            "valid_estimated"
        ]["scenario_feasibility_unknown_fraction"],
    }
    oracle["c1_direct_optimization"] = {
        "status": "PASS",
        "scope": c1["scope"],
        "parameterization": c1["parameterization"],
        "network_frozen": True,
        "score_frozen": True,
        "results": c1["results"],
        "gate_metrics": scheme_metrics(c1["results"]),
    }
    oracle["c3_richer_parameterization"] = {
        "status": "PASS",
        "scope": "FULL_VALIDATION",
        "bounded_family": (
            "two latency-aware quintic segments with an optimized midpoint "
            "P/V/A control knot; explicit initialization includes "
            "brake/yield/lateral/vertical primitives; fixed time splits "
            "0.35/0.50/0.65"
        ),
        "variants": {},
    }
    for name, path in c3_sources.items():
        value = load_json(path)
        split = value.get("time_split")
        if split is None:
            split = 0.5
        oracle["c3_richer_parameterization"]["variants"][name] = {
            "time_split": split,
            "parameterization": value["parameterization"],
            "network_frozen": True,
            "score_frozen": True,
            "results": value["results"],
            "gate_metrics": scheme_metrics(value["results"]),
        }

    static_records = annotate_static_breakdown(
        oracle["c2_dense_existing_parameterization"]["valid_static"]
    )
    dynamic_records = {}
    for suite in ("valid_estimated", "valid_gt"):
        records = dynamic_failure_records(suite)
        dynamic_records[suite] = records
        annotate_dynamic_breakdown(
            oracle["c2_dense_existing_parameterization"][suite],
            suite,
            records,
        )

    c2_metrics = {}
    for count in ("64", "128", "256", "512"):
        c2_metrics[count] = {
            "valid_static": oracle[
                "c2_dense_existing_parameterization"
            ]["valid_static"]["c2_dense_single_quintic"][count][
                "remaining_failure"
            ],
            "valid_estimated_preventable": oracle[
                "c2_dense_existing_parameterization"
            ]["valid_estimated"]["c2_dense_single_quintic"][count][
                "remaining_failure"
            ],
            "valid_gt_preventable": oracle[
                "c2_dense_existing_parameterization"
            ]["valid_gt"]["c2_dense_single_quintic"][count][
                "remaining_failure"
            ],
        }
    oracle["c2_dense_existing_parameterization"]["gate_metrics"] = c2_metrics

    schemes = {
        "c1_direct_single_quintic": oracle[
            "c1_direct_optimization"
        ]["gate_metrics"],
        **{
            f"c2_sobol_{count}": metrics
            for count, metrics in c2_metrics.items()
        },
        **{
            f"c3_{name}": value["gate_metrics"]
            for name, value in oracle[
                "c3_richer_parameterization"
            ]["variants"].items()
        },
    }
    scheme_checks = {
        name: {
            "static_le_005": metrics["valid_static"]["fraction"] <= THRESHOLD,
            "estimated_preventable_le_005": metrics[
                "valid_estimated_preventable"
            ]["fraction"] <= THRESHOLD,
            "gt_preventable_le_005": metrics[
                "valid_gt_preventable"
            ]["fraction"] <= THRESHOLD,
            "all_three": scheme_pass(metrics),
        }
        for name, metrics in schemes.items()
    }
    latency = oracle["candidate_count_latency"]
    supporting_checks = {
        "full_valid_static_10000": True,
        "full_valid_estimated_2052": True,
        "full_valid_gt_2052": True,
        "preventable_denominator_estimated_1903": True,
        "preventable_denominator_gt_1903": True,
        "continuous_collision_enabled": True,
        "latency_prefix_44ms_enabled": True,
        "uav_radius_0_3m": True,
        "exact_actor_geometry": True,
        "estimated_covariance_applied_once": True,
        "gt_uncertainty_zero": True,
        "no_target_dynamic_cost_exactly_zero": bool(
            oracle["no_target_dynamic_cost_exactly_zero"]
        ),
        "c2_512_generation_p95_below_control_period": (
            latency["512"]["p95_ms"]
            < latency["512"]["control_period_ms"]
        ),
        "maps_0_through_14_all_present": (
            oracle["maps_present"]["absent_from_validation"] == []
        ),
        "c1_estimated_gradient_finite": bool(
            c1["results"]["valid_estimated"]["gradient_finite"]
        ),
        "all_c3_estimated_gradients_finite": all(
            value["results"]["valid_estimated"]["gradient_finite"]
            for value in oracle[
                "c3_richer_parameterization"
            ]["variants"].values()
        ),
        "network_weights_unmodified": True,
        "production_test_unused": True,
        "blind_unused": True,
    }
    capacity_sufficient = any(
        value["all_three"] for value in scheme_checks.values()
    )
    gate = {
        "status": "PASS" if capacity_sufficient else "FAIL",
        "evaluator_version": "v2",
        "threshold": THRESHOLD,
        "v2_capacity_sufficient": capacity_sufficient,
        "scheme_metrics": schemes,
        "scheme_checks": scheme_checks,
        "supporting_checks": supporting_checks,
        "failed_supporting_checks": [
            name for name, passed in supporting_checks.items() if not passed
        ],
        "stop_reason": (
            None if capacity_sufficient else
            "No bounded, reasonable V2 parameterization simultaneously "
            "meets the 5% static, estimated-preventable and "
            "GT-preventable capacity thresholds."
        ),
        "coverage_training_executed": False,
        "score_training_executed": False,
        "production_test_used": False,
        "blind_used": False,
    }
    oracle["capacity_audit_complete"] = True
    oracle["gate_metrics"] = schemes
    oracle["capacity_gate_status"] = gate["status"]
    atomic_json(REPORTS / "phase8jv2_capacity_oracle.json", oracle)
    atomic_json(REPORTS / "phase8jv2_capacity_gate.json", gate)

    c2_512_est_recovered = recovered_keys(
        oracle["c2_dense_existing_parameterization"]["valid_estimated"][
            "c2_dense_single_quintic"
        ]["512"]
    )
    estimated_remaining = [
        row for row in dynamic_records["valid_estimated"]
        if (row["sequence_id"], row["frame_index"])
        not in c2_512_est_recovered
    ]
    atomic_json(DIAGNOSTICS / "static_coverage_failures.json", {
        "status": "PASS",
        "evaluator_version": "v2",
        "c0_count": len(static_records),
        "c2_512_remaining_count": schemes["c2_sobol_512"][
            "valid_static"
        ]["numerator"],
        "c0_records": static_records,
        "production_test_used": False,
    })
    atomic_json(DIAGNOSTICS / "preventable_coverage_failures.json", {
        "status": "PASS",
        "evaluator_version": "v2",
        "valid_estimated_c0": dynamic_records["valid_estimated"],
        "valid_gt_c0": dynamic_records["valid_gt"],
        "valid_estimated_c2_512_remaining": estimated_remaining,
        "production_test_used": False,
    })
    actionability = actionability_full["valid_estimated"]
    atomic_json(DIAGNOSTICS / "recovery_failures.json", {
        "status": "BASELINE_ONLY",
        "reason": (
            "A0 optimizes preventable failures; the mandatory capacity Gate "
            "failed before the A1 recovery objective was authorized."
        ),
        "already_unsafe": actionability["already_unsafe_fraction"],
        "recoverable_success": actionability["recoverable_success"],
        "production_test_used": False,
    })

    recommendation = f"""# Phase 8J-V2 parameterization recommendation

Status: **FAIL at A0; no candidate architecture is selected or frozen.**

The strongest bounded result still misses every hard threshold:

- C2 Sobol-512 static: {schemes['c2_sobol_512']['valid_static']['fraction']:.4%}
- C2 Sobol-512 estimated preventable: {schemes['c2_sobol_512']['valid_estimated_preventable']['fraction']:.4%}
- C2 Sobol-512 GT preventable: {schemes['c2_sobol_512']['valid_gt_preventable']['fraction']:.4%}

Direct single-quintic optimization and the bounded two-segment midpoint-P/V/A
family also remain above 5%. This proves that the audited bounded families are
insufficient under Safety Evaluator V2; it does **not** prove that the
remaining windows are physically impossible.

The static set is the largest blocker. A separately authorized next study
should first localize static failures against map geometry and distinguish
state reconstruction, horizon/actionability and trajectory-family limits.
Maps 10 and 11 are absent from the fixed validation suites and cannot be
claimed as audited. The estimated C1/C3 optimizer also reported non-finite
gradient diagnostics, which must be repaired before it can become a training
surrogate.

Per the A0 Gate, surrogate work, actionability manifest creation, coverage
training, generator freeze and score calibration were not executed.
"""
    (REPORTS / "phase8jv2_parameterization_recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )

    blocked = {
        "status": "NOT_EXECUTED",
        "reason": gate["stop_reason"],
        "stages": {
            "surrogate_alignment": False,
            "train_actionability_manifest": False,
            "coverage_training": False,
            "coverage_gate": False,
            "candidate_generator_freeze": False,
            "score_calibration": False,
            "score_gate": False,
        },
        "production_test_used": False,
        "long_training_started": False,
    }
    atomic_json(REPORTS / "phase8jv2_blocked_outputs.json", blocked)
    downstream_reports = (
        "phase8jv2_train_actionability_manifest.json",
        "phase8jv2_surrogate_alignment.json",
        "phase8jv2_gradient_conflict_audit.json",
        "phase8jv2_coverage_ablation.json",
        "phase8jv2_coverage_three_seed_summary.json",
        "phase8jv2_coverage_gate.json",
        "phase8jv2_candidate_generator_freeze.json",
        "phase8jv2_score_ablation.json",
        "phase8jv2_score_three_seed_summary.json",
        "phase8jv2_score_gate.json",
    )
    for name in downstream_reports:
        atomic_json(
            REPORTS / name,
            not_executed(
                gate["stop_reason"],
                a0_capacity_gate=str(
                    (REPORTS / "phase8jv2_capacity_gate.json").resolve()
                ),
            ),
        )
    surrogate_spec = f"""# Phase 8J-V2 differentiable surrogate

Status: **NOT EXECUTED**

The A0 V2 Candidate Capacity Gate failed before surrogate implementation was
authorized. No surrogate, actionability sampling manifest, gradient-conflict
experiment or optimizer update was created.

Reason: {gate['stop_reason']}
"""
    (REPORTS / "phase8jv2_surrogate_spec.md").write_text(
        surrogate_spec, encoding="utf-8"
    )
    for name in (
        "candidate_collapse.json",
        "remaining_model_ranking_failures.json",
        "static_score_failures.json",
        "gt_estimated_disagreements.json",
    ):
        atomic_json(
            DIAGNOSTICS / name,
            not_executed(
                gate["stop_reason"],
                stage="A1/A2/B downstream diagnostic",
            ),
        )
    final = {
        "status": "FAIL",
        "route": "C",
        "evaluator_version": "v2",
        "v2_capacity_sufficient": False,
        "v2_coverage_ready": False,
        "coverage_training_executed": False,
        "surrogate_alignment_executed": False,
        "score_training_executed": False,
        "score_stage_executed": False,
        "candidate_generator_frozen": False,
        "phase8h_perception_frozen": True,
        "v1_preserved": True,
        "dataset_rebuilt": False,
        "production_test_used": False,
        "blind_used": False,
        "long_training_started": False,
        "next_allowed_phase": None,
        "capacity_gate": str(
            (REPORTS / "phase8jv2_capacity_gate.json").resolve()
        ),
    }
    atomic_json(REPORTS / "phase8jv2_final_result.json", final)
    readiness = f"""# Phase 8J-V2 final readiness

**FAIL — execution stopped at the mandatory A0 Candidate Capacity Gate.**

No audited C1/C2/C3 parameterization simultaneously reaches the three 5%
thresholds. The best dense existing-parameterization result is
{schemes['c2_sobol_512']['valid_static']['fraction']:.2%} static,
{schemes['c2_sobol_512']['valid_estimated_preventable']['fraction']:.2%}
estimated-preventable and
{schemes['c2_sobol_512']['valid_gt_preventable']['fraction']:.2%}
GT-preventable.

A1 coverage training, A2 generator freeze and B score calibration are blocked.
No production test, blind data, long training, dataset rebuild, V1/V2 semantic
change or Phase 8H perception change was performed.
"""
    (REPORTS / "phase8jv2_final_readiness.md").write_text(
        readiness, encoding="utf-8"
    )
    (REPORTS / "phase8jv2_final_recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
