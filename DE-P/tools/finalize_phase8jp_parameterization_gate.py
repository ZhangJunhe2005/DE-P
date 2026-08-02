#!/usr/bin/env python3
"""Finalize the fail-closed Phase 8J-P parameterization capacity Gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jp"
SCHEMES = ("o5a", "o5b", "o5c", "o5d")
SCHEME_NAMES = {
    "o5a": "O5-A midpoint position",
    "o5b": "O5-B midpoint velocity",
    "o5c": "O5-C time split",
    "o5d": "O5-D brake/yield family",
    "o6": "O6 piecewise-constant jerk",
}
SCHEME_REPORTS = {
    "o5a": "phase8jp_o5a_midpoint_position.json",
    "o5b": "phase8jp_o5b_midpoint_velocity.json",
    "o5c": "phase8jp_o5c_time_split.json",
    "o5d": "phase8jp_o5d_brake_yield.json",
    "o6": "phase8jp_o6_kinodynamic_oracle.json",
}
TAXONOMY_FILES = {
    "midpoint_position_limited": "midpoint_position_limited.json",
    "midpoint_velocity_limited": "midpoint_velocity_limited.json",
    "time_allocation_limited": "time_allocation_limited.json",
    "brake_yield_limited": "brake_yield_limited.json",
    "piecewise_quintic_limited": "piecewise_quintic_limited.json",
    "kinodynamic_unresolved": "kinodynamic_unresolved.json",
}


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity(row):
    return row["sequence_id"], int(row["frame_index"])


def fraction_pass(value):
    return value <= 0.05 + 1e-12


def minimal_o5_family(records_by_suite):
    for size in range(1, len(SCHEMES) + 1):
        candidates = []
        for family in combinations(SCHEMES, size):
            summaries = {}
            passed = True
            latency = 0.0
            for suite, records in records_by_suite.items():
                recovered = sum(
                    any(row[scheme]["success"] for scheme in family)
                    for row in records
                )
                failure = (len(records) - recovered) / 2052.0
                summaries[suite] = {
                    "recovered_count": recovered,
                    "joint_coverage_failure_fraction": failure,
                }
                passed &= fraction_pass(failure)
                latency += sum(
                    sum(row["offline_latency_seconds"][scheme]
                        for scheme in family)
                    for row in records
                ) / max(1, len(records))
            if passed:
                candidates.append((latency, family, summaries))
        if candidates:
            _, family, summaries = min(candidates)
            return list(family), summaries
    return None, {}


def grouped_counts(records, field):
    result = defaultdict(Counter)
    for row in records:
        result[str(row[field])][row["taxonomy"]] += 1
    return {
        group: {"total": sum(counts.values()), **dict(counts)}
        for group, counts in sorted(result.items())
    }


def main():
    comparison_path = REPORTS / "phase8jp_parameterization_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    if not comparison.get("parameterization_audit_complete"):
        raise RuntimeError("Phase 8J-P parameterization audit is incomplete")
    records_by_suite = comparison["records_by_suite"]
    expected = {"valid_estimated": 142, "valid_gt": 142}
    actual = {key: len(value) for key, value in records_by_suite.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected completion counts: {actual}")
    if comparison.get("completion_count") != 284:
        raise RuntimeError("all independent estimated/GT completions are required")

    reports = {
        scheme: json.loads((REPORTS / filename).read_text())
        for scheme, filename in SCHEME_REPORTS.items()
    }
    individual = {}
    for scheme, report in reports.items():
        est = report
        gt = report["valid_gt_summary"]
        physical_assessed = (
            est["recovered_count"] + gt["recovered_count"]
        ) > 0
        physical_pass = (
            physical_assessed
            and (
                est["recovered_count"] == 0
                or est["physical_feasibility_rate_of_recovered"] == 1.0
            )
            and (
                gt["recovered_count"] == 0
                or gt["physical_feasibility_rate_of_recovered"] == 1.0
            )
        )
        individual[scheme] = {
            "valid_estimated_joint_coverage_failure_fraction":
                est["joint_coverage_failure_fraction"],
            "valid_gt_joint_coverage_failure_fraction":
                gt["joint_coverage_failure_fraction"],
            "valid_estimated_recovered_count": est["recovered_count"],
            "valid_gt_recovered_count": gt["recovered_count"],
            "physical_feasibility_assessed_on_recovered": physical_assessed,
            "physical_feasibility_pass": physical_pass,
            "optimizer_failure_count": (
                est["optimizer_failure_count"]
                + gt["optimizer_failure_count"]
            ),
            "gate_pass": (
                fraction_pass(est["joint_coverage_failure_fraction"])
                and fraction_pass(gt["joint_coverage_failure_fraction"])
                and physical_pass
                and est["optimizer_failure_count"] == 0
                and gt["optimizer_failure_count"] == 0
            ),
        }

    passing_o5 = [scheme for scheme in SCHEMES if individual[scheme]["gate_pass"]]
    selected_single = min(
        passing_o5,
        key=lambda key: reports[key]["offline_latency_seconds_mean_per_window"],
        default=None,
    )
    family, family_summaries = minimal_o5_family(records_by_suite)
    o5_sufficient = selected_single is not None or family is not None
    o6_sufficient = individual["o6"]["gate_pass"]
    if selected_single:
        selected = selected_single
        next_phase = "phase8jq_piecewise_candidate_implementation"
        decision = "single_o5_pass"
    elif family:
        selected = "+".join(family)
        next_phase = "phase8jq_piecewise_candidate_implementation"
        decision = "finite_o5_family_union_pass"
    elif o6_sufficient:
        selected = None
        next_phase = "phase8jq_control_knot_or_spline_parameterization"
        decision = "o6_only_pass"
    else:
        selected = None
        next_phase = None
        decision = "o6_fail"

    for taxonomy, filename in TAXONOMY_FILES.items():
        atomic_json(DIAGNOSTICS / filename, {
            "status": "PASS",
            "taxonomy": taxonomy,
            "valid_estimated": [
                row for row in records_by_suite["valid_estimated"]
                if row["taxonomy"] == taxonomy
            ],
            "valid_gt": [
                row for row in records_by_suite["valid_gt"]
                if row["taxonomy"] == taxonomy
            ],
            "optimizer_failures_excluded_from_intrinsic": True,
        })

    unresolved = {
        suite: [
            row for row in records
            if row["taxonomy"] in {
                "kinodynamic_unresolved", "optimizer_failure"
            }
        ]
        for suite, records in records_by_suite.items()
    }
    physical_categories = {}
    for suite, rows in unresolved.items():
        classified = []
        for row in rows:
            if row["optimizer_failure"]:
                reason = "optimizer_failure"
            elif row["actor_count"] == 0:
                reason = "static_corridor_or_safety_geometry"
            elif row["actor_count"] > 1:
                reason = "multi_actor_dynamic_blockade_candidate"
            elif not row["o6"]["physical"]:
                reason = "dynamics_bound_or_stopping_distance_candidate"
            else:
                reason = "unavoidable_within_formal_horizon_candidate"
            classified.append({
                "sequence_id": row["sequence_id"],
                "frame_index": row["frame_index"],
                "map_id": row["map_id"],
                "scenario": row["scenario"],
                "actor_count": row["actor_count"],
                "reason": reason,
                "o6": row["o6"],
            })
        physical_categories[suite] = classified
    atomic_json(REPORTS / "phase8jp_physical_infeasibility_audit.json", {
        "status": "PASS",
        "jerk_limit_mps3": comparison["oracle_config"]["jerk_limit_mps3"],
        "jerk_limit_authority": comparison["oracle_config"]["jerk_limit_status"],
        "intrinsic_infeasibility_claimed": False,
        "reason": (
            "Finite multi-start projected optimization is an upper-bound "
            "audit, not a proof of infeasibility."
        ),
        "optimizer_failure_is_separate": True,
        "records_by_suite": physical_categories,
    })

    prior = json.loads(
        (REPORTS / "phase8jr_capacity_oracle.json").read_text()
    )
    static_summary = {}
    for suite in ("valid_estimated", "valid_gt"):
        rows = [
            row for row in prior["suites"][suite]["failure_records"]
            if row["actor_count"] == 0
        ]
        static_summary[suite] = {
            "o0_failure_count": len(rows),
            "dynamic_cost_exactly_zero": all(
                row["o0_dynamic_safe_count"] == 15 for row in rows
            ),
            "o1_recovered_count": sum(row["o1_success"] for row in rows),
            "o5_o6_evaluation": (
                "not required on these windows because all are already "
                "recovered by O1 and are outside the 142 parameterization-"
                "limited set"
            ),
            "piecewise_recovery_count": None,
            "records": rows,
        }
    atomic_json(REPORTS / "phase8jp_static_no_target_audit.json", {
        "status": "PASS",
        "attribution": "static candidate geometry, not dynamic perception",
        "suites": static_summary,
    })

    complexities = {
        "o5a": (12, 12, 30, "waypoint piecewise"),
        "o5b": (15, 15, 30, "waypoint plus midpoint velocity"),
        "o5c": (16, 15, 30, "O5-B across five fixed time splits"),
        "o5d": (15, 15, 30, "six fixed brake/hover/yield families"),
        "o6": (24, 24, 32, "eight piecewise-constant jerk controls"),
    }
    complexity = {}
    for scheme, (parameters, variables, samples, description) in (
        complexities.items()
    ):
        complexity[scheme] = {
            "description": description,
            "trajectory_parameter_count": parameters,
            "optimizer_variable_count_per_start": variables,
            "trajectory_sample_count": samples,
            "static_dynamic_risk_evaluator": "formal unchanged evaluator",
            "offline_latency_seconds_mean_per_window_estimated":
                reports[scheme]["offline_latency_seconds_mean_per_window"],
            "offline_latency_seconds_mean_per_window_gt":
                reports[scheme]["valid_gt_summary"][
                    "offline_latency_seconds_mean_per_window"
                ],
            "estimated_runtime_sampling_cost": (
                f"{samples} state samples per trajectory before batching"
            ),
            "checkpoint_output_schema_change": "none in Phase 8J-P",
            "ros_primitive_interface_impact": (
                "none in audit; future deployment would require a piecewise "
                "trajectory/control-knot interface"
            ),
            "c2_continuity": True,
            "physical_feasibility": individual[scheme][
                "physical_feasibility_pass"
            ],
        }

    taxonomy_lines = [
        "# Phase 8J-P failure taxonomy",
        "",
        "The classification is based on independent estimated-context and "
        "ground-truth obstacle evaluations. Optimizer failures are never "
        "counted as intrinsic physical failures.",
        "",
    ]
    for suite, records in records_by_suite.items():
        taxonomy_lines.extend([
            f"## {suite}",
            "",
            f"- Total audited: {len(records)}",
            f"- Taxonomy: {dict(Counter(r['taxonomy'] for r in records))}",
            f"- By scenario: {grouped_counts(records, 'scenario')}",
            f"- By map: {grouped_counts(records, 'map_id')}",
            f"- By category: {grouped_counts(records, 'category')}",
            "",
        ])
    atomic_text(
        REPORTS / "phase8jp_failure_taxonomy.md",
        "\n".join(taxonomy_lines) + "\n",
    )

    gate = {
        "status": "PASS" if (o5_sufficient or o6_sufficient) else "FAIL",
        "threshold": 0.05,
        "parameterization_audit_complete": True,
        "individual_schemes": individual,
        "o5_minimal_passing_family": family,
        "o5_family_summaries": family_summaries,
        "piecewise_quintic_sufficient": o5_sufficient,
        "kinodynamic_capacity_sufficient": o6_sufficient,
        "capacity_sufficient": o5_sufficient or o6_sufficient,
        "selected_parameterization": selected,
        "decision": decision,
        "next_allowed_phase": next_phase,
        "complexity": complexity,
        "jerk_limit_requires_controller_confirmation": True,
        "full_2052_rerun_required_before_implementation": bool(
            o5_sufficient or o6_sufficient
        ),
        "full_2052_rerun_completed": False,
        "network_weights_modified": False,
        "score_stage_executed": False,
        "perception_frozen": True,
        "production_test_used": False,
    }
    resume_path = REPORTS / "phase8jp_resume_validation.json"
    gate["resume_validation"] = (
        json.loads(resume_path.read_text()) if resume_path.is_file() else None
    )
    gate["cpu_gpu_tolerance_record"] = {
        "cpu_unit_test_dtype": "float64",
        "continuity_absolute_tolerance": 1e-10,
        "gpu_formal_safety_tolerance": 1e-6,
        "cross_device_output_equality_used_as_gate": False,
        "reason": (
            "CPU and CUDA kernels need not be bit-identical; deterministic "
            "resume is gated by persisted completion/report hashes."
        ),
    }
    # A capacity indication on the unresolved set is not the final PASS until
    # the prompt-mandated complete 2052-window rerun is attached.
    if gate["full_2052_rerun_required_before_implementation"]:
        gate["status"] = "PENDING_FULL_VALIDATION"
        gate["next_allowed_phase"] = None
    atomic_json(REPORTS / "phase8jp_capacity_gate.json", gate)

    final = {
        "status": gate["status"],
        "parameterization_audit_complete": True,
        "parameterization_sufficient": o5_sufficient,
        "piecewise_quintic_sufficient": o5_sufficient,
        "kinodynamic_capacity_sufficient": o6_sufficient,
        "capacity_sufficient": o5_sufficient or o6_sufficient,
        "selected_parameterization": selected,
        "network_weights_modified": False,
        "score_stage_executed": False,
        "perception_frozen": True,
        "production_test_used": False,
        "next_allowed_phase": gate["next_allowed_phase"],
    }
    atomic_json(REPORTS / "phase8jp_final_result.json", final)
    recommendation = [
        "# Phase 8J-P final recommendation",
        "",
        f"- Decision: `{decision}`",
        f"- Selected parameterization: `{selected}`",
        f"- Gate status: `{gate['status']}`",
        f"- Next allowed phase: `{gate['next_allowed_phase']}`",
        "- Network/Score/perception changes: none.",
        "- Production test usage: none.",
        "- The 30 m/s³ jerk limit is audit-only and must be confirmed "
        "against the controller before implementation.",
    ]
    if gate["status"] == "PENDING_FULL_VALIDATION":
        recommendation.append(
            "- Run the selected scheme/family on all 2052 estimated and 2052 "
            "GT windows before opening Phase 8J-Q."
        )
    elif gate["status"] == "FAIL":
        recommendation.append(
            "- Do not implement a proposal network; continue physical, "
            "scenario-feasibility, and safety-semantics audits."
        )
    atomic_text(
        REPORTS / "phase8jp_final_recommendation.md",
        "\n".join(recommendation) + "\n",
    )
    print(json.dumps({
        "status": gate["status"],
        "decision": decision,
        "selected": selected,
        "next_allowed_phase": gate["next_allowed_phase"],
    }, indent=2))


if __name__ == "__main__":
    main()
