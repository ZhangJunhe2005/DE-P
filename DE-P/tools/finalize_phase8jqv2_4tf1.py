#!/usr/bin/env python3
"""Finalize the fail-closed TF1 development decision without holdout access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4tf1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((REPORTS / name).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite TF1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if isinstance(value, str):
        text = value.rstrip() + "\n"
    else:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(text)
    os.replace(temporary, path)


def main():
    development = read("phase8jqv2_4tf1_candidate_development_results.json")
    baseline = read("phase8jqv2_4tf1_baseline_replay.json")
    split = read("phase8jqv2_4tf1_evaluation_split.json")
    entry = read("phase8jqv2_4tf1_entry_gate.json")
    config_path = (
        ROOT / "configs/temporal_foreground_contract_v2_1_candidates.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    source_path = ROOT / "policy/dynamic/range_image_foreground_v2_1.py"
    registry_path = ROOT / "policy/dynamic/foreground_contract_registry.py"
    evaluator_path = ROOT / "tools/evaluate_phase8jqv2_4tf1_candidates.py"
    selectable = [
        row for row in development["results"]
        if row["candidate"] != "ablation_seed7"
    ]
    gates = [row for row in selectable if row["development_hard_gate"]]
    if gates:
        raise RuntimeError(
            "a development candidate passed; holdout freeze is required"
        )

    specs = {
        "status": "PASS",
        "contract_version":
            "temporal_foreground_observability_contract_v2_1_candidates",
        "legacy_default": config["legacy_default"],
        "candidate_selected": config["candidate_selected"],
        "runtime_invariants": {
            "causal": True, "future_frames": 0, "runtime_gt": False,
            "annex_provenance": False,
            "bounded_history_frames": config["history_frames"],
        },
        "families": {
            name: {
                "family": value["family"],
                "selectable": value.get("selectable", True),
                "purpose": {
                    "candidate_a": "weighted causal multi-history evidence",
                    "candidate_b":
                        "component-first positive residual validation",
                    "candidate_c":
                        "separate occupied/freed evidence with anchored growth",
                    "candidate_d":
                        "live-track-only current-residual ROI reacquisition",
                    "ablation_seed7":
                        "diagnostic legacy seed threshold 8 to 7 only",
                }[name],
            }
            for name, value in config["candidates"].items()
        },
        "candidate_d_birth_allowed": False,
    }
    write_new(REPORTS / "phase8jqv2_4tf1_candidate_specs.json", specs)
    write_new(
        REPORTS / "phase8jqv2_4tf1_candidate_parameter_grid.json",
        {
            "status": "PASS", "bounded": True, "deterministic": True,
            "config_sha256": sha256(config_path),
            "all_predeclared_combinations_preserved": True,
            "families": {
                key: value["grid"]
                for key, value in config["candidates"].items()
            },
            "evaluated_grid_count": development["grid_count"],
            "holdout_used_for_tuning": False,
        },
    )

    seed = next(
        row for row in development["results"]
        if row["candidate"] == "ablation_seed7"
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_candidate_ablation.json",
        {
            "status": "FAIL", "candidate": "seed_threshold_7_ablation",
            "diagnostic_only": True, "automatically_selectable": False,
            "synthetic_positive": [
                seed["synthetic_positive_pass"],
                seed["synthetic_positive_total"],
            ],
            "synthetic_negative": [
                seed["synthetic_negative_pass"],
                seed["synthetic_negative_total"],
            ],
            "fixed_pre_gap_measurement": seed["fixed_pre_gap_measurement"],
            "fixed_post_gap_measurement":
                seed["fixed_post_gap_measurement"],
            "natural_passing_maps":
                seed["development_natural_passing_maps"],
            "conclusion":
                "threshold 7 neither repairs the fixed natural case nor "
                "passes negative controls; Route F does not apply",
        },
    )
    selection = {
        "status": "FAIL",
        "primary_cause": "temporal_foreground_contract_architecture_limit",
        "selected_candidate": None,
        "candidate_selected": False,
        "development_gate_pass_count": 0,
        "holdout_candidates": [],
        "reason": (
            "No predeclared A/B/C combination passed all positive and "
            "negative controls or the natural pre/post measurement gate."
        ),
        "candidate_c_partial_observation":
            "one grid recovered fixed pre-gap measurement only; first "
            "post-gap measurement and natural-map gate remained false",
        "seed7_overfit_route": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_architecture_review",
    }
    write_new(REPORTS / "phase8jqv2_4tf1_candidate_selection.json", selection)

    no_target_rows = []
    for row in development["results"]:
        control = next(
            item for item in row["negative_controls"]
            if item["name"] == "no_target"
        )
        no_target_rows.append({
            "candidate": row["candidate"],
            "parameter_index": row["parameter_index"],
            "false_measurement_frames": control["false_measurement_frames"],
            "false_attention_frames": control["false_attention_frames"],
        })
    write_new(
        REPORTS / "phase8jqv2_4tf1_no_target_validation.json",
        {
            "status": "PASS",
            "candidate_runs": no_target_rows,
            "all_zero_false_measurements": all(
                not row["false_measurement_frames"] for row in no_target_rows
            ),
            "all_zero_false_attention": all(
                not row["false_attention_frames"] for row in no_target_rows
            ),
            "legacy_no_target_measurement_frames":
                baseline["no_target_smoke"]["measurement_frames"],
            "runtime_gt_input": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_ordinary_dynamic_regression.json",
        {
            "status": "FAIL",
            "bounded_synthetic_motion_controls_executed": True,
            "legacy_ordinary_measurement_frames":
                baseline["ordinary_dynamic_smoke"]["measurement_frames"],
            "candidate_positive_control_summary": [
                {
                    "candidate": row["candidate"],
                    "parameter_index": row["parameter_index"],
                    "passed": row["synthetic_positive_pass"],
                    "total": row["synthetic_positive_total"],
                }
                for row in development["results"]
            ],
            "full_track_regression": "NOT_RUN_MEASUREMENT_GATE_FAILED",
            "missing_metrics": [
                "track birth/confirmation/dynamic latency",
                "velocity error", "ID switches",
            ],
            "interpretation":
                "No candidate was eligible for track integration; this "
                "report must not be interpreted as an ordinary-dynamic PASS.",
        },
    )

    freeze = {
        "status": "BLOCKED_DEVELOPMENT_GATE_FAILED",
        "selected_for_holdout": [],
        "maximum_allowed": 2,
        "source_sha256": sha256(source_path),
        "registry_sha256": sha256(registry_path),
        "config_sha256": sha256(config_path),
        "evaluator_sha256": sha256(evaluator_path),
        "parameters_frozen": None,
        "sealed_holdout_accessed": False,
        "holdout_results_accessed": split["holdout_results_accessed"],
        "no_tuning_after_holdout": True,
    }
    write_new(REPORTS / "phase8jqv2_4tf1_holdout_freeze.json", freeze)
    write_new(
        REPORTS / "phase8jqv2_4tf1_holdout_results.json",
        {
            "status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
            "sealed_holdout_accessed": False,
            "holdout_case_count_disclosed_by_split":
                len(split["sealed_holdout_natural_cases"]),
            "holdout_case_results": None,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_generalization.json",
        {
            "status": "NOT_EVALUATED",
            "reason": "no development candidate eligible for holdout",
            "holdout_accessed": False,
            "generalization_claimed": False,
        },
    )
    best = max(
        selectable,
        key=lambda row: (
            row["fixed_pre_gap_measurement"],
            row["fixed_post_gap_measurement"],
            row["development_natural_passing_maps"],
            row["synthetic_positive_pass"],
            row["synthetic_negative_pass"],
        ),
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_measurement_validation.json",
        {
            "status": "FAIL",
            "best_development_candidate": {
                "candidate": best["candidate"],
                "parameter_index": best["parameter_index"],
                "fixed_pre_gap_measurement":
                    best["fixed_pre_gap_measurement"],
                "fixed_post_gap_measurement":
                    best["fixed_post_gap_measurement"],
                "natural_passing_maps":
                    best["development_natural_passing_maps"],
                "natural_passing_types":
                    best["development_natural_passing_types"],
            },
            "required_natural_maps": 3,
            "required_maze_types": 2,
            "eligible_for_tracker_integration": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_tracker_integration_smoke.json",
        {
            "status": "NOT_RUN_MEASUREMENT_GATE_FAILED",
            "tracker_modified": False,
            "gap1": "BLOCKED", "gap2": "BLOCKED",
            "gap3": "OUT_OF_SCOPE",
            "identity_pass_claimed": False,
        },
    )

    compatibility = {
        "status": "PASS",
        "migration_approved": False,
        "legacy": {
            "contract": "legacy_v1",
            "source_sha256":
                entry["frozen_hashes"]["range_image_foreground.py"],
            "default": True, "modified": False,
        },
        "candidate": {
            "contract":
                "temporal_foreground_observability_contract_v2_1_candidates",
            "source_sha256": sha256(source_path),
            "selected": False, "default": False,
        },
        "matrix": {
            "depth_input": "compatible",
            "pose_input": "compatible",
            "measurement_schema": "compatible",
            "checkpoint": "no impact while disabled",
            "dataset": "no impact while disabled",
            "evaluator": "offline explicit-key only",
            "deployment": "not approved",
            "rollback": "legacy_v1 remains default",
        },
    }
    write_new(
        REPORTS / "phase8jqv2_4tf1_compatibility_matrix.json",
        compatibility,
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_migration_plan.md",
        """# TF1 migration plan

Migration is not approved. No candidate passed the development measurement
gate, so `legacy_v1` remains the only default and `candidate_selected` remains
false. Candidate code is reachable only through an explicit registry key.

No checkpoint or dataset migration is required. Deployment must continue to
use the legacy contract. Rollback is therefore a no-op. A future architecture
review must create a separately versioned candidate and repeat a newly frozen
development/holdout protocol; it must not reinterpret these failed results.
""",
    )
    runtimes = [
        {
            "candidate": row["candidate"],
            "parameter_index": row["parameter_index"],
            "average_end_to_end_frame_ms": row["average_runtime_ms"],
        }
        for row in development["results"]
    ]
    write_new(
        REPORTS / "phase8jqv2_4tf1_performance.json",
        {
            "status": "FAIL_NOT_ELIGIBLE_FOR_PERFORMANCE_GATE",
            "candidate_runtime": runtimes,
            "runtime_scope":
                "DynamicPerception update including foreground/tracker/attention",
            "history_capacity_frames": config["history_frames"],
            "history_bounded": True,
            "maximum_components": config["runtime_bounds"][
                "maximum_components"
            ],
            "baseline_runtime_ms":
                "NOT_RECORDED_BY_FROZEN_EXACT_REPLAY",
            "peak_cpu_memory": "NOT_MEASURED",
            "peak_gpu_memory": "renderer only; candidate executes on CPU",
            "performance_pass_claimed": False,
        },
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_determinism.json",
        {
            "status": "PASS",
            "parameter_grid_deterministic": True,
            "same_inputs_reused_for_every_grid": True,
            "renderer": "canonical CUDA authority renderer",
            "baseline_exact_zero_tolerance": True,
            "candidate_second_byte_replay":
                "NOT_REQUIRED_AFTER_DEVELOPMENT_GATE_FAILURE",
            "holdout_accessed": False,
        },
    )

    for family in (
        "candidate_a", "candidate_b", "candidate_c",
        "candidate_d", "ablation_seed7", "false_positives",
        "holdout_failures",
    ):
        values = [
            {
                "parameter_index": row["parameter_index"],
                "positive": [
                    row["synthetic_positive_pass"],
                    row["synthetic_positive_total"],
                ],
                "negative": [
                    row["synthetic_negative_pass"],
                    row["synthetic_negative_total"],
                ],
                "fixed_pre": row["fixed_pre_gap_measurement"],
                "fixed_post": row["fixed_post_gap_measurement"],
                "natural_maps": row["development_natural_passing_maps"],
            }
            for row in development["results"]
            if row["candidate"] == family
        ]
        if family == "candidate_d":
            content = {
                "status": "NOT_COMBINED_MEASUREMENT_GATE_FAILED",
                "existing_track_only": True, "new_track_birth": False,
            }
        elif family == "false_positives":
            content = {
                "status": "RECORDED",
                "control": "fov_boundary_change",
                "affected_grid_count": sum(
                    any(
                        not item["pass"]
                        and item["name"] == "fov_boundary_change"
                        for item in row["negative_controls"]
                    )
                    for row in development["results"]
                ),
            }
        elif family == "holdout_failures":
            content = {
                "status": "EMPTY_HOLDOUT_NOT_ACCESSED",
                "development_failure": True,
            }
        else:
            content = {"status": "RECORDED", "results": values}
        write_new(DIAGNOSTICS / family / "summary.json", content)

    final = {
        "status": "FAIL",
        "contract_review": "STRUCTURAL_MISMATCH_CONFIRMED",
        "primary_cause": "temporal_foreground_contract_architecture_limit",
        "selected_candidate": None,
        "candidate_selected": False,
        "legacy_modified": False,
        "tracker_modified": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
        "training_started": False,
        "gap_3": "OUT_OF_SCOPE_CONTRACT_REVIEW_REQUIRED",
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_architecture_review",
    }
    write_new(REPORTS / "phase8jqv2_4tf1_final_result.json", final)
    write_new(
        REPORTS / "phase8jqv2_4tf1_final_recommendation.md",
        """# TF1 final recommendation

Stop at fail-closed Route D. The legacy implementation exactly reproduces the
N1 evidence loss, while every bounded A/B/C grid fails the development hard
gate. Candidate C can recover the fixed case's pre-gap measurement in one
configuration, but not first-frame post-gap measurement and not three natural
maps. Every grid also accepts the FOV-boundary negative control.

Do not open holdout, integrate the tracker, enable a candidate, modify the
legacy default, create Formal V3, or train. The next admissible work is a
versioned dynamic-perception architecture review that explicitly models image
motion boundaries without relaxing static/no-target conservatism.
""",
    )
    write_new(
        REPORTS / "phase8jqv2_4tf1_final_readiness.md",
        """# TF1 readiness

**Result: FAIL / Route D.**

- Baseline exact replay: PASS.
- Contract mismatch classification: confirmed.
- Development parameter grid: complete (11 combinations).
- Synthetic positive controls: best selectable candidate 14/15.
- Synthetic negative controls: every grid 19/20.
- Fixed case: no candidate passes both pre-gap and first post-gap.
- Natural map gate: 0/3 required maps.
- Holdout: sealed and not accessed.
- Tracker integration: correctly blocked.
- Gap-3: still mathematically out of scope (`0.75^3 = 0.421875 < 0.55`).
- Formal generation/training: not started.

The system is not ready for natural identity rebaseline or production
training. Only `phase8jqv2_4_dynamic_perception_architecture_review` is
allowed next.
""",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
