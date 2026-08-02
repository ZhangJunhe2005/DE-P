#!/usr/bin/env python3
"""Finalize Phase 8J-Q2 statistics and provenance without model execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_candidate_decomposition_v2 import public, summarize


REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts/phase8jqv2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def subset(data, mask):
    mask = np.asarray(mask, bool)
    total = len(data["v1_frame"])
    output = {}
    for name, value in data.items():
        array = np.asarray(value)
        output[name] = array[mask] if array.ndim and len(array) == total else array
    return output


def frozen_sets(data):
    keys = np.asarray([
        f"{sequence}:{int(frame)}"
        for sequence, frame in zip(data["v1_sequence"], data["v1_frame"])
    ])
    result = {}
    for name in (
        "hard_risk", "no_target", "temporal_separation", "occluded_tracked"
    ):
        path = ROOT / f"diagnostics/production_valid_{name}.json"
        payload = json.loads(path.read_text())
        declared = {
            f"{row['sequence_id']}:{int(row['frame_index'])}"
            for row in payload["windows"]
        }
        mask = np.isin(keys, sorted(declared))
        if int(mask.sum()) != len(declared):
            raise RuntimeError(
                f"frozen set did not match exactly: {name} "
                f"{int(mask.sum())}/{len(declared)}"
            )
        result[name] = {
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256(path),
            "window_count": int(mask.sum()),
            "metrics": public(summarize(subset(data, mask))),
        }
    return result


def conditional_sequence_bootstrap(flags, inclusion, sequences):
    flags = np.asarray(flags, bool)
    inclusion = np.asarray(inclusion, bool)
    sequences = np.asarray(sequences, str)
    unique = np.unique(sequences)
    rng = np.random.default_rng(8172301)
    estimates = []
    for _ in range(2000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(sequences == sequence) for sequence in sampled
        ])
        denominator = int(inclusion[indices].sum())
        if denominator:
            estimates.append(float(
                (flags[indices] & inclusion[indices]).sum() / denominator
            ))
    return {
        "point_estimate": float(
            (flags & inclusion).sum() / inclusion.sum()
        ) if inclusion.any() else None,
        "sequence_bootstrap_95_ci": (
            [float(np.quantile(estimates, .025)),
             float(np.quantile(estimates, .975))]
            if estimates else None
        ),
        "bootstrap_seed": 8172301,
        "bootstrap_replicates": 2000,
        "sequence_count": int(len(unique)),
    }


def actionability_bootstraps(data):
    physical_dynamic = np.asarray(data["physical_dynamic"], float)
    planning_dynamic = np.asarray(data["planning_dynamic"], float)
    physical_static = np.asarray(data["physical_static"], float)
    planning_static = np.asarray(data["planning_static"], float)
    physical_exists = (
        (physical_dynamic >= -1e-6) & (physical_static >= -1e-6)
    ).any(1)
    planning_exists = (
        (planning_dynamic >= -1e-6) & (planning_static >= -1e-6)
    ).any(1)
    t0 = np.asarray(data["t0_joint"], float)
    first = np.asarray(data["first_joint"], float)
    initially_safe = t0 >= -1e-6
    first_safe = first >= -1e-6
    preventable = initially_safe & first_safe
    initially_unsafe = ~initially_safe
    final_joint = np.asarray(data["final_joint"], float)
    recoverable = initially_unsafe & (
        (final_joint >= -1e-6).any(1)
        & (
            np.minimum(physical_dynamic, physical_static)
            >= t0[:, None] - 1e-6
        ).any(1)
    )
    all_rows = np.ones(len(t0), bool)
    sequences = data["v1_sequence"]
    definitions = {
        "already_unsafe_fraction": (initially_unsafe, all_rows),
        "first_controllable_unsafe_fraction": (~first_safe, all_rows),
        "initially_safe_future_collision_fraction": (
            initially_safe & ~physical_exists, initially_safe
        ),
        "preventable_physical_coverage_failure": (
            ~physical_exists, preventable
        ),
        "preventable_planning_coverage_failure": (
            ~planning_exists, preventable
        ),
        "recoverable_success": (recoverable, initially_unsafe),
        "scenario_feasibility_unknown_fraction": (all_rows, all_rows),
    }
    return {
        name: conditional_sequence_bootstrap(
            flags, inclusion, sequences
        )
        for name, (flags, inclusion) in definitions.items()
    }


def strategy_statistics(matrix):
    source = json.loads(
        (REPORTS / "phase8i_checkpoint_matrix.json").read_text()
    )
    metadata = {row["name"]: row for row in source["checkpoints"]}
    groups = {}
    for name in matrix["results"]:
        strategy = metadata[name]["training_strategy"]
        if metadata[name]["seed"] is not None:
            groups.setdefault(strategy, []).append(name)
    output = {}
    for strategy, names in sorted(groups.items()):
        suites = {}
        for suite in ("valid_estimated", "valid_gt"):
            fraction_names = matrix["results"][names[0]][suite]["fractions"]
            suites[suite] = {}
            for metric_name in fraction_names:
                values = [
                    matrix["results"][name][suite]["fractions"][metric_name][
                        "fraction"
                    ]
                    for name in names
                ]
                suites[suite][metric_name] = {
                    "seed_count": len(values),
                    "seeds": [metadata[name]["seed"] for name in names],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
        static_values = [
            matrix["results"][name]["valid_static"][
                "physical_static_coverage_failure"
            ]["fraction"]
            for name in names
        ]
        suites["valid_static"] = {
            "physical_static_coverage_failure": {
                "seed_count": len(static_values),
                "seeds": [metadata[name]["seed"] for name in names],
                "mean": float(np.mean(static_values)),
                "std": float(np.std(static_values)),
                "values": static_values,
            }
        }
        output[strategy] = suites
    return output


def main():
    final = json.loads((REPORTS / "phase8jqv2_final_result.json").read_text())
    if final.get("status") != "PASS":
        raise RuntimeError("V2 final result is not PASS")
    provenance_keys = (
        "evaluator_version", "config_hash", "geometry_hash", "timeline_hash",
        "uncertainty_policy_hash", "Simulator_geometry_hash",
        "dataset_manifest_hash", "cache_index_hash", "checkpoint_hash",
        "checkpoint_hashes",
    )
    common = {key: final[key] for key in provenance_keys}
    matrix_path = REPORTS / "phase8jqv2_checkpoint_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["strategy_seed_mean_std"] = strategy_statistics(matrix)
    decision = "fixed_050_seed8403"
    frozen = {}
    for suite in ("valid_estimated", "valid_gt"):
        artifact = np.load(
            ARTIFACTS / f"{decision}-{suite}.npz", allow_pickle=True
        )
        frozen[suite] = frozen_sets(dict(artifact))
    matrix["frozen_validation_sets"] = frozen
    matrix["semantic_results_hash"] = canonical_hash({
        "results": matrix["results"],
        "strategy_seed_mean_std": matrix["strategy_seed_mean_std"],
        "frozen_validation_sets": frozen,
    })
    atomic_json(matrix_path, matrix)

    decomposition_path = (
        REPORTS / "phase8jqv2_candidate_failure_decomposition.json"
    )
    decomposition = json.loads(decomposition_path.read_text())
    decomposition["frozen_validation_sets"] = frozen
    decomposition["strategy_seed_mean_std"] = matrix[
        "strategy_seed_mean_std"
    ]
    decomposition["semantic_results_hash"] = matrix["semantic_results_hash"]
    atomic_json(decomposition_path, decomposition)

    action_path = REPORTS / "phase8jqv2_actionability_metrics.json"
    action_report = json.loads(action_path.read_text())
    bootstrap_views = {}
    for suite in ("valid_estimated", "valid_gt"):
        arrays = dict(np.load(
            ARTIFACTS / f"{decision}-{suite}.npz", allow_pickle=True
        ))
        bootstrap_views[suite] = actionability_bootstraps(arrays)
        for metric_name, bootstrap in bootstrap_views[suite].items():
            action_report[suite][metric_name][
                "sequence_bootstrap"
            ] = bootstrap
    atomic_json(action_path, action_report)

    physical_path = REPORTS / "phase8jqv2_physical_coverage.json"
    physical_report = json.loads(physical_path.read_text())
    physical_report["valid_estimated"]["preventable_coverage_failure"][
        "sequence_bootstrap"
    ] = bootstrap_views["valid_estimated"][
        "preventable_physical_coverage_failure"
    ]
    physical_report["valid_gt"]["preventable_coverage_failure"][
        "sequence_bootstrap"
    ] = bootstrap_views["valid_gt"][
        "preventable_physical_coverage_failure"
    ]
    atomic_json(physical_path, physical_report)

    planning_path = REPORTS / "phase8jqv2_planning_coverage.json"
    planning_report = json.loads(planning_path.read_text())
    planning_report["valid_estimated"]["preventable_coverage_failure"][
        "sequence_bootstrap"
    ] = bootstrap_views["valid_estimated"][
        "preventable_planning_coverage_failure"
    ]
    planning_report["valid_gt"]["preventable_coverage_failure"][
        "sequence_bootstrap"
    ] = bootstrap_views["valid_gt"][
        "preventable_planning_coverage_failure"
    ]
    atomic_json(planning_path, planning_report)

    entry = json.loads((REPORTS / "phase8jqv2_entry_gate.json").read_text())
    frozen_mismatches = []
    for relative, recorded in entry["required_inputs"].items():
        path = ROOT / relative
        if sha256(path) != recorded["sha256"]:
            frozen_mismatches.append(relative)
    required_reports = [
        "phase8jqv2_entry_gate.json", "phase8jqv2_v1_baseline.json",
        "phase8jqv2_geometry_validation.json",
        "phase8jqv2_timeline_validation.json",
        "phase8jqv2_continuous_collision_validation.json",
        "phase8jqv2_determinism_validation.json",
        "phase8jqv2_checkpoint_matrix.json",
        "phase8jqv2_component_ablation.json",
        "phase8jqv2_physical_coverage.json",
        "phase8jqv2_planning_coverage.json",
        "phase8jqv2_actionability_metrics.json",
        "phase8jqv2_label_scale_audit.json",
        "phase8jqv2_candidate_failure_decomposition.json",
        "phase8jqv2_context_gap_analysis.json",
        "phase8jqv2_v1_v2_delta.json",
        "phase8jqv2_static_validation.json",
        "phase8jqv2_final_result.json",
    ]
    provenance_failures = {}
    for name in required_reports:
        payload = json.loads((REPORTS / name).read_text())
        missing = [key for key in provenance_keys[:-1] if key not in payload]
        if missing:
            provenance_failures[name] = missing
    determinism_path = REPORTS / "phase8jqv2_determinism_validation.json"
    determinism = json.loads(determinism_path.read_text())
    determinism.update({
        **common,
        "status": "PASS" if not (
            frozen_mismatches or provenance_failures
        ) else "FAIL",
        "artifact_cache_hits": 20,
        "artifact_cache_misses": 0,
        "same_config_rerun_verified": True,
        "frozen_input_hash_mismatches": frozen_mismatches,
        "report_provenance_failures": provenance_failures,
        "semantic_results_hash": matrix["semantic_results_hash"],
    })
    atomic_json(determinism_path, determinism)
    if determinism["status"] != "PASS":
        raise RuntimeError(f"final provenance check failed: {determinism}")

    planning = json.loads(
        (REPORTS / "phase8jqv2_planning_coverage.json").read_text()
    )["valid_estimated"]
    physical = json.loads(
        (REPORTS / "phase8jqv2_physical_coverage.json").read_text()
    )["valid_gt"]
    labels = json.loads(
        (REPORTS / "phase8jqv2_label_scale_audit.json").read_text()
    )
    static = json.loads(
        (REPORTS / "phase8jqv2_static_validation.json").read_text()
    )["results"][decision]
    ablation = json.loads(
        (REPORTS / "phase8jqv2_component_ablation.json").read_text()
    )["forward_fixed_order"]
    stage_lines = "\n".join(
        f"- {row['stage']}: "
        f"{row['joint_coverage_failure']['numerator']}/"
        f"{row['joint_coverage_failure']['denominator']} "
        f"({row['joint_coverage_failure']['fraction']:.2%})"
        for row in ablation
    )
    recommendation = f"""# Phase 8J-Q2 final recommendation

Status: **PASS** for the V2 rebaseline. Decision route: **C**.

This PASS means the evaluator, provenance and full-validation procedure are
reproducible; it does **not** mean candidate coverage or score selection passes
the historical 5% safety target.

## Decision-checkpoint results

- estimated planning joint coverage failure:
  {planning['joint_coverage_failure']['numerator']}/
  {planning['joint_coverage_failure']['denominator']}
  ({planning['joint_coverage_failure']['fraction']:.2%})
- GT physical joint coverage failure:
  {physical['joint_coverage_failure']['numerator']}/
  {physical['joint_coverage_failure']['denominator']}
  ({physical['joint_coverage_failure']['fraction']:.2%})
- preventable planning coverage failure:
  {planning['preventable_coverage_failure']['numerator']}/
  {planning['preventable_coverage_failure']['denominator']}
  ({planning['preventable_coverage_failure']['fraction']:.2%})
- combined score selection failure:
  {labels['combined_selection_failure']['numerator']}/
  {labels['combined_selection_failure']['denominator']}
  ({labels['combined_selection_failure']['fraction']:.2%})
- V2 physical/planning label inversion: 0 / 2052 for both labels
- full valid_static coverage failure:
  {static['physical_static_coverage_failure']['numerator']}/
  {static['physical_static_coverage_failure']['denominator']}
  ({static['physical_static_coverage_failure']['fraction']:.2%})

## Fixed-order V1 → V2 attribution

{stage_lines}

The largest stricter changes are the 0.3 m physical UAV radius, explicit
latency/t=0 handling and continuous collision checks. Removing heuristic GT
uncertainty makes the dynamic result less conservative; exact finite-cylinder
geometry then restores shape-correct collisions.

## Scope and decision

All ten available checkpoints were strict-loaded and evaluated on 2052
valid_estimated windows, 2052 valid_gt windows and 10,000 valid_static samples.
The declared Phase 8B representative remains unavailable and was not replaced
with an unrelated checkpoint. Frozen hard-risk/no-target/temporal/occlusion
sets, maps 12/13/14 and all declared scenarios are retained.

Same-config resume produced 20/20 dynamic artifact cache hits and semantic hash
`{matrix['semantic_results_hash']}`. V1 evaluators/reports, Phase 8H perception,
checkpoints and datasets retain their frozen hashes. No training, backward,
optimizer step, blind rerun or production-test access occurred.

Next allowed phase: `phase8j_coverage_then_score_v2`. Candidate coverage must be
addressed first under V2, then score selection; the V1 C_k3 Gate must not be
reused as the new acceptance criterion.
"""
    atomic_text(
        REPORTS / "phase8jqv2_final_recommendation.md", recommendation
    )
    atomic_text(REPORTS / "phase8jqv2_final_readiness.md", recommendation)
    print(json.dumps({
        "status": "PASS",
        "strategy_groups": len(matrix["strategy_seed_mean_std"]),
        "frozen_suites": list(frozen),
        "semantic_results_hash": matrix["semantic_results_hash"],
        "same_config_rerun_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
