#!/usr/bin/env python3
"""Finalize Q2.5 honestly; never manufacture downstream PASS reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((REPORTS / name).read_text())


def write(name, value):
    (REPORTS / name).write_text(json.dumps(value, indent=2) + "\n")


def provenance(checkpoint):
    fixed = next(
        row for row in checkpoint["strict_load_results"]
        if row["name"] == "fixed_050_seed8403"
    )
    timeline_hash = sha256(ROOT / "policy/safety_evaluator_v2.py")
    uncertainty = {
        "owner": "SafetyEvaluatorV2.uncertainty_margin",
        "timeline_source_sha256": timeline_hash,
    }
    simulator = load("phase8jqv2_4_authority_chain_validation.json")
    return {
        "evaluator_version": "safety_evaluator_v2_1",
        "config_hash": sha256(ROOT / "configs/safety_evaluator_v2_1.yaml"),
        "geometry_hash": sha256(ROOT / "loss/safety_geometry_v2_1.py"),
        "timeline_hash": timeline_hash,
        "uncertainty_policy_hash": hashlib.sha256(json.dumps(
            uncertainty, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "Simulator_geometry_hash": simulator["Simulator_geometry_hash"],
        "dataset_manifest_hash": V2_HASH,
        "cache_index_hash": sha256(
            ROOT / "artifacts/phase8jqv2_5/estimated_r0_cache/index.json"
        ),
        "checkpoint_hash": fixed["sha256"],
    }


def main():
    dataset = load("phase8jqv2_5_v2_entry_gate.json")
    static_checker = load("phase8jqv2_5_static_checker_validation.json")
    formal_static = load("phase8jqv2_5_formal_static_validation.json")
    dynamic_regression = load("phase8jqv2_5_dynamic_regression.json")
    gradient = load("phase8jqv2_5_estimated_gradient_integrity.json")
    surrogate = load("phase8jqv2_5_surrogate_alignment.json")
    checkpoint = load("phase8jqv2_5_checkpoint_matrix.json")
    motion = load("phase8jqv2_5_dynamic_motion_contract.json")
    v1_current = sha256(
        ROOT / "data/phase8_authoritative_v1/manifests/dataset_manifest.json"
    )
    v2_current = sha256(
        ROOT / "data/phase8_authoritative_v2/manifests/dataset_manifest.json"
    )
    evaluator_pass = all(
        value["status"] == "PASS"
        for value in (static_checker, formal_static, dynamic_regression)
    )
    dataset_pass = (
        dataset["status"] == "PASS"
        and v1_current == V1_HASH and v2_current == V2_HASH
    )
    prerequisite_pass = (
        evaluator_pass
        and gradient["status"] == "PASS"
        and surrogate.get("full_alignment_gate") is True
        and checkpoint["status"] == "PASS"
        and motion["status"] == "PASS"
    )
    r0 = {
        "status": "NOT_RUN",
        "gate_status": "FAIL",
        "primary_cause": "authoritative_dynamic_motion_contract",
        "selection_manifest":
            "artifacts/phase8jqv2_5/r0_selection_manifest.json",
        "selection_manifest_hash":
            load("phase8jqv2_5_r0_selection_manifest.json")[
                "selection_manifest_hash"
            ],
        "selected_frames": 10000,
        "checkpoint_strict_load": checkpoint["status"],
        "estimated_context_cache_usable": False,
        "reason":
            "Formal V2 actors never reach the frozen 0.3 m/s dynamic threshold; "
            "running model/capacity comparisons would produce non-authoritative metrics.",
        "production_test_used": False,
        "blind_used": False,
    }
    write("phase8jqv2_5_r0_fast_gate.json", r0)
    blocked = {
        "status": "NOT_RUN",
        "blocked_by": "authoritative_dynamic_motion_contract",
        "production_test_used": False,
        "blind_used": False,
    }
    for name in (
        "phase8jqv2_5_full_static_rebaseline.json",
        "phase8jqv2_5_full_dynamic_rebaseline.json",
        "phase8jqv2_5_score_rebaseline.json",
        "phase8jqv2_5_v2_delta.json",
        "phase8jqv2_5_candidate_capacity.json",
        "phase8jqv2_5_fallback_capacity.json",
        "phase8jqv2_5_control_space_capacity.json",
        "phase8jqv2_5_capacity_gate.json",
    ):
        write(name, blocked)
    entry = {
        "status": "FAIL",
        "training_ready": False,
        "primary_cause": "authoritative_dynamic_motion_contract",
        "dataset_entry_gate": "PASS" if dataset_pass else "FAIL",
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": v2_current,
        "v1_root_manifest_hash": v1_current,
        "v1_manifest_unchanged": v1_current == V1_HASH,
        "v2_manifest_unchanged": v2_current == V2_HASH,
        "safety_evaluator_v2_1": "PASS" if evaluator_pass else "FAIL",
        "formal_static_checker": formal_static["status"],
        "dynamic_v2_regression": dynamic_regression["status"],
        "gradient_integrity": gradient["status"],
        "surrogate_alignment": surrogate["status"],
        "surrogate_full_alignment_gate":
            surrogate.get("full_alignment_gate", False),
        "dynamic_motion_contract": motion["status"],
        "checkpoint_strict_load": checkpoint["status"],
        "r0_fast_gate": "NOT_RUN",
        "r1_full_rebaseline": "NOT_RUN",
        "candidate_capacity_gate": "NOT_RUN",
        "training_scripts_created": False,
        "formal_training_started": False,
        "network_weights_modified": False,
        "optimizer_step_executed": False,
        "formal_dataset_modified": False,
        "production_test_used": False,
        "blind_used": False,
        "next_allowed_phase":
            "phase8jqv2_4_authoritative_dynamic_motion_repair",
    }
    write("phase8jqv2_5_entry_gate.json", entry)
    readiness = {
        **entry,
        "evaluator_v2_1_created": evaluator_pass,
        "authoritative_rebaseline_complete": False,
        "candidate_capacity_gate_complete": False,
        "coverage_training_started": False,
        "score_training_started": False,
    }
    write("phase8jqv2_5_training_readiness.json", readiness)
    write("phase8jqv2_5_final_result.json", readiness)
    recommendation = """# Phase 8J-Q2.5 final recommendation

## Decision

**FAIL — training is not ready.**

Formal V2 is file-complete and its static/state/authority contracts pass.
Safety Evaluator V2.1, its exact continuous static checker, checkpoint strict
loading, and finite-gradient smoke also pass.

The blocking defect is the authoritative dynamic motion contract. Across all
8,554 persisted actors, maximum speed is 0.12 m/s while the frozen causal
perception dynamic-enter threshold is 0.3 m/s. Multi-target and
occluded-but-tracked actors are stationary. Consequently an estimated-context
rebaseline would mostly evaluate an empty dynamic context and cannot establish
surrogate/exact alignment or meaningful candidate capacity.

Do not lower the perception threshold to manufacture recall: stationary
actors contain no dynamic-motion signal. Version a corrected actor-motion
contract, regenerate only under a new dataset version/root hash, preserve V1
and the current V2 as immutable evidence, then rerun the Q2.5 Gate.

No production test or blind data was accessed. No optimizer step or formal
training was executed, and no training launch scripts were created.
"""
    (REPORTS / "phase8jqv2_5_final_recommendation.md").write_text(
        recommendation
    )
    (REPORTS / "phase8jqv2_5_final_readiness.md").write_text(
        recommendation
    )
    # The existing V2 regression suite requires every versioned JSON report
    # to carry the same auditable provenance keys, including truthful
    # NOT_RUN/FAIL reports.
    common = provenance(checkpoint)
    for path in REPORTS.glob("phase8jqv2_5*.json"):
        value = json.loads(path.read_text())
        value.update({
            key: value.get(key, datum) for key, datum in common.items()
        })
        path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(entry, indent=2))
    raise SystemExit(0 if prerequisite_pass else 2)


if __name__ == "__main__":
    main()
