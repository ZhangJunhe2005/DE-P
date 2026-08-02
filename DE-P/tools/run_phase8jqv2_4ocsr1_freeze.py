#!/usr/bin/env python3
"""Verify OCSR1 entry and freeze the validation boundary exactly once."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from tools.run_phase8jqv2_4ocsr1_collect import SCENARIOS, split_for


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"validation freeze artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def all_case_ids():
    controls = json.loads((
        ROOT / "data/phase8_dynamic_perception_controls_v1/manifest.json"
    ).read_text())
    ids = []
    for control_id in controls["control_ids"]:
        control = json.loads((
            ROOT / "data/phase8_dynamic_perception_controls_v1/controls"
            / control_id / "control.json"
        ).read_text())
        if control["split"] == "development":
            ids.append(control_id)
    selected = defaultdict(list)
    for path in sorted((
        ROOT / "data/phase8_dynamic_production/sequences"
    ).glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        scenario = metadata["scenario_type"]
        if scenario in SCENARIOS and len(selected[scenario]) < 3:
            selected[scenario].append(path.parent.name)
    for scenario in sorted(selected):
        ids.extend(selected[scenario])
    return ids


def main():
    tccr = json.loads((
        REPORTS / "phase8jqv2_4tccr1_final_result.json"
    ).read_text())
    gap2 = json.loads((
        REPORTS / "phase8jqv2_4tccr1_gap2_classification.json"
    ).read_text())
    regression = json.loads((
        REPORTS / "phase8jqv2_4tccr1_regression.json"
    ).read_text())
    expected = {
        "status": tccr["status"] == "PASS",
        "route_b": tccr["route"] == "B",
        "confirmed_dynamic_tracks":
            tccr["confirmed_dynamic_tracks"] == 33,
        "gap1_events": tccr["real_gap1_enter_events"] == 26,
        "gap2_events": gap2["events"] == 16,
        "negative_sequences": tccr["negative_sequences"] == 15,
        "static_checkpoint_strict_load":
            tccr["static_checkpoint_strict_load"],
        "mode_a_b_c": tccr["mode_a_b_c"] == "PASS",
        "formal_command_unmodified":
            not tccr.get("formal_command_modified", False),
        "track_manager_unmodified":
            not tccr["TrackManager_algorithm_modified"],
        "detector_unmodified": not tccr["detector_algorithm_modified"],
        "training_not_run": not tccr["training_started"],
        "holdout_test_blind_not_accessed": not any((
            tccr["holdout_accessed"], tccr["production_test_accessed"],
            tccr["blind_accessed"],
        )),
        "checkpoint_hash_frozen":
            regression["static_checkpoint_hash_unchanged"],
    }
    entry = {
        "status": "PASS" if all(expected.values()) else "FAIL",
        "checks": expected,
        "natural_eosr1_tracker_gate": "FAIL_SEPARATE",
        "formal_v3_entry_created": False,
    }
    if entry["status"] != "PASS":
        raise RuntimeError(f"OCSR1 entry gate failed: {entry}")

    ids = all_case_ids()
    split = {
        "status": "PASS",
        "method": "sha256(case_id) integer modulo 3",
        "validation_bucket": 0,
        "case_grouped": True,
        "frame_random_split": False,
        "calibration_case_ids": [
            case for case in ids if split_for(case) == "calibration"
        ],
        "validation_case_ids": [
            case for case in ids if split_for(case) == "validation"
        ],
        "overlap": [],
        "holdout_accessed": False,
    }
    artifacts = [
        "controller/dynamic_safety_shadow_adapter_v1.py",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/physical_control_residual_v1.py",
        "policy/dep_network.py",
        "saved/DEP_0/epoch10.pth",
        "reports/phase8jqv2_4tccr1_final_result.json",
        "diagnostics/phase8jqv2_4tccr1/runtime_telemetry.jsonl",
        "diagnostics/phase8jqv2_4tccr1/case_summary.json",
        "reports/phase8jqv2_4eosr1_proof_witness_manifest.json",
        "configs/natural_short_occlusion_contract_v3_candidate.yaml",
    ]
    frozen = {
        "status": "PASS",
        "artifacts": {
            path: digest(ROOT / path) for path in artifacts
        },
        "tccr1_adapter_v1_frozen": True,
        "track_manager_frozen": True,
        "detector_frozen": True,
        "checkpoint_frozen": True,
        "eosr1_witnesses_frozen": True,
    }
    calibration = (
        ROOT / "diagnostics/phase8jqv2_4ocsr1/calibration_prediction_summary.json"
    )
    if not calibration.is_file():
        raise RuntimeError("calibration summary must exist before validation freeze")
    contract = (
        ROOT / "configs/occlusion_coasting_safety_contract_v1_candidate.yaml"
    )
    document = yaml.safe_load(contract.read_text())
    if not document["selected_parameters"]["frozen"]:
        raise RuntimeError("selected parameters are not frozen")
    validation_freeze = {
        "status": "PASS",
        "frozen_before_validation": True,
        "contract_source_hash": digest(
            ROOT / "controller/dynamic_safety_shadow_adapter_v2.py"
        ),
        "config_hash": digest(contract),
        "adapter_hash": digest(
            ROOT / "controller/dynamic_safety_shadow_adapter_v2.py"
        ),
        "gt_evaluator_hash": digest(
            ROOT / "tools/evaluate_yopo_dynamic_candidate_risk_v1.py"
        ),
        "collector_hash": digest(
            ROOT / "tools/run_phase8jqv2_4ocsr1_collect.py"
        ),
        "calibration_summary_hash": digest(calibration),
        "selected_parameters": document["selected_parameters"],
        "risk_thresholds":
            document["predeclared_validation_gates"],
        "validation_accessed_at_freeze": False,
        "post_validation_tuning_allowed": False,
    }
    write_once(REPORTS / "phase8jqv2_4ocsr1_entry_gate.json", entry)
    write_once(REPORTS / "phase8jqv2_4ocsr1_frozen_artifacts.json", frozen)
    write_once(REPORTS / "phase8jqv2_4ocsr1_evaluation_split.json", split)
    write_once(
        REPORTS / "phase8jqv2_4ocsr1_validation_freeze.json",
        validation_freeze,
    )
    print(json.dumps({
        "status": "PASS",
        "calibration_cases": len(split["calibration_case_ids"]),
        "validation_cases": len(split["validation_case_ids"]),
        "validation_accessed": False,
        "config_hash": validation_freeze["config_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
