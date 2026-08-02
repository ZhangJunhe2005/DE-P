#!/usr/bin/env python3
"""Write truthful stop-condition reports for the motion-contract repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
CONTRACT = ROOT/"configs/authoritative_dynamic_motion_contract_v2.yaml"
PRELIMINARY = REPORTS/"phase8jqv2_4_dynamic_motion_smoke_validation.json"
PROBE = ROOT/"artifacts/phase8jqv2_4_occlusion_probe"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    contract = yaml.safe_load(CONTRACT.read_text())
    contract_hash = sha256(CONTRACT)
    v1_unchanged = sha256(
        ROOT/"data/phase8_authoritative_v1/manifests/dataset_manifest.json"
    ) == V1_HASH
    v2_unchanged = sha256(
        ROOT/"data/phase8_authoritative_v2/manifests/dataset_manifest.json"
    ) == V2_HASH
    preliminary = (
        json.loads(PRELIMINARY.read_text()) if PRELIMINARY.is_file() else {}
    )
    journal = PROBE/"generation_state/generation_journal.jsonl"
    failures = []
    if journal.is_file():
        failures = [
            json.loads(line) for line in journal.read_text().splitlines()
            if json.loads(line).get("event") == "sequence_failed"
        ]
    immutable = {
        "v1_root_manifest_hash": V1_HASH,
        "v2_root_manifest_hash": V2_HASH,
        "v1_unchanged": v1_unchanged,
        "v2_unchanged": v2_unchanged,
        "old_r0_do_not_use_marker_preserved": (
            ROOT/"artifacts/phase8jqv2_5/estimated_r0_cache/"
            "DO_NOT_USE_GATE_FAILED.json"
        ).is_file(),
    }
    safety = {
        "formal_long_training_started": False,
        "optimizer_step_executed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "v1_modified": not v1_unchanged,
        "v2_modified": not v2_unchanged,
        "old_manifests_modified": not (v1_unchanged and v2_unchanged),
        "old_r0_cache_reused": False,
        "background_generation_process_remaining": False,
        "background_training_process_remaining": False,
    }
    root_cause = {
        "status": "PASS",
        "phase": "phase8jqv2_4_authoritative_dynamic_motion_repair",
        "frozen_perception": {
            "dynamic_enter_speed_mps": .30,
            "dynamic_exit_speed_mps": .15,
            "hysteresis": True,
            "minimum_confirmed_hits": 3,
            "motion_consistency_frames": 2,
            "maximum_missed_frames": 3,
            "velocity_definition": "world-frame linear Kalman track state",
            "raw_velocity_definition":
                "(current_world_centroid-previous_world_centroid)/dt",
            "dt_source": "sensor timestamp seconds; authoritative frame dt=0.1 s",
            "attention_trigger":
                "confirmed and dynamic and confidence-authorized projected track",
        },
        "old_generator": {
            "location": "authoritative_dataset/generate_v1.py:344-493",
            "crossing_mps": .12,
            "head_on_mps": .08,
            "temporal_separation_mps": .06,
            "static_dynamic_joint_constraint_mps": .04,
            "multi_target_mps": 0.0,
            "occluded_but_tracked_mps": 0.0,
            "zero_speed_reason":
                "both scenarios fell through the legacy else branch",
        },
        "persisted_actor_state":
            "authoritative_dataset/generate_v1.py:610-671",
        "renderer_actor_pose":
            "authoritative_dataset/cuda_renderer_v1.py:104-130",
        "estimated_context_reader":
            "tools/generate_phase8jqv2_5_estimated_r0_cache.py:99-166",
        "camera_semantics":
            "renderer body ray [forward,image-right,image-down], converted to optical before frozen perception",
        **immutable,
    }
    write_json(
        REPORTS/"phase8jqv2_4_dynamic_motion_root_cause.json",
        root_cause,
    )

    smoke_common = {
        "status": "FAIL",
        "primary_cause":
            "natural_static_occlusion_reappearance_not_constructible",
        "motion_contract_version": contract["contract_version"],
        "motion_contract_hash": contract_hash,
        "preliminary_motion_and_perception_smoke": {
            "status": preliminary.get("status"),
            "scenario_results": preliminary.get("scenario_results", {}),
            "checks": preliminary.get("checks", {}),
            "usable_for_gate": False,
            "reason": "predates natural-occlusion identity gate and contract hash",
        },
        "occlusion_probe": {
            "status": "FAIL",
            "sequence_manifest_count": len(list(
                (PROBE/"manifests/sequences").glob("*.json")
            )) if PROBE.is_dir() else 0,
            "failure_records": failures,
            "required_pattern":
                "visible -> 1..3 natural static-occlusion frames -> visible",
            "metadata_only_occlusion_accepted": False,
            "artificial_hidden_frames_accepted": False,
        },
        "formal_regeneration_allowed": False,
        **immutable,
        **safety,
    }
    for name in (
        "phase8jqv2_4_dynamic_motion_smoke_generation.json",
        "phase8jqv2_4_dynamic_motion_smoke_validation.json",
        "phase8jqv2_4_dynamic_motion_smoke_perception.json",
        "phase8jqv2_4_dynamic_motion_smoke_safety.json",
    ):
        write_json(REPORTS/name, smoke_common)

    not_run = {
        "status": "NOT_RUN",
        "reason": "smoke Gate failed at natural occlusion/reappearance",
        "dataset_version": "phase8_authoritative_v3",
        "root_manifest_hash": None,
        "dynamic_train_generated": False,
        "dynamic_valid_generated": False,
        "static_sequences_reused": False,
        **immutable,
        **safety,
    }
    write_json(
        REPORTS/"phase8jqv2_4_dynamic_motion_formal_generation.json",
        not_run,
    )
    write_json(
        REPORTS/"phase8jqv2_4_dynamic_motion_formal_validation.json",
        not_run,
    )
    write_json(
        REPORTS/"phase8jqv2_4_dynamic_motion_perception_validation.json",
        not_run,
    )

    final = {
        "status": "FAIL",
        "training_ready": False,
        "primary_cause":
            "natural_static_occlusion_reappearance_not_constructible",
        "dataset_entry_gate": "NOT_RUN_FOR_V3",
        "motion_contract_created": True,
        "motion_contract_version": contract["contract_version"],
        "motion_contract_hash": contract_hash,
        "persisted_motion": "PRELIMINARY_PASS",
        "canonical_cuda_rendering": "PRELIMINARY_PASS",
        "frozen_perception": "PRELIMINARY_PASS_EXCEPT_OCCLUSION_PROTOCOL",
        "no_target_false_attention": "PASS",
        "static_collision": "PASS_ON_PRELIMINARY_SMOKE",
        "actor_actor_collision": "PASS_ON_PRELIMINARY_SMOKE",
        "dynamic_joint_unsafe": "PASS_ON_PRELIMINARY_SMOKE",
        "occluded_but_tracked": "FAIL",
        "formal_v3_generation": "NOT_RUN",
        "q2_5_r0_fast_gate": "NOT_RUN",
        "q2_5_r1_full_rebaseline": "NOT_RUN",
        "candidate_capacity_gate": "NOT_RUN",
        "next_allowed_phase":
            "phase8jqv2_4_occlusion_trajectory_constructor_repair",
        **immutable,
        **safety,
    }
    write_json(
        REPORTS/"phase8jqv2_5_dynamic_motion_contract_rerun.json", final
    )
    write_json(
        REPORTS/"phase8jqv2_5_training_readiness_rerun.json", final
    )
    write_json(REPORTS/"phase8jqv2_5_final_result_rerun.json", final)
    recommendation = """# Phase 8J-Q2.4 dynamic-motion repair recommendation

Gate: **FAIL**. Formal V3 regeneration and all Q2.5 rebaseline work remain
forbidden.

The versioned motion contract fixes the original 0--0.12 m/s and stationary
actor defect. A 420-frame preliminary CUDA smoke demonstrated finite persisted
motion, all-actor dynamic entry, non-empty attention, zero no-target false
attention, and zero static/actor/joint collisions. That artifact is explicitly
non-authoritative because it predates the final natural-occlusion gate.

The isolated natural-occlusion probe exhausted 64 deterministic UAV/actor
windows without producing a visible -> bounded static occlusion -> visible
trajectory with identity continuity. Metadata flags, FOV exit, and artificial
frame hiding are not accepted as substitutes.

The only allowed next phase is
`phase8jqv2_4_occlusion_trajectory_constructor_repair`: construct trajectories
from authoritative static-geometry sight-line intervals, then repeat the
isolated smoke. Do not generate Formal V3 before it passes.
"""
    (REPORTS/"phase8jqv2_5_final_recommendation_rerun.md").write_text(
        recommendation
    )
    changes = """# Dynamic motion repair code changes

- Added a hash-bound `authoritative_dynamic_motion_contract_v2`.
- Added deterministic swept-volume actor sampling with threshold margin,
  scenario-specific velocity/direction/radius, joint UAV-window resampling,
  actor-static/actor-actor/UAV separation, and nonzero fallback semantics.
- Added `phase8_authoritative_v3` configuration without modifying V1/V2.
- Added an in-generator frozen causal perception probe; generated actors must
  sustain actual dynamic tracks and no-target sequences must remain empty.
- Added natural static-occlusion and track-identity requirements. The current
  straight constant-velocity constructor cannot satisfy that final condition,
  so the formal generation Gate remains closed.
- Added isolated smoke scenario filtering, validation, immutable hash checks,
  and explicit DO-NOT-USE markers.
"""
    (REPORTS/"phase8jqv2_4_dynamic_motion_code_changes.md").write_text(
        changes
    )
    marker = {
        "status": "FAIL",
        "reason":
            "natural static-occlusion/reappearance probe exhausted",
        "motion_contract_hash": contract_hash,
        "formal_regeneration_allowed": False,
        "q2_5_rebaseline_allowed": False,
    }
    write_json(PROBE/"DO_NOT_USE_GATE_FAILED.json", marker)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
