#!/usr/bin/env python3
"""Authorize the visibility-retry hotfix after validating existing units."""

import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authoritative_dataset.generate_v1 import (
    append_journal, atomic_json, sha256, verify_manifest,
)


def main():
    dataset = ROOT/"data/phase8_authoritative_v1"
    locks = list((dataset/"generation_state").rglob("*.lock"))
    if locks:
        raise RuntimeError(f"generation is active: {locks}")
    sweep = json.loads(
        (ROOT/"reports/phase8jqv2_4_visibility_sweep.json").read_text())
    if sweep["status"] != "PASS":
        raise RuntimeError("visibility sweep did not pass")
    frozen = json.loads(
        (ROOT/"reports/phase8jqv2_4_generation_plan.json").read_text())
    plan_path = dataset/"generation_state/generation_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["source_hash"] != frozen["source_hash"]:
        raise RuntimeError("base source hash mismatch")
    completed = []
    for state_path in sorted(
        (dataset/"generation_state/sequence_state").glob("*.json")
    ):
        state = json.loads(state_path.read_text())
        if state["status"] != "complete":
            raise RuntimeError(f"non-complete state: {state_path}")
        manifest = dataset/state["unit_manifest"]
        if not verify_manifest(dataset, manifest):
            raise RuntimeError(f"unit verification failed: {manifest}")
        completed.append(json.loads(manifest.read_text()))
    for state_path in sorted(
        (dataset/"generation_state/map_state").glob("*.json")
    ):
        state = json.loads(state_path.read_text())
        if not verify_manifest(
            dataset, dataset/state["unit_manifest"]
        ):
            raise RuntimeError(f"map verification failed: {state_path}")
    current = {
        name: sha256(name)
        for name in frozen["source_hashes"]}
    changed = {
        name: value for name, value in current.items()
        if value != frozen["source_hashes"][name]}
    allowed = {
        str(ROOT/"authoritative_dataset/generate_v1.py"),
        str(ROOT/"scripts/phase8jqv2_4_preflight.sh"),
    }
    if set(changed) != allowed:
        raise RuntimeError(
            f"hotfix touches unauthorized frozen sources: {changed}")
    hotfix_payload = {
        "hotfix_id": "phase8jqv2_4_visibility_retry_v1",
        "base_source_hash": frozen["source_hash"],
        "base_config_hash": frozen["config_hash"],
        "renderer_version": plan["renderer_version"],
        "authorized_source_hashes": changed,
        "semantic_scope": [
            "bounded deterministic post-render actor visibility resampling",
            "smaller safe in-FOV actor distance range",
            "complete CUDA failure journal records",
            "generation plan permits this explicit additive hotfix record",
            "shard completion cannot impersonate full split completion",
        ],
        "completed_sequences_verified": len(completed),
        "completed_frames_verified":
            sum(row["frame_count"] for row in completed),
        "completed_sequence_hashes_unchanged": True,
        "authority_artifacts_unchanged": True,
        "renderer_implementation_unchanged": True,
        "network_weights_modified": False,
        "training_executed": False,
        "visibility_sweep_hash": sha256(
            ROOT/"reports/phase8jqv2_4_visibility_sweep.json"),
    }
    hotfix_hash = hashlib.sha256(json.dumps(
        hotfix_payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    hotfix_record = {
        "hotfix_id": hotfix_payload["hotfix_id"],
        "hotfix_hash": hotfix_hash,
        "applied_ns": time.time_ns(),
        "host": socket.gethostname(),
    }
    existing = plan.get("compatible_hotfixes", [])
    if existing and existing[0]["hotfix_id"] == hotfix_record["hotfix_id"] \
            and existing[0]["hotfix_hash"] == hotfix_hash:
        hotfix_record = existing[0]
    plan["compatible_hotfixes"] = [hotfix_record]
    atomic_json(plan_path, plan)
    q23 = json.loads(
        (ROOT/"reports/phase8jqv2_3_final_result.json").read_text())
    report = {
        **{key: q23[key] for key in (
            "Simulator_geometry_hash", "cache_index_hash",
            "checkpoint_hash", "config_hash", "dataset_manifest_hash",
            "evaluator_version", "geometry_hash", "timeline_hash",
            "uncertainty_policy_hash")},
        **hotfix_payload, **hotfix_record, "status": "PASS",
    }
    atomic_json(
        ROOT/"reports/phase8jqv2_4_visibility_hotfix.json", report)
    append_journal(dataset, {
        "event": "compatible_hotfix_applied", "status": "complete",
        **hotfix_record,
        "completed_sequences_verified": len(completed),
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
