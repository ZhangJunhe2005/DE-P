#!/usr/bin/env python3
"""Migrate only the zero-sequence V4 plan made by the latency-key bug."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/route_a_v4_raw_static"
CONFIG = ROOT / "configs/route_a_v4_raw_static_resolved.yaml"
BUGGY_CONFIG_HASH = (
    "2a5e8471617713dc77299e26f9c91635e5d5daa9be9d027300b4a4475578bc55"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan_path = DATASET / "generation_state/generation_plan.json"
    if not plan_path.is_file():
        print(json.dumps({"status": "NOT_NEEDED", "reason": "no_existing_plan"}))
        return
    plan = json.loads(plan_path.read_text())
    config = yaml.safe_load(CONFIG.read_text())
    new_hash = sha(CONFIG)
    if plan["config_hash"] == new_hash:
        print(json.dumps({"status": "NOT_NEEDED", "reason": "already_current"}))
        return
    sequence_manifests = list(
        (DATASET / "manifests/sequences").glob("*.json")
    )
    sequence_states = list(
        (DATASET / "generation_state/sequence_state").glob("*.json")
    )
    completions = list(
        (DATASET / "generation_state/completion").glob("*COMPLETE*")
    )
    locks = list((DATASET / "generation_state/locks").glob("*.lock"))
    checks = {
        "dataset_version": (
            plan.get("dataset_version") == "route_a_v4_raw_static_v1"
            and config.get("dataset_version") == "route_a_v4_raw_static_v1"
        ),
        "known_buggy_config_hash":
            plan.get("config_hash") == BUGGY_CONFIG_HASH,
        "source_hash_unchanged":
            plan.get("source_hash") == config["frozen_hashes"]["source_hash"],
        "split_hash_unchanged": (
            plan.get("split_manifest_hash")
            == config["frozen_hashes"]["split_manifest_hash"]
        ),
        "command_latency_restored":
            config["state_sampling_contract"].get("command_latency_s") == 0.04,
        "zero_sequence_manifests": not sequence_manifests,
        "zero_sequence_states": not sequence_states,
        "zero_completion_markers": not completions,
        "no_active_locks": not locks,
    }
    result = {
        "status": "READY_TO_APPLY" if all(checks.values()) else "REFUSED",
        "checks": checks,
        "old_config_hash": plan.get("config_hash"),
        "new_config_hash": new_hash,
        "preserved_map_states": len(list(
            (DATASET / "generation_state/map_state").glob("*.json")
        )),
        "preserved_sequence_count": 0,
    }
    if not all(checks.values()):
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)
    if not args.apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    archive = (
        DATASET / "generation_state/migrations/"
        "generation_plan_before_command_latency_repair.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        atomic_json(archive, plan)
    plan["config_hash"] = new_hash
    plan.setdefault("compatible_hotfixes", []).append({
        "id": "route_a_v4_command_latency_contract_repair_v1",
        "scope": "restore inherited command_latency_s before first sequence",
        "existing_sequence_semantics_changed": False,
        "completed_sequence_count_at_migration": 0,
        "old_config_hash": BUGGY_CONFIG_HASH,
        "new_config_hash": new_hash,
    })
    atomic_json(plan_path, plan)
    result["status"] = "PASS"
    result["archived_plan"] = str(archive)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
