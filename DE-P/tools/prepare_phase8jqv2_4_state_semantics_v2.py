#!/usr/bin/env python3
"""Freeze V2 config while preserving the immutable V1 dataset and maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset import FORMAL_DATASET_VERSION_V2


OLD_ROOT = ROOT/"data/phase8_authoritative_v1"
NEW_ROOT = ROOT/"data/phase8_authoritative_v2"
OLD_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    old_manifest = OLD_ROOT/"manifests/dataset_manifest.json"
    if sha256(old_manifest) != OLD_HASH:
        raise RuntimeError("frozen V1 root manifest changed")
    old_config_path = ROOT/"configs/phase8_authoritative_v1_generation.yaml"
    config = yaml.safe_load(old_config_path.read_text())
    split = {
        "version": "authoritative_formal_split_manifest_v2",
        "derivation": "identical map UUID/seed split as immutable V1",
        "train": config["formal_splits"]["train"]["maps"],
        "valid": config["formal_splits"]["valid"]["maps"],
        "test": [], "blind": [],
        "train_valid_overlap": 0,
        "authority_maps_reused": True,
        "source_dataset": "phase8_authoritative_v1",
        "source_root_manifest_hash": OLD_HASH,
    }
    split_path = (
        ROOT/"reports/phase8jqv2_4_state_semantics_v2_split_manifest.json"
    )
    write_json(split_path, split)
    sources = [
        ROOT/"authoritative_dataset/__init__.py",
        ROOT/"authoritative_dataset/generate_v1.py",
        ROOT/"authoritative_dataset/state_semantics_v2.py",
        ROOT/"authoritative_dataset/cuda_renderer_v1.py",
        ROOT/"authoritative_dataset/exact_backend_v1.py",
        ROOT/"authoritative_dataset/continuous_v1.py",
        ROOT/"geometry_authority/static_v1.py",
        ROOT/"tools/validate_authoritative_state_semantics_v2.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {str(path): sha256(path) for path in sources}
    source_hash = hashlib.sha256(json.dumps(
        source_hashes, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    config["dataset_version"] = FORMAL_DATASET_VERSION_V2
    config["output_root"] = str(NEW_ROOT)
    config["authority_source_root"] = str(OLD_ROOT)
    config["authority_source_dataset"] = "phase8_authoritative_v1"
    config["authority_source_root_manifest_hash"] = OLD_HASH
    config["state_sampling_contract"] = {
        "source": "phase8jqv2_3_state_sampling_contract.json",
        "semantics_version": "authoritative_state_semantics_v2",
        "command_latency_s": .04,
        "frame_state_source": "continuous_reference_trajectory",
        "velocity_source": "position_finite_difference",
        "acceleration_source": "velocity_finite_difference",
        "demonstrated_speed_mps": 2.0,
        "demonstrated_acceleration_mps2": 2.0,
        "network_speed_limit_mps": 6.0,
        "network_acceleration_limit_mps2": 6.0,
        "runtime_random_sampling": False,
    }
    config["goal_distribution"] = {
        "policy": "safe_local_reference_goal_persisted",
        "distance_min_m": 2.0,
        "distance_max_m": 8.0,
        "body_transform": "R_body_from_world@(goal_world-position_world)",
    }
    config["smoke_settings"] = {
        "map_count": 3,
        "frames_per_split": 300,
        "frames_per_sequence": 20,
    }
    config["frozen_hashes"] = {
        "source_hash": source_hash,
        "split_manifest_hash": sha256(split_path),
        "authority_source_root_manifest_hash": OLD_HASH,
    }
    config["parent_git_commit"] = subprocess.check_output(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    config_path = ROOT/"configs/phase8_authoritative_v2_generation.yaml"
    config_path.write_text(yaml.safe_dump(
        config, sort_keys=True, default_flow_style=False
    ))
    result = {
        "status": "PASS",
        "phase": "phase8jqv2_4_state_semantics_repair_prepare",
        "old_dataset_version": "phase8_authoritative_v1",
        "old_root_manifest_hash": OLD_HASH,
        "old_dataset_unchanged": sha256(old_manifest) == OLD_HASH,
        "new_dataset_version": FORMAL_DATASET_VERSION_V2,
        "new_output_root": str(NEW_ROOT),
        "authority_maps_reused": 60,
        "authority_source_root": str(OLD_ROOT),
        "config": str(config_path),
        "config_hash": sha256(config_path),
        "source_hash": source_hash,
        "source_hashes": source_hashes,
        "split_manifest": str(split_path),
        "split_manifest_hash": sha256(split_path),
        "test_enabled": False,
        "blind_enabled": False,
        "training_started": False,
        "optimizer_step_executed": False,
        "network_weights_modified": False,
    }
    write_json(
        ROOT/"reports/phase8jqv2_4_state_semantics_v2_plan.json",
        result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
