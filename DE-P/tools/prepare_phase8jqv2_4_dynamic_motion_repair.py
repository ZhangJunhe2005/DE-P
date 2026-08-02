#!/usr/bin/env python3
"""Freeze the V3 motion-contract config without touching immutable V1/V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
CONTRACT = ROOT/"configs/authoritative_dynamic_motion_contract_v2.yaml"
V3_CONFIG = ROOT/"configs/phase8_authoritative_v3_generation.yaml"
SMOKE_CONFIG = ROOT/"configs/phase8_authoritative_v3_motion_smoke.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(
        value, sort_keys=True, default_flow_style=False
    ))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    v1_manifest = ROOT/"data/phase8_authoritative_v1/manifests/dataset_manifest.json"
    v2_manifest = ROOT/"data/phase8_authoritative_v2/manifests/dataset_manifest.json"
    if sha256(v1_manifest) != V1_HASH or sha256(v2_manifest) != V2_HASH:
        raise RuntimeError("immutable V1/V2 manifest changed")
    config = yaml.safe_load(
        (ROOT/"configs/phase8_authoritative_v2_generation.yaml").read_text()
    )
    contract_hash = sha256(CONTRACT)
    sources = [
        ROOT/"authoritative_dataset/__init__.py",
        ROOT/"authoritative_dataset/generate_v1.py",
        ROOT/"authoritative_dataset/dynamic_motion_v2.py",
        ROOT/"authoritative_dataset/perception_probe_v2.py",
        ROOT/"authoritative_dataset/cuda_renderer_v1.py",
        ROOT/"authoritative_dataset/exact_backend_v1.py",
        CONTRACT,
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    source_hash = hashlib.sha256(json.dumps(
        source_hashes, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    config.update({
        "dataset_version": "phase8_authoritative_v3",
        "output_root": str(ROOT/"data/phase8_authoritative_v3"),
        "dynamic_motion_contract_path":
            "configs/authoritative_dynamic_motion_contract_v2.yaml",
        "dynamic_motion_contract_hash": contract_hash,
        "authority_source_dataset": "phase8_authoritative_v1",
        "authority_source_root":
            str(ROOT/"data/phase8_authoritative_v1"),
        "authority_source_root_manifest_hash": V1_HASH,
        "parent_dataset_version": "phase8_authoritative_v2",
        "parent_dataset_root_manifest_hash": V2_HASH,
        "parent_git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True
        ).strip(),
    })
    config["frozen_hashes"]["source_hash"] = source_hash
    config["frozen_hashes"]["authority_source_root_manifest_hash"] = V1_HASH
    write_yaml(V3_CONFIG, config)

    smoke = yaml.safe_load(V3_CONFIG.read_text())
    smoke["output_root"] = str(
        ROOT/"artifacts/phase8jqv2_4_dynamic_motion_smoke"
    )
    smoke["static_scenarios"] = []
    smoke["smoke_settings"] = {
        "map_count": 3,
        "frames_per_split": 420,
        "frames_per_sequence": 20,
        "scenario_map_indices": {
            "multi_target": [0],
        },
    }
    write_yaml(SMOKE_CONFIG, smoke)
    report = {
        "status": "PASS",
        "phase": "phase8jqv2_4_authoritative_dynamic_motion_repair",
        "dataset_version": "phase8_authoritative_v3",
        "parent_dataset_version": "phase8_authoritative_v2",
        "v1_root_manifest_hash": V1_HASH,
        "v2_root_manifest_hash": V2_HASH,
        "v1_unchanged": sha256(v1_manifest) == V1_HASH,
        "v2_unchanged": sha256(v2_manifest) == V2_HASH,
        "motion_contract_version": "authoritative_dynamic_motion_contract_v2",
        "motion_contract_hash": contract_hash,
        "source_hash": source_hash,
        "source_hashes": source_hashes,
        "formal_config": str(V3_CONFIG),
        "formal_config_hash": sha256(V3_CONFIG),
        "smoke_config": str(SMOKE_CONFIG),
        "smoke_config_hash": sha256(SMOKE_CONFIG),
        "formal_generation_started": False,
        "training_started": False,
        "production_test_used": False,
        "blind_used": False,
    }
    write_json(
        ROOT/"reports/phase8jqv2_4_dynamic_motion_contract_v2.json",
        report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
