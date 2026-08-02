#!/usr/bin/env python3
"""Add the omitted V4 split identity without rebuilding immutable samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.build_route_a_v4_static_dataset import (
    SPLIT_CONTRACT_VERSION,
    validate_split_contract,
    write_json,
    write_split_contract,
)

DERIVED = ROOT / "data/route_a_v4_static_yopo"
MANIFEST = DERIVED / "manifests/dataset_manifest.json"
CONFIG = ROOT / "configs/route_a_v4_static_yopo_training.yaml"
MIGRATIONS = DERIVED / "generation_state/migrations"


def atomic_yaml(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    yaml = YAML()
    with open(temporary, "w", encoding="utf-8") as stream:
        yaml.dump(document, stream)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("dataset_version") != "route_a_v4_static_yopo"
        or manifest.get("status") != "COMPLETE_FROZEN"
        or manifest.get("training_started") is not False
    ):
        raise RuntimeError("V4 derived dataset is not eligible for repair")
    existing_runs = list(
        (ROOT / "runs").glob("route_a_v4*/**/run_state.json")
    )
    if existing_runs:
        raise RuntimeError(f"V4 training has already started: {existing_runs}")
    if manifest.get("split_hash"):
        split_hash = validate_split_contract(DERIVED, manifest)
        print(json.dumps({
            "status": "PASS",
            "action": "already_current",
            "split_hash": split_hash,
            "training_started": False,
        }, indent=2))
        return
    counts = manifest["split_counts"]
    excluded = manifest["excluded_canonical_no_return"]
    result = {
        "status": "READY" if not args.apply else "PASS",
        "action": "add_missing_v4_split_contract",
        "sample_counts": counts,
        "training_started": False,
    }
    if not args.apply:
        print(json.dumps(result, indent=2))
        return
    MIGRATIONS.mkdir(parents=True, exist_ok=True)
    old_manifest = MIGRATIONS / "dataset_manifest_before_split_contract.json"
    old_config = MIGRATIONS / "training_config_before_split_contract.yaml"
    if old_manifest.exists() or old_config.exists():
        raise FileExistsError("V4 split repair archive already exists")
    shutil.copy2(MANIFEST, old_manifest)
    if CONFIG.is_file():
        shutil.copy2(CONFIG, old_config)
    split_hash = write_split_contract(DERIVED, counts, excluded)
    manifest["split_contract_version"] = SPLIT_CONTRACT_VERSION
    manifest["split_hash"] = split_hash
    manifest["metadata_migrations"] = [
        *manifest.get("metadata_migrations", []),
        {
            "id": "route_a_v4_split_contract_repair_v1",
            "sample_payload_modified": False,
            "training_started": False,
        },
    ]
    write_json(MANIFEST, manifest)
    build_complete = DERIVED / "generation_state/BUILD_COMPLETE.json"
    complete = json.loads(build_complete.read_text())
    complete["manifest_sha256"] = sha256_file(MANIFEST)
    complete["metadata_migration"] = "route_a_v4_split_contract_repair_v1"
    write_json(build_complete, complete)
    if CONFIG.is_file():
        yaml = YAML(typ="safe")
        config = yaml.load(CONFIG)
        config["derived_dataset_manifest_hash"] = sha256_file(MANIFEST)
        atomic_yaml(CONFIG, config)
    validate_split_contract(DERIVED, manifest)
    result.update({
        "split_hash": split_hash,
        "derived_manifest_hash": sha256_file(MANIFEST),
        "config_hash": sha256_file(CONFIG) if CONFIG.is_file() else None,
        "sample_payload_modified": False,
        "archives": [str(old_manifest), str(old_config)],
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
