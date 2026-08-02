#!/usr/bin/env python3
"""Bind the frozen V4.1 optimization strategy to the V4.2 dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash

PARENT = ROOT / "configs/route_a_v4_1_local_goal_training.yaml"
OUTPUT = ROOT / "configs/route_a_v4_2_scene_relative_training.yaml"
SOURCE = ROOT / "data/route_a_v4_2_raw_static"
DERIVED = ROOT / "data/route_a_v4_2_static_yopo"


def main():
    yaml = YAML()
    config = yaml.load(PARENT)
    config["contract_version"] = "route_a_static_yopo_training_v4_2"
    config["parent_v4_1_config_hash"] = sha256_file(PARENT)
    config["training_implementation_hash"] = training_implementation_hash()
    config["source_dataset_root"] = str(SOURCE)
    config["derived_dataset_root"] = str(DERIVED)
    config["source_dataset_manifest_hash"] = sha256_file(
        SOURCE / "manifests/dataset_manifest.json"
    )
    config["derived_dataset_manifest_hash"] = sha256_file(
        DERIVED / "manifests/dataset_manifest.json"
    )
    config["output_root"] = str(
        ROOT / "runs/route_a_static_yopo_v4_2_scene_relative"
    )
    config["implementation_hotfixes"] = [
        *config.get("implementation_hotfixes", []),
        "route_a_v4_2_scene_relative_height_and_observability_contract_v1",
    ]
    serialized = Path(OUTPUT)
    if serialized.exists():
        existing = yaml.load(serialized)
        # Identity hashes are permitted to refresh only before training starts.
        old = dict(existing)
        new = dict(config)
        for key in ("source_dataset_manifest_hash", "derived_dataset_manifest_hash"):
            old.pop(key, None)
            new.pop(key, None)
        if old != new:
            raise FileExistsError(f"refusing to overwrite divergent config: {OUTPUT}")
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "parent_v4_1_config_sha256": sha256_file(PARENT),
        "initial_checkpoint": config["model"]["initial_checkpoint"],
        "training_strategy": "V4.1_UNCHANGED",
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
