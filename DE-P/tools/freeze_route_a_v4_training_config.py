#!/usr/bin/env python3
"""Fill immutable V4 dataset/code identities after the one-time data build."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from policy.static_yopo_contract_v2 import (
    LOSS_CONTRACT_V2_HASH,
    MODEL_CONTRACT_V2_HASH,
    NORMALIZATION_CONTRACT_V2_HASH,
)
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


TEMPLATE = ROOT / "configs/route_a_v4_static_yopo_training.template.yaml"
OUTPUT = ROOT / "configs/route_a_v4_static_yopo_training.yaml"


def main():
    if OUTPUT.exists():
        raise FileExistsError("frozen V4 training config already exists")
    yaml = YAML()
    config = yaml.load(TEMPLATE)
    source_manifest = (
        Path(config["source_dataset_root"]) / "manifests/dataset_manifest.json"
    )
    derived_manifest = (
        Path(config["derived_dataset_root"]) / "manifests/dataset_manifest.json"
    )
    manifest = json.loads(derived_manifest.read_text())
    if manifest["status"] != "COMPLETE_FROZEN":
        raise RuntimeError("V4 derived dataset is not frozen")
    config["training_implementation_hash"] = training_implementation_hash()
    config["source_dataset_manifest_hash"] = sha256_file(source_manifest)
    config["derived_dataset_manifest_hash"] = sha256_file(derived_manifest)
    config["normalization_contract_hash"] = NORMALIZATION_CONTRACT_V2_HASH
    config["model_contract_hash"] = MODEL_CONTRACT_V2_HASH
    config["loss_contract_hash"] = LOSS_CONTRACT_V2_HASH
    with open(OUTPUT, "w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
