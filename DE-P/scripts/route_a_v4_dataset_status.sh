#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_maps.py status
conda run --no-capture-output -n yopo python -c '
import json
from pathlib import Path
root = Path("data/route_a_v4_raw_static")
manifests = root / "manifests/sequences"
counts = {"train": 0, "valid": 0}
if manifests.is_dir():
    for path in manifests.glob("*.json"):
        row = json.loads(path.read_text())
        counts[row["split"]] += int(row["frame_count"])
markers = root / "generation_state/completion"
train_frames = counts["train"]
valid_frames = counts["valid"]
print(json.dumps({
    "raw_frames": {
        "train": f"{train_frames}/300000",
        "valid": f"{valid_frames}/60000",
    },
    "train_complete": (markers/"TRAIN_SPLIT_GENERATION_COMPLETE").is_file(),
    "valid_complete": (markers/"VALID_SPLIT_GENERATION_COMPLETE").is_file(),
    "full_complete": (markers/"FULL_GENERATION_COMPLETE").is_file(),
    "derived_complete": Path(
        "data/route_a_v4_static_yopo/generation_state/BUILD_COMPLETE.json"
    ).is_file(),
    "training_config_frozen": Path(
        "configs/route_a_v4_static_yopo_training.yaml"
    ).is_file(),
}, indent=2, sort_keys=True))
'
