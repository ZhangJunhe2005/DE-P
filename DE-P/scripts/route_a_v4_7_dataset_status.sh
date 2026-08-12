#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo \
  python tools/prepare_route_a_v4_7_maps.py status
conda run --no-capture-output -n yopo python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path("data/route_a_v4_7_raw_static")
counts = Counter()
types = Counter()
for path in (root / "manifests/sequences").glob("*.json"):
    row = json.loads(path.read_text())
    counts[row["split"]] += int(row["frame_count"])
    sequence_root = root / row["suite"] / row["split"] / row["sequence_id"]
    diagnostics = sequence_root / "render_diagnostics.json"
    if diagnostics.is_file():
        value = json.loads(diagnostics.read_text())
        kind = value.get("scene_spatial_observability", {}).get("map_type")
        if kind:
            types[(row["split"], kind)] += 1
markers = root / "generation_state/completion"
print(json.dumps({
    "dataset_version": "route_a_v4_7_raw_static_v1",
    "raw_frames": {
        "train": f"{counts['train']}/300000",
        "valid": f"{counts['valid']}/60000",
    },
    "sequence_types": {
        f"{split}/{kind}": count
        for (split, kind), count in sorted(types.items())
    },
    "train_complete": (markers / "TRAIN_SPLIT_GENERATION_COMPLETE").is_file(),
    "valid_complete": (markers / "VALID_SPLIT_GENERATION_COMPLETE").is_file(),
    "full_complete": (markers / "FULL_GENERATION_COMPLETE").is_file(),
    "derived_complete": Path(
        "data/route_a_v4_7_static_yopo/generation_state/BUILD_COMPLETE.json"
    ).is_file(),
    "validation_report": Path(
        "reports/route_a_v4_7_dataset_validation.json"
    ).is_file(),
    "training_started": False,
}, indent=2, sort_keys=True))
PY
