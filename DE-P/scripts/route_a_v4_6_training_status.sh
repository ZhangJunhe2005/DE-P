#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python - <<'PY'
import json
from pathlib import Path

root = Path("runs/route_a_static_yopo_v4_6_four_scene_finetune")
runs = sorted(path for path in root.glob("*") if path.is_dir())
if not runs:
    print(json.dumps({"status": "NOT_STARTED"}, indent=2))
    raise SystemExit
run = runs[-1]
state = run / "run_state.json"
completion = run / "training_complete.json"
print(json.dumps({
    "run": str(run.resolve()),
    "state": json.loads(state.read_text()) if state.is_file() else None,
    "completion": json.loads(completion.read_text())
        if completion.is_file() else None,
    "epoch_checkpoints": len(list((run / "checkpoints").glob("epoch_*.pth"))),
    "best_checkpoint": str((run / "checkpoints/best.pth").resolve())
        if (run / "checkpoints/best.pth").is_file() else None,
    "lock_present": (run / "training.lock").is_file(),
}, indent=2, sort_keys=True))
PY
