#!/usr/bin/env bash
set -euo pipefail

cd /home/zjh/YOPO/DE-P
conda run --no-capture-output -n yopo python - <<'PY'
import json
from pathlib import Path

root = Path("runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown")
runs = sorted(path for path in root.glob("*") if path.is_dir())
if not runs:
    print(json.dumps({"status": "NOT_STARTED"}, indent=2))
    raise SystemExit
run = runs[-1]
def read(name):
    path = run / name
    return json.loads(path.read_text()) if path.is_file() else None
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()] \
    if (run / "metrics.jsonl").is_file() else []
print(json.dumps({
    "run": str(run.resolve()),
    "state": read("run_state.json"),
    "completion": read("training_complete.json"),
    "epochs_recorded": [row["epoch"] for row in rows],
    "initial_baseline_recorded": any(row["epoch"] == -1 for row in rows),
    "epoch_checkpoints": len(list((run / "checkpoints").glob("epoch_*.pth"))),
    "best_checkpoint": str((run / "checkpoints/best.pth").resolve())
        if (run / "checkpoints/best.pth").is_file() else None,
    "lock_present": (run / "training.lock").is_file(),
}, indent=2, sort_keys=True))
PY
