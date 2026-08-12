#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_2_6_feasibility_score"
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1 || true)"
if [[ -z "$LATEST" ]]; then
  echo "No V4.2.6 run found under $RUN_ROOT" >&2
  exit 1
fi
echo "run_dir: $LATEST"
[[ -f "$LATEST/run_state.json" ]] && { echo "run_state:"; cat "$LATEST/run_state.json"; }
[[ -f "$LATEST/metrics.jsonl" ]] && { echo "latest_epoch_metrics:"; tail -1 "$LATEST/metrics.jsonl"; }
echo "epoch_checkpoints: $(find "$LATEST/checkpoints" -maxdepth 1 -name 'epoch_*.pth' 2>/dev/null | wc -l)"
if [[ -f "$LATEST/checkpoints/best.pth" ]]; then
  echo "best_checkpoint: $LATEST/checkpoints/best.pth"
elif [[ -f "$LATEST/checkpoints/best_unqualified.pth" ]]; then
  echo "best_unqualified_checkpoint: $LATEST/checkpoints/best_unqualified.pth"
fi
