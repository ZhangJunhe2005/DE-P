#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_2_9_projected_score"
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1 || true)"
if [[ -z "$LATEST" ]]; then
  echo "No V4.2.9 training run found under $RUN_ROOT" >&2
  exit 1
fi
echo "run_dir: $LATEST"
if [[ -f "$LATEST/run_state.json" ]]; then
  echo "run_state:"
  cat "$LATEST/run_state.json"
fi
if [[ -f "$LATEST/metrics.jsonl" ]]; then
  echo "latest_epoch_metrics:"
  tail -1 "$LATEST/metrics.jsonl"
fi
if [[ -f "$LATEST/training_complete.json" ]]; then
  echo "training_complete:"
  cat "$LATEST/training_complete.json"
fi
