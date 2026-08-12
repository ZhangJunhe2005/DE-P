#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_3_simple"
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
if [[ -z "$LATEST" ]]; then
  echo '{"status":"NOT_STARTED"}'
  exit 0
fi
echo "run_dir: $LATEST"
[[ -f "$LATEST/run_state.json" ]] && cat "$LATEST/run_state.json"
if [[ -f "$LATEST/metrics.jsonl" ]]; then
  echo
  echo "latest_epoch_metrics:"
  tail -n 1 "$LATEST/metrics.jsonl"
fi
[[ -f "$LATEST/checkpoints/best.pth" ]] && \
  echo "best_checkpoint: $LATEST/checkpoints/best.pth"
