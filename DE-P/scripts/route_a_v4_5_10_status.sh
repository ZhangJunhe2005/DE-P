#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_5_10_tail_aware_safety"
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -n 1 || true)"
if [[ -z "$LATEST" ]]; then
  echo '{"status":"NOT_STARTED"}'
  exit 0
fi
echo "run_dir: $LATEST"
if [[ -f "$LATEST/run_state.json" ]]; then
  cat "$LATEST/run_state.json"
fi
if [[ -f "$LATEST/metrics.jsonl" ]]; then
  echo "latest_epoch_metrics:"
  tail -n 1 "$LATEST/metrics.jsonl"
fi
