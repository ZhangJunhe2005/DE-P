#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_5_9_time_mean_safety"
LATEST="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -n 1 || true)"
if [[ -z "$LATEST" ]]; then
  echo '{"status":"NOT_STARTED"}'
  exit 0
fi
if [[ -f "$LATEST/run_state.json" ]]; then
  cat "$LATEST/run_state.json"
else
  printf '{"status":"STARTING","run":"%s"}\n' "$LATEST"
fi
