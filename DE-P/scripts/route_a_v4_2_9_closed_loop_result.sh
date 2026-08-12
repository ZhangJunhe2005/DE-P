#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
DEMO_RUN="${1:-}"
if [[ -z "$DEMO_RUN" ]]; then
  DEMO_RUN="$(find "$ROOT/runs/dep_interactive_demo" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r | head -1 || true)"
fi
if [[ -z "$DEMO_RUN" || ! -f "$DEMO_RUN/collision_report.json" ]]; then
  echo "Missing completed collision_report.json; close the RViz demo first." >&2
  exit 1
fi
cd "$ROOT"
conda run --no-capture-output -n yopo python \
  tools/evaluate_route_a_v4_2_9_closed_loop.py \
  --collision-report "$DEMO_RUN/collision_report.json" \
  --config configs/route_a_v4_2_9_projected_score_training.yaml \
  --output "$DEMO_RUN/v4_2_9_closed_loop_result.json"
