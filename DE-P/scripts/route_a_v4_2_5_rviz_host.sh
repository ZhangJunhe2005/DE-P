#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_2_5_safe_progress_recovery"
SCENE="${1:-forest}"
if [[ $# -gt 0 ]]; then shift; fi

LATEST="$(
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if [[ -f "$candidate/checkpoints/best.pth" ]]; then
          printf '%s\n' "$candidate"
          break
        fi
      done
)"
if [[ -z "$LATEST" || ! -f "$LATEST/checkpoints/best.pth" ]]; then
  echo "No Gate-qualified V4.2.5 best checkpoint found under $RUN_ROOT" >&2
  exit 1
fi

cd "$ROOT"
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --run-dir "$LATEST" \
  --epoch best \
  --runtime-safety 1 \
  --deadlock-recovery 1 \
  "$@"
