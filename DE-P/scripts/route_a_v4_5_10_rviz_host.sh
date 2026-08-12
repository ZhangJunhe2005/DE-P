#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_5_10_tail_aware_safety"
SCENE="${1:-forest}"
if [[ $# -gt 0 ]]; then shift; fi
LATEST="$({
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if [[ -f "$candidate/training_complete.json" \
              && -f "$candidate/checkpoints/best.pth" ]] \
              && grep -q 'DIAGNOSTIC_COMPLETE_NOT_PRODUCTION' \
                "$candidate/training_complete.json"; then
          printf '%s\n' "$candidate"
          break
        fi
      done
} || true)"
if [[ -z "$LATEST" || ! -f "$LATEST/checkpoints/best.pth" ]]; then
  echo "No completed V4.5.10 best checkpoint found under $RUN_ROOT" >&2
  exit 1
fi

cd "$ROOT"
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --checkpoint "$LATEST/checkpoints/best.pth" \
  --runtime-safety 1 \
  --runtime-profile v4_5_10_tail_aware_retimed \
  --planning-speed 3.0 \
  --deadlock-recovery 0 \
  "$@"
