#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_8_4_score_adaptation"
SCENE="${1:-pillar}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "V4.8.4 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
esac

LATEST="$({
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if [[ -f "$candidate/training_complete.json" \
              && -f "$candidate/checkpoints/best.pth" ]] \
              && grep -q 'DIAGNOSTIC_COMPLETE_NOT_PRODUCTION' \
                "$candidate/training_complete.json"; then
          printf '%s\n' "$candidate/checkpoints/best.pth"
          break
        fi
      done
} || true)"
if [[ -z "$LATEST" ]]; then
  echo "No completed V4.8.4 checkpoint found under $RUN_ROOT" >&2
  exit 1
fi

exec bash "$ROOT/scripts/route_a_v4_8_2_rviz_host.sh" "$SCENE" \
  --checkpoint "$LATEST" "$@"
