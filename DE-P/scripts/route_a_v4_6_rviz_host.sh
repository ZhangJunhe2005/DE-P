#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_6_four_scene_finetune"
SCENES="$ROOT/configs/dep_interactive_demo_scenes_v4_6.json"
SCENE="${1:-forest}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "V4.6 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
esac

LATEST="$({
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if [[ -f "$candidate/training_complete.json" \
              && -f "$candidate/checkpoints/best.pth" ]] \
              && grep -q 'TRAINING_COMPLETE_PENDING_CLOSED_LOOP' \
                "$candidate/training_complete.json"; then
          printf '%s\n' "$candidate"
          break
        fi
      done
} || true)"
if [[ -z "$LATEST" ]]; then
  echo "No completed V4.6 best checkpoint found under $RUN_ROOT" >&2
  exit 1
fi

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_launch_fixture.py \
    --config "$SCENES" --scene "$SCENE"
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --scenes-config "$SCENES" \
  --checkpoint "$LATEST/checkpoints/best.pth" \
  --actors none \
  --actor-count 0 \
  --goal-mode fixed-ab \
  --runtime-safety 1 \
  --runtime-profile v4_5_10_tail_aware_retimed \
  --planning-speed 3.0 \
  --deadlock-recovery 0 \
  --arrival-radius 5 \
  "$@"
