#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_8_recovery_capacity_shakedown"
# Keep both the V4.7 fixed closed-loop fixture and runtime profile unchanged
# so the V4.8 recovery-capacity comparison has only one learned-variable axis.
SCENES="$ROOT/configs/dep_interactive_demo_scenes_v4_6.json"
SCENE="${1:-pillar}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "V4.8 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
esac

CHECKPOINT=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint requires a path" >&2; exit 2; }
      CHECKPOINT="$2"
      shift 2
      ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
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
    echo "No completed V4.8 best checkpoint found under $RUN_ROOT" >&2
    echo "Use --checkpoint PATH to test an explicit epoch or checkpoint." >&2
    exit 1
  fi
  CHECKPOINT="$LATEST"
elif [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$ROOT/$CHECKPOINT"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi

# The recovery-capacity objective is exercised by the same bounded pillar scan
# used in V4.7; other scenes retain the unchanged legacy recovery setting.
if [[ "$SCENE" == "pillar" ]]; then
  RECOVERY_ARGS=(
    --deadlock-recovery 1
    --deadlock-recovery-profile bounded_scan_v3
  )
else
  RECOVERY_ARGS=(
    --deadlock-recovery 0
    --deadlock-recovery-profile legacy_v2
  )
fi

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_launch_fixture.py \
    --config "$SCENES" --scene "$SCENE"
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --scenes-config "$SCENES" \
  --checkpoint "$CHECKPOINT" \
  --actors none \
  --actor-count 0 \
  --goal-mode fixed-ab \
  --runtime-safety 1 \
  --runtime-profile v4_7_balanced_dynamic \
  --planning-speed 3.0 \
  --dynamic-foreground-mode range_image_hybrid \
  "${RECOVERY_ARGS[@]}" \
  --arrival-radius 5 \
  "${FORWARD_ARGS[@]}"
