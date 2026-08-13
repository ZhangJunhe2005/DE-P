#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_8_recovery_capacity_shakedown"
SCENES="$ROOT/configs/dep_interactive_demo_scenes_v4_6.json"
SCENE="${1:-pillar}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "V4.8.2 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
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
    echo "No completed V4.8 checkpoint found under $RUN_ROOT" >&2
    echo "Use --checkpoint PATH to test an explicit checkpoint." >&2
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

# All supported scenes deliberately use the same runtime and recovery
# contract.  No maze type, scene name or map UUID changes the trigger,
# scan angles, handoff confirmation or candidate safety rules.
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
  --runtime-profile v4_8_2_universal_stagnation_recovery \
  --planning-speed 3.0 \
  --dynamic-foreground-mode range_image_hybrid \
  --deadlock-recovery 1 \
  --deadlock-recovery-profile bounded_scan_v3 \
  --arrival-radius 5 \
  "${FORWARD_ARGS[@]}"
