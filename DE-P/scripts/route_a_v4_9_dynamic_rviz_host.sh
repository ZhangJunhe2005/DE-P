#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
SCENES="$ROOT/configs/dep_interactive_demo_scenes_v4_6.json"
CHECKPOINT="$ROOT/runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown/20260813T050742Z-13178/checkpoints/best.pth"
EXPECTED_SHA256="22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
SCENE="${1:-pillar}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "V4.9 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
esac
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Frozen V4.8.3 checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi
ACTUAL_SHA256="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Frozen V4.8.3 checkpoint hash mismatch" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 1
fi

cd "$ROOT"
conda run --no-capture-output -n yopo \
  python tools/validate_route_a_launch_fixture.py \
    --config "$SCENES" --scene "$SCENE"

# V4.9 changes no learned weight.  Its runtime profile keeps the V4.8.5
# recovery state machine but removes the loop-rate-dependent 30-replan delay:
# one complete 2 s / 0.20 m stationary odometry window starts recovery.  The
# default validation fixture places 16 deterministic, canonical-clear actors
# in seed-stratified XY cells and Z layers over the complete map.  Each actor
# follows an independently sampled 3-D trajectory, so the fixture neither
# collapses onto the nominal route nor confines actors to one altitude.
# Extra arguments are appended intentionally, so callers can still select an
# older layout explicitly when reproducing a historical run.
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --scenes-config "$SCENES" \
  --checkpoint "$CHECKPOINT" \
  --actors multi_target \
  --actor-count 16 \
  --actor-layout uniform_3d \
  --actor-seed 9098 \
  --actor-vertical-span 2.0 \
  --dynamic-mode dynamic_safety \
  --goal-mode fixed-ab \
  --runtime-safety 1 \
  --runtime-profile v4_9_dynamic_motion_preserving_safety \
  --planning-speed 3.0 \
  --dynamic-foreground-mode range_image_hybrid \
  --deadlock-recovery 1 \
  --deadlock-recovery-profile bounded_scan_v3 \
  --arrival-radius 5 \
  "$@"
