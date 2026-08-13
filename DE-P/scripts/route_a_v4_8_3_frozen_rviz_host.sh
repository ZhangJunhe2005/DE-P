#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
CHECKPOINT="$ROOT/runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown/20260813T050742Z-13178/checkpoints/best.pth"
EXPECTED_SHA256="22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
SCENE="${1:-pillar}"
if [[ $# -gt 0 ]]; then shift; fi

case "$SCENE" in
  cave|forest|pillar|wall) ;;
  *) echo "Frozen V4.8.3 scene must be cave, forest, pillar, or wall" >&2; exit 2 ;;
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

exec bash "$ROOT/scripts/route_a_v4_8_2_rviz_host.sh" "$SCENE" \
  --checkpoint "$CHECKPOINT" "$@"
