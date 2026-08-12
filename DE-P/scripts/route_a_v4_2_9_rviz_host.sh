#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN_ROOT="$ROOT/runs/route_a_static_yopo_v4_2_9_projected_score"
SCENE="${1:-forest}"
if [[ $# -gt 0 ]]; then shift; fi
ALLOW_UNQUALIFIED=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-unqualified) ALLOW_UNQUALIFIED=1; shift ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

LATEST="$(
  find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -r \
    | while IFS= read -r candidate; do
        if [[ -f "$candidate/checkpoints/best.pth" ]] \
            || [[ "$ALLOW_UNQUALIFIED" == "1" \
                  && -f "$candidate/checkpoints/best_unqualified.pth" ]]; then
          printf '%s\n' "$candidate"
          break
        fi
      done
)"
CHECKPOINT_NAME="best.pth"
if [[ "$ALLOW_UNQUALIFIED" == "1" && -n "$LATEST" \
      && ! -f "$LATEST/checkpoints/$CHECKPOINT_NAME" ]]; then
  CHECKPOINT_NAME="best_unqualified.pth"
fi
if [[ -z "$LATEST" || ! -f "$LATEST/checkpoints/$CHECKPOINT_NAME" ]]; then
  echo "No offline-qualified V4.2.9 checkpoint found under $RUN_ROOT" >&2
  echo "For diagnostic testing only, add --allow-unqualified." >&2
  exit 1
fi
if [[ "$CHECKPOINT_NAME" == "best_unqualified.pth" ]]; then
  echo "WARNING: loading offline-unqualified V4.2.9 checkpoint." >&2
else
  echo "Loading offline-qualified checkpoint; production qualification still requires the closed-loop result." >&2
fi

cd "$ROOT"
exec bash scripts/run_dep_interactive_demo.sh \
  --scene "$SCENE" \
  --checkpoint "$LATEST/checkpoints/$CHECKPOINT_NAME" \
  --runtime-safety 1 \
  --deadlock-recovery 1 \
  "${FORWARD_ARGS[@]}"
