#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN="${1:-}"
if [[ -z "$RUN" ]]; then
  RUN="$({
    find "$ROOT/runs/dep_interactive_demo" -mindepth 1 -maxdepth 1 -type d \
      | sort -r \
      | while IFS= read -r candidate; do
          manifest="$candidate/manifest.json"
          if [[ -f "$manifest" ]] \
              && grep -q 'v4_8_5_motion_verified_recovery_handoff' "$manifest"; then
            printf '%s\n' "$candidate"
            break
          fi
        done
  } || true)"
elif [[ "$RUN" != /* ]]; then
  RUN="$ROOT/$RUN"
fi

if [[ -z "$RUN" || ! -d "$RUN" ]]; then
  echo "No completed V4.8.5 recovery run found" >&2
  exit 1
fi

cd "$ROOT"
exec conda run --no-capture-output -n yopo \
  python tools/summarize_deadlock_recovery_run.py "$RUN"
