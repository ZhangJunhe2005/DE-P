#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zjh/YOPO/DE-P"
RUN="${1:-}"

if [[ -z "$RUN" ]]; then
  RUN="$({
    find "$ROOT/runs/dep_interactive_demo" -mindepth 1 -maxdepth 1 \
      -type d -name '*-pillar' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n 1 | cut -d' ' -f2-
  } || true)"
fi
if [[ -z "$RUN" || ! -d "$RUN" ]]; then
  echo "No completed Pillar interactive run found: $RUN" >&2
  exit 1
fi
if [[ ! -f "$RUN/collision_report.json" ]]; then
  echo "Close the RViz run first; collision_report.json is not complete: $RUN" >&2
  exit 1
fi

cd "$ROOT"
exec conda run --no-capture-output -n yopo \
  python tools/summarize_deadlock_recovery_run.py "$RUN"
