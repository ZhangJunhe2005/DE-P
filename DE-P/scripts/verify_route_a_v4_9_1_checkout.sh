#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="$ROOT/runs/route_a_static_yopo_v4_8_3_candidate_only_shakedown/20260813T050742Z-13178/checkpoints/best.pth"
EXPECTED_CHECKPOINT_SHA256="22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
VERIFY_RUNTIME_ASSETS=0

if [[ "${1:-}" == "--runtime-assets" ]]; then
  VERIFY_RUNTIME_ASSETS=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--runtime-assets]" >&2
  exit 2
fi

if [[ -n "${YOPO_PYTHON:-}" ]]; then
  PYTHON_COMMAND=("$YOPO_PYTHON")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "yopo" \
    && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_COMMAND=("$CONDA_PREFIX/bin/python")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_COMMAND=(conda run --no-capture-output -n yopo python)
else
  echo "The yopo Conda environment is unavailable." >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Release checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi

ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)"
if [[ "$ACTUAL_CHECKPOINT_SHA256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "Release checkpoint hash mismatch: $ACTUAL_CHECKPOINT_SHA256" >&2
  exit 1
fi

cd "$ROOT"
"${PYTHON_COMMAND[@]}" -m pytest -q \
  tests/test_deadlock_recovery_v3.py \
  tests/test_recovery_subgoal_v1.py \
  tests/test_route_a_v4_9_dynamic_runtime_entry.py \
  tests/test_route_a_v4_9_1_runtime_entry.py \
  tests/test_runtime_profile_v4_9.py \
  tests/test_runtime_profile_v4_9_1.py

if [[ "$VERIFY_RUNTIME_ASSETS" -eq 1 ]]; then
  "${PYTHON_COMMAND[@]}" -m pytest -q \
    tests/test_dep_interactive_demo.py
  "${PYTHON_COMMAND[@]}" tools/validate_route_a_launch_fixture.py \
    --config configs/dep_interactive_demo_scenes_v4_6.json
  for workspace in Controller Simulator; do
    if [[ ! -d "$ROOT/../$workspace" ]]; then
      echo "Required sibling ROS workspace is missing: $ROOT/../$workspace" >&2
      exit 1
    fi
  done
fi

printf '%s\n' '{"status":"PASS","release":"route_a_v4_9_1_recovery_subgoal_v6","training_started":false,"ros_master_required":false}'
