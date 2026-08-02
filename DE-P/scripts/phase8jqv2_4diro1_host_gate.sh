#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "yopo" ]]; then
  echo "ERROR: activate the existing yopo environment first." >&2
  exit 2
fi

python tools/run_phase8jqv2_4diro1_review.py
python tools/run_phase8jqv2_4diro1_host_runtime.py --device 0
python -m unittest tests.test_phase8jqv2_4diro1
python -m compileall -q \
  policy/dynamic/dynamic_frame_artifacts_v1.py \
  policy/dynamic/component_aggregation_fast_v1.py \
  policy/dynamic/dynamic_perception_fast_path_v1.py \
  policy/dynamic/runtime_telemetry_fast_v1.py \
  tools/run_phase8jqv2_4diro1_review.py \
  tools/run_phase8jqv2_4diro1_host_runtime.py
git diff --check
