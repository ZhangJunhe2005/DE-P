#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "yopo" ]]; then
  echo "ERROR: activate the existing yopo environment first." >&2
  exit 2
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("host CUDA is required")
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

python tools/run_phase8jqv2_4mar1_host_runtime.py --device 0
python -m unittest tests.test_phase8jqv2_4mar1
python -m compileall -q \
  policy/dynamic/safety_measurement_availability_v1.py \
  policy/dynamic/rejected_component_safety_adapter_v1.py \
  policy/dynamic/fragmented_component_support_v1.py \
  policy/dynamic/measurement_availability_provisional_feed_v1.py \
  policy/dynamic/cold_start_foreground_availability_v1.py \
  tools/run_phase8jqv2_4mar1_review.py \
  tools/run_phase8jqv2_4mar1_host_runtime.py
git diff --check
