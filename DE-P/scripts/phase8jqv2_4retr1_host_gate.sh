#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

conda run --no-capture-output -n yopo python - <<'PY'
import torch
assert torch.cuda.is_available(), "host CUDA is required"
assert torch.cuda.get_device_capability(0) == (12, 0)
print("RETR1_HOST_CUDA", torch.cuda.get_device_name(0))
PY
conda run --no-capture-output -n yopo python tools/run_phase8jqv2_4retr1_review.py
conda run --no-capture-output -n yopo python -m unittest tests.test_phase8jqv2_4retr1
conda run --no-capture-output -n yopo python -m unittest tests.test_phase8jqv2_4perto1
conda run --no-capture-output -n yopo python -m unittest tests.test_phase8jqv2_4pecr1
conda run --no-capture-output -n yopo python -m compileall -q policy/dynamic tools tests
git diff --check
