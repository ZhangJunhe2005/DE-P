#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Run this script on the host, not in the workspace sandbox."
    )
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

python tools/run_phase8jqv2_entry_gate.py
python tools/validate_phase8jqv2_semantics.py
python tools/evaluate_phase8jqv2_static.py --batch-size 32 --workers 4
python tools/evaluate_candidate_decomposition_v2.py --batch-size 32
# Immediate identical rerun is the formal resume/cache determinism check.
python tools/evaluate_candidate_decomposition_v2.py --batch-size 32
python tools/finalize_phase8jqv2_reports.py
python -m unittest \
  tests.test_phase8jq_semantics \
  tests.test_phase8jqv2_safety_evaluator \
  -v
