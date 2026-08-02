#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

conda run --no-capture-output -n yopo python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'

if [[ -f reports/phase8jqv2_4brir1_fresh_validation_freeze.json \
      && -f diagnostics/phase8jqv2_4brir1/fresh_summary.json ]]; then
  echo "BRIR1 fresh validation already observed; preserving the frozen run."
else
  conda run --no-capture-output -n yopo \
    python tools/run_phase8jqv2_4brir1_fresh_validation.py \
    --stage development

  conda run --no-capture-output -n yopo \
    python tools/run_phase8jqv2_4brir1_fresh_validation.py \
    --stage fresh
fi

conda run --no-capture-output -n yopo \
  python tools/build_phase8jqv2_4brir1_reports.py

conda run --no-capture-output -n yopo python -m unittest \
  tests.test_phase8jqv2_4kucr1 \
  tests.test_phase8jqv2_4ocsr1 \
  tests.test_phase8jqv2_4tccr1 \
  tests.test_phase8jqv2_4socr1 \
  tests.test_phase8jqv2_4eosr1 \
  tests.test_phase8jqv2_4ptar1 \
  tests.test_phase8jqv2_4dogmr1 \
  tests.test_phase8jqv2_4samsr1 \
  tests.test_phase8jqv2_4bdrr1 \
  tests.test_phase8jqv2_4brir1

conda run --no-capture-output -n yopo \
  python tools/finalize_phase8jqv2_4brir1_regression.py

conda run --no-capture-output -n yopo \
  python tools/build_phase8jqv2_4brir1_reports.py

conda run --no-capture-output -n yopo \
  python -m unittest tests.test_phase8jqv2_4brir1

conda run --no-capture-output -n yopo python -m compileall -q \
  controller policy/dynamic tools tests/test_phase8jqv2_4brir1.py

git diff --check

echo "BRIR1 host gate complete."
