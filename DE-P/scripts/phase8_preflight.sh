#!/usr/bin/env bash
set -euo pipefail
root=/home/zjh/YOPO/DE-P
cd "$root"
echo "PHASE8 preflight command: conda run --no-capture-output -n yopo python tests/run_host_phase8_preflight.py"
echo "Bound: 30 fixed overfit steps + 1 epoch capped at 8 steps; output under runs/dynamic"
echo "Stop: Ctrl-C (managed runner saves latest after its current batch)"
conda run --no-capture-output -n yopo python -m unittest discover -s tests -p 'test_*.py'
conda run --no-capture-output -n yopo python tools/validate_dynamic_dataset.py data/phase8_preflight_multimap
conda run --no-capture-output -n yopo python tests/run_backbone_variant_cpu_validation.py
conda run --no-capture-output -n yopo python tests/run_host_phase8_preflight.py
