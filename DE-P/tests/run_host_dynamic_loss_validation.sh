#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1
conda run --no-capture-output -n yopo python tests/run_host_dynamic_loss_validation.py
