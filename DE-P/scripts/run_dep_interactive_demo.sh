#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

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

cd "$ROOT"
exec "${PYTHON_COMMAND[@]}" tools/run_dep_interactive_demo.py "$@"
