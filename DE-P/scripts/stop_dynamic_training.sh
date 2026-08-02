#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 2; }
root=/home/zjh/YOPO/DE-P/runs/dynamic
run_dir=$(realpath "$1")
[[ "$run_dir" == "$root"/* ]] || { echo "refusing unmanaged run directory" >&2; exit 2; }
pid_file="$run_dir/pid"
[[ -f "$pid_file" ]] || { echo "run has no active PID"; exit 0; }
pid=$(<"$pid_file")
[[ -r "/proc/$pid/cmdline" ]] || { echo "PID is no longer running"; exit 0; }
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
expected_start=$(<"$run_dir/pid_start_time")
actual_start=$(awk '{print $22}' "/proc/$pid/stat")
[[ "$cmd" == *run_managed_dynamic_training.py* && "$actual_start" == "$expected_start" ]] || {
  echo "PID identity does not match target run; refusing" >&2; exit 2;
}
kill -INT "$pid"
for _ in $(seq 1 30); do
  [[ ! -e "/proc/$pid" ]] && { echo "safe stop complete"; exit 0; }
  sleep 1
done
echo "Process did not exit after 30 seconds. Inspect it, then use: kill -TERM $pid"
exit 1
