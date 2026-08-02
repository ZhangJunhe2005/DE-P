#!/usr/bin/env python3
"""Read-only watchdog for one managed run; never signals unrelated processes."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--max-status-age", type=float, default=180.0)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    status_path, pid_path = run / "status.json", run / "pid"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text())
    failures = []
    for key in ("train_loss", "valid_loss", "dynamic_loss", "static_loss", "score_loss"):
        value = status.get(key)
        if value is not None and not isinstance(value, (int, float)):
            failures.append(f"{key}_invalid")
    age = time.time() - status_path.stat().st_mtime
    if status.get("state") == "running" and age > args.max_status_age:
        failures.append("status_stale_or_dataloader_stalled")
    free_gib = shutil.disk_usage(run).free / 2**30
    if free_gib < args.min_free_gib:
        failures.append("disk_low")
    pid = int(pid_path.read_text()) if pid_path.is_file() else None
    alive = bool(pid and Path(f"/proc/{pid}").exists())
    if status.get("state") == "running" and not alive:
        failures.append("training_process_missing")
    print(json.dumps({"status": "FAIL" if failures else "PASS", "run": str(run),
                      "pid": pid, "alive": alive, "status_age_seconds": age,
                      "free_disk_gib": free_gib, "failures": failures,
                      "training": status}, indent=2))
    raise SystemExit(2 if failures else 0)


if __name__ == "__main__":
    main()
