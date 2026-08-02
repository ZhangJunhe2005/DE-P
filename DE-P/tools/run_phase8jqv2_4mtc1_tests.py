#!/usr/bin/env python3
"""Dependency-free runner for the 81 named MTC1 contract checks."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_phase8jqv2_4mtc1.py"
REPORT = ROOT / "reports/phase8jqv2_4mtc1_test_results.json"


def atomic_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    spec = importlib.util.spec_from_file_location("mtc1_contract", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for name, check in module.CHECKS:
        try:
            passed = bool(check())
            error = None
        except Exception as exception:
            passed = False
            error = f"{type(exception).__name__}: {exception}"
        rows.append({"name": name, "passed": passed, "error": error})
    compile_result = subprocess.run([
        sys.executable, "-m", "compileall", "-q",
        str(ROOT / "tools/prepare_phase8jqv2_4mtc1.py"),
        str(ROOT / "tools/run_phase8jqv2_4mtc1_parameter_audit.py"),
        str(ROOT / "tools/run_phase8jqv2_4mtc1_map_sensitivity.py"),
        str(ROOT / "tools/run_phase8jqv2_4mtc1_near_miss_audit.py"),
        str(ROOT / "tools/audit_phase8jqv2_4mtc1_pillar.py"),
        str(ROOT / "tools/run_phase8jqv2_4mtc1_sensitivity_cuda.py"),
        str(ROOT / "tools/validate_phase8jqv2_4mtc1_independent_rerender.py"),
        str(ROOT / "tools/finalize_phase8jqv2_4mtc1.py"),
        str(TEST),
    ], capture_output=True, text=True)
    status = "PASS" if (
        len(rows) == 81 and all(row["passed"] for row in rows)
        and compile_result.returncode == 0
    ) else "FAIL"
    value = {
        "status": status,
        "runner": "dependency_free_named_contract_runner_v1",
        "pytest_available": False,
        "pytest_absence_is_environment_fact_not_skipped_assertions": True,
        "named_check_count": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "checks": rows,
        "compileall_returncode": compile_result.returncode,
        "compileall_stderr": compile_result.stderr,
    }
    atomic_new(REPORT, value)
    print(json.dumps({
        "status": status, "checks": len(rows),
        "passed": value["passed"], "failed": value["failed"],
    }, indent=2))
    if status != "PASS":
        for row in rows:
            if not row["passed"]:
                print(json.dumps(row))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
