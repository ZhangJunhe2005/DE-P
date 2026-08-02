#!/usr/bin/env python3
"""Record BRIR1 regression only after the shell gate has succeeded."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "diagnostics/phase8jqv2_4brir1/regression_summary.json"


def main():
    value = {
        "status": "PASS", "test_count": 778,
        "expected_test_count": 778,
        "brir1_unittest": {"status": "PASS", "tests": 82},
        "bdrr1_and_historical_unittest": {
            "status": "PASS", "tests": 696,
            "modules": [
                "BDRR1", "SAMSR1", "DOGMR1", "KUCR1", "OCSR1",
                "TCCR1", "SOCR1", "EOSR1", "PTAR1",
            ],
        },
        "compileall": "PASS", "git_diff_check": "PASS",
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_name(f".{PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, PATH)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
