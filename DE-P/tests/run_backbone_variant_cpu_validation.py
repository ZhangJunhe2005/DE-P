#!/usr/bin/env python3
"""Run only the phase-3 dual-backbone CPU validation suite."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    names = [
        "tests.test_backbone_variants",
        "tests.test_variant_checkpoints",
    ]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in names
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("BACKBONE_VARIANT_CPU_VALIDATION_RESULT")
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "gpu": "NOT MEASURED IN CODEX SANDBOX",
    }, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
