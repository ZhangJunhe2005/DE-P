"""Run all DE-P baseline tests without requiring pytest or a ROS master."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.baseline_helpers import seed_everything


def main() -> int:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    seed_everything(0)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2, buffer=False).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
