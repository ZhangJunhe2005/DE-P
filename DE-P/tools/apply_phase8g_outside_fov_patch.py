#!/usr/bin/env python3
"""Recoverably replace the development outside-FOV controls."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1] / "data/phase8g_perception_protocol"
target = ROOT / "development"
patch = ROOT / "development_outside_fov_patch_v2"
backup = ROOT / "historical_invalid_outside_fov_v1/development"
if backup.exists():
    raise FileExistsError(backup)
(backup / "sequences").mkdir(parents=True)
(backup / "scenario_configs").mkdir()
ids = sorted(path.name for path in (patch / "sequences").iterdir() if path.is_dir())
if len(ids) != 4:
    raise ValueError("expected four outside-FOV controls")
for sequence in ids:
    for folder, suffix in (("sequences", ""), ("scenario_configs", ".yaml")):
        old = target / folder / f"{sequence}{suffix}"
        new = patch / folder / f"{sequence}{suffix}"
        saved = backup / folder / f"{sequence}{suffix}"
        os.replace(old, saved)
        os.replace(new, old)
(backup / "README.txt").write_text(
    "Retained because the v1 14 m lateral actor entered FOV on a curved path.\n"
)
print({"status": "PASS", "replaced": ids, "backup": str(backup)})
