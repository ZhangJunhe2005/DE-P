#!/usr/bin/env python3
"""Recoverably replace invalid Phase 8G control recordings."""

from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]
protocol = ROOT / "data/phase8g_perception_protocol"
target = protocol / "development"
patch = protocol / "development_static_wall_patch_v2"
backup = protocol / "historical_invalid_static_wall_v1/development"
if backup.exists():
    raise FileExistsError(backup)
(backup / "sequences").mkdir(parents=True)
(backup / "scenario_configs").mkdir()
sequence_ids = sorted(
    path.name for path in (patch / "sequences").iterdir() if path.is_dir()
)
if len(sequence_ids) != 4:
    raise ValueError("expected exactly four corrected development controls")
for sequence in sequence_ids:
    old_sequence = target / "sequences" / sequence
    new_sequence = patch / "sequences" / sequence
    old_scenario = target / "scenario_configs" / f"{sequence}.yaml"
    new_scenario = patch / "scenario_configs" / f"{sequence}.yaml"
    os.replace(old_sequence, backup / "sequences" / sequence)
    os.replace(old_scenario, backup / "scenario_configs" / f"{sequence}.yaml")
    os.replace(new_sequence, old_sequence)
    os.replace(new_scenario, old_scenario)
(backup / "README.txt").write_text(
    "These four recordings were retained because the v1 static-wall control "
    "was visible instead of fully occluded.\n"
)
print({"status": "PASS", "replaced": sequence_ids, "backup": str(backup)})
