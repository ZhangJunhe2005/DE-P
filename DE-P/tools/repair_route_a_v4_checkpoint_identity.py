#!/usr/bin/env python3
"""Close the V4 checkpoint-v1 identity alias failure before retrying training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash

CONFIG = ROOT / "configs/route_a_v4_static_yopo_training.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4"
ARCHIVE = ROOT / "reports/route_a_v4_checkpoint_identity_repair"
EXPECTED_ERROR = (
    "checkpoint identity fields missing: ['source_v3_manifest_hash']"
)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_yaml(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    yaml = YAML()
    with open(temporary, "w", encoding="utf-8") as stream:
        yaml.dump(value, stream)
    os.replace(temporary, path)


def failed_runs():
    result = []
    for run in sorted(RUN_ROOT.glob("*")):
        failure = run / "failure.json"
        if not failure.is_file():
            continue
        document = json.loads(failure.read_text())
        if EXPECTED_ERROR in document.get("error", ""):
            result.append(run)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    yaml = YAML(typ="safe")
    config = yaml.load(CONFIG)
    current_hash = training_implementation_hash()
    runs = failed_runs()
    if not runs:
        raise RuntimeError("no exact V4 checkpoint-identity failure found")
    for run in runs:
        if (run / "training.lock").exists():
            raise RuntimeError(f"training lock still exists: {run}")
        if list((run / "checkpoints").glob("*.pth")):
            raise RuntimeError(f"refusing run with a checkpoint: {run}")
    already_current = (
        config.get("training_implementation_hash") == current_hash
        and all(
            json.loads((run / "run_state.json").read_text()).get("status")
            == "FAILED"
            for run in runs
        )
    )
    if already_current:
        print(json.dumps({
            "status": "PASS",
            "action": "already_current",
            "training_implementation_hash": current_hash,
            "failed_runs": [str(run) for run in runs],
        }, indent=2))
        return
    result = {
        "status": "READY" if not args.apply else "PASS",
        "action": "route_a_v4_checkpoint_identity_alias_repair_v1",
        "old_training_implementation_hash":
            config.get("training_implementation_hash"),
        "new_training_implementation_hash": current_hash,
        "failed_runs": [str(run) for run in runs],
        "checkpoint_recovery_possible": False,
        "restart_epoch": 0,
    }
    if not args.apply:
        print(json.dumps(result, indent=2))
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    config_archive = ARCHIVE / "training_config_before_repair.yaml"
    if not config_archive.exists():
        shutil.copy2(CONFIG, config_archive)
    config["training_implementation_hash"] = current_hash
    hotfixes = list(config.get("implementation_hotfixes", []))
    hotfix_id = "route_a_v4_checkpoint_identity_alias_repair_v1"
    if hotfix_id not in hotfixes:
        hotfixes.append(hotfix_id)
    config["implementation_hotfixes"] = hotfixes
    atomic_yaml(CONFIG, config)
    for run in runs:
        state = run / "run_state.json"
        state_archive = run / "run_state_before_identity_repair.json"
        if not state_archive.exists():
            shutil.copy2(state, state_archive)
        old = json.loads(state.read_text())
        atomic_json(state, {
            **old,
            "status": "FAILED",
            "stop_reason": "CHECKPOINT_IDENTITY_SCHEMA_MISMATCH",
            "checkpoint_saved": False,
            "restart_required": True,
        })
    result.update({
        "config_hash": sha256_file(CONFIG),
        "config_archive": str(config_archive),
        "failed_run_states_repaired": len(runs),
    })
    atomic_json(ARCHIVE / "repair_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
