#!/usr/bin/env python3
"""One-shot Phase 8G blind protocol.  This file must be run by the user."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from evaluate_phase8g_instance_perception import evaluate
except ModuleNotFoundError:  # Imported as tools.phase8g_blind_protocol in tests.
    from tools.evaluate_phase8g_instance_perception import evaluate


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "Simulator"
REPORTS = ROOT / "reports"
LOCK = REPORTS / ".phase8g_blind_once.lock"
HISTORICAL_CONSUMED_TEST = ROOT / "data/phase8_dynamic_production/test"
MAX_BLIND_SEED = 336_868_795


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_clean(path):
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"blind protocol requires clean Git worktree: {path}")


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def one_shot_targets(blind_root, static_root):
    return (
        LOCK,
        Path(blind_root),
        Path(static_root),
        REPORTS / "phase8g_blind_once_quality.json",
        REPORTS / "phase8g_final_perception_gate.json",
        REPORTS / "phase8g_final_readiness.md",
    )


def assert_one_shot_targets_absent(blind_root, static_root):
    for path in one_shot_targets(blind_root, static_root):
        if path.exists():
            raise FileExistsError(f"one-shot blind target already exists: {path}")


def assert_not_historical_dataset(path):
    if Path(path).resolve() == HISTORICAL_CONSUMED_TEST.resolve():
        raise RuntimeError("historical Phase 8F test is audit-only and cannot be rerun")


def frozen_hashes(source_files, required_reports, blind_seed):
    return {
        "blind_seed_sha256": hashlib.sha256(str(blind_seed).encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            b"".join(path.read_bytes() for path in source_files)
        ).hexdigest(),
        "config_sha256": sha256(ROOT / "config/traj_opt.yaml"),
        "evaluator_sha256": sha256(
            ROOT / "tools/evaluate_phase8g_instance_perception.py"
        ),
        "development_report_sha256": sha256(required_reports[0]),
        "shadow_validation_report_sha256": sha256(required_reports[1]),
    }


def assert_frozen(expected, source_files, required_reports, blind_seed):
    actual = frozen_hashes(source_files, required_reports, blind_seed)
    if actual != {key: expected[key] for key in actual}:
        raise RuntimeError("blind protocol frozen implementation or report hash changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-seed", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.blind_seed <= MAX_BLIND_SEED:
        raise ValueError(
            f"blind seed must be in 1..{MAX_BLIND_SEED} so all scenario "
            "seeds and actor IDs fit uint32"
        )
    require_clean(ROOT)
    require_clean(SIM)
    required = [
        REPORTS / "phase8g_development_quality.json",
        REPORTS / "phase8g_shadow_validation_quality.json",
        REPORTS / "phase8g_protocol_readiness.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("development, shadow and readiness reports must exist")
    if any(json.loads(path.read_text()).get("status") != "PASS" for path in required):
        raise RuntimeError("development/shadow/readiness must PASS before blind")
    blind_root = ROOT / "data/phase8g_perception_protocol/blind"
    static_root = ROOT / "data/phase8g_perception_protocol/blind_static_maps"
    assert_not_historical_dataset(blind_root)
    assert_one_shot_targets_absent(blind_root, static_root)
    source_files = [
        ROOT / "policy/dynamic/dynamic_perception.py",
        ROOT / "policy/dynamic/range_image_foreground.py",
        ROOT / "policy/dynamic/track_manager.py",
        ROOT / "policy/dynamic/types.py",
        ROOT / "policy/dynamic/instance_evaluation.py",
        ROOT / "tools/evaluate_phase8g_instance_perception.py",
    ]
    freeze = {
        **frozen_hashes(source_files, required, args.blind_seed),
        "blind_run_count": 1,
        "status": "STARTED_NO_RETRY",
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(freeze, stream, indent=2)
        stream.write("\n")

    map_seed = args.blind_seed
    pose_seed = args.blind_seed + 100_000_003
    actor_seed = args.blind_seed + 200_000_003
    generator = SIM / "devel/lib/sensor_simulator/dataset_generator"
    run([
        sys.executable, ROOT / "tools/run_safe_map_generation.py",
        "--generator", generator, "--output", static_root / "blind",
        "--generator-arg=--map-seed", f"--generator-arg={map_seed}",
        "--generator-arg=--pose-seed", f"--generator-arg={pose_seed}",
        "--generator-arg=--actor-seed", f"--generator-arg={actor_seed}",
        "--generator-arg=--env-num", "--generator-arg=3",
        "--generator-arg=--image-num", "--generator-arg=64",
    ])
    run([
        sys.executable, ROOT / "tools/finalize_phase8g_static_maps.py",
        static_root, "--split", "blind", "3", str(map_seed),
        str(pose_seed), str(actor_seed),
    ])
    run([
        sys.executable, ROOT / "tools/run_phase8g_instance_recording.py",
        "--output", blind_root,
        "--static-catalog", static_root / "map_catalog.yaml",
        "--split", "blind", "--workers", "8", "--ros-master-port-base", "13300",
    ])
    assert_frozen(freeze, source_files, required, args.blind_seed)
    quality = evaluate(blind_root, "blind")
    quality["blind_run_count"] = 1
    quality["freeze"] = freeze
    quality_path = REPORTS / "phase8g_blind_once_quality.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n")
    gate = {
        "status": quality["status"],
        "perception_ready": quality["status"] == "PASS",
        "blind_run_count": 1,
        "blind_manifest_sha256": hashlib.sha256(
            json.dumps(quality, sort_keys=True).encode()
        ).hexdigest(),
        "quality_report": str(quality_path),
    }
    (REPORTS / "phase8g_final_perception_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    (REPORTS / "phase8g_final_readiness.md").write_text(
        "# Phase 8G final readiness\n\n"
        f"Status: **{gate['status']}**\n\n"
        f"`perception_ready`: `{str(gate['perception_ready']).lower()}`  \n"
        "`blind_run_count`: `1`\n"
    )
    print(json.dumps(gate, indent=2))
    if quality["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
