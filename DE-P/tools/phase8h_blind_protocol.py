#!/usr/bin/env python3
"""One-shot infrastructure-recovery blind protocol after consumed Phase 8G."""

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
except ModuleNotFoundError:
    from tools.evaluate_phase8g_instance_perception import evaluate


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "Simulator"
REPORTS = ROOT / "reports"
LOCK = REPORTS / ".phase8h_blind_once.lock"
PHASE8G_LOCK = REPORTS / ".phase8g_blind_once.lock"
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


def frozen_hashes(source_files, required_reports, seed):
    return {
        "blind_seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            b"".join(path.read_bytes() for path in source_files)
        ).hexdigest(),
        "config_sha256": sha256(ROOT / "config/traj_opt.yaml"),
        "evaluator_sha256": sha256(
            ROOT / "tools/evaluate_phase8g_instance_perception.py"
        ),
        "development_report_sha256": sha256(required_reports[0]),
        "shadow_validation_report_sha256": sha256(required_reports[1]),
        "phase8g_failure_report_sha256": sha256(required_reports[2]),
        "infrastructure_validation_report_sha256": sha256(required_reports[3]),
        "simulator_binary_sha256": sha256(
            SIM / "devel/lib/sensor_simulator/sensor_simulator_cuda"
        ),
        "map_generator_binary_sha256": sha256(
            SIM / "devel/lib/sensor_simulator/dataset_generator"
        ),
    }


def assert_frozen(expected, source_files, required_reports, seed):
    actual = frozen_hashes(source_files, required_reports, seed)
    if actual != {key: expected[key] for key in actual}:
        raise RuntimeError("Phase 8H frozen implementation or report hash changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.blind_seed <= MAX_BLIND_SEED:
        raise ValueError(
            f"blind seed must be in 1..{MAX_BLIND_SEED} so all scenario "
            "seeds and actor IDs fit uint32"
        )
    if not 1 <= args.workers <= 4:
        raise ValueError("Phase 8H recovery uses 1..4 Simulator workers")

    require_clean(ROOT)
    require_clean(SIM)
    if not PHASE8G_LOCK.is_file():
        raise RuntimeError("missing consumed Phase 8G lock")
    required = [
        REPORTS / "phase8g_development_quality.json",
        REPORTS / "phase8g_shadow_validation_quality.json",
        REPORTS / "phase8g_blind_infrastructure_failure.json",
        REPORTS / "phase8h_infrastructure_validation.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError(
            "development, shadow, Phase 8G failure and infrastructure "
            "validation reports are required"
        )
    if json.loads(required[0].read_text()).get("status") != "PASS":
        raise RuntimeError("development Gate is not PASS")
    if json.loads(required[1].read_text()).get("status") != "PASS":
        raise RuntimeError("shadow validation Gate is not PASS")
    failure = json.loads(required[2].read_text())
    if failure.get("status") != "INFRASTRUCTURE_FAIL":
        raise RuntimeError("Phase 8G failure is not classified as infrastructure")
    if json.loads(required[3].read_text()).get("status") != "PASS":
        raise RuntimeError("Phase 8H infrastructure validation is not PASS")

    blind_root = ROOT / "data/phase8h_perception_protocol/blind"
    static_root = ROOT / "data/phase8h_perception_protocol/blind_static_maps"
    targets = (
        LOCK,
        blind_root,
        static_root,
        REPORTS / "phase8h_blind_once_quality.json",
        REPORTS / "phase8h_final_perception_gate.json",
        REPORTS / "phase8h_final_readiness.md",
    )
    for path in targets:
        if path.exists():
            raise FileExistsError(f"one-shot Phase 8H target exists: {path}")

    source_files = [
        ROOT / "policy/dynamic/dynamic_perception.py",
        ROOT / "policy/dynamic/range_image_foreground.py",
        ROOT / "policy/dynamic/track_manager.py",
        ROOT / "policy/dynamic/types.py",
        ROOT / "policy/dynamic/instance_evaluation.py",
        ROOT / "tools/evaluate_phase8g_instance_perception.py",
        ROOT / "tools/generate_phase8c_scenario_matrix.py",
        ROOT / "tools/generate_phase8g_scenario_matrix.py",
        ROOT / "tools/run_phase8c_formal_recording.py",
        ROOT / "tools/run_phase8g_instance_recording.py",
        ROOT / "tools/phase8h_blind_protocol.py",
        SIM / "src/include/dynamic_actor.hpp",
        SIM / "src/src/dynamic_actor.cpp",
        SIM / "src/src/test_simulator_cuda.cpp",
    ]
    freeze = {
        **frozen_hashes(source_files, required, args.blind_seed),
        "phase8g_blind_consumed": True,
        "phase8h_blind_run_count": 1,
        "workers": args.workers,
        "status": "STARTED_NO_RETRY",
    }
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
        "--split", "blind", "--phase-label", "phase8h",
        "--workers", str(args.workers), "--ros-master-port-base", "13500",
    ])
    assert_frozen(freeze, source_files, required, args.blind_seed)

    quality = evaluate(blind_root, "blind")
    quality["phase8h_blind_run_count"] = 1
    quality["phase8g_blind_consumed"] = True
    quality["freeze"] = freeze
    quality_path = REPORTS / "phase8h_blind_once_quality.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n")
    gate = {
        "status": quality["status"],
        "perception_ready": quality["status"] == "PASS",
        "phase8h_blind_run_count": 1,
        "phase8g_blind_consumed": True,
        "blind_manifest_sha256": hashlib.sha256(
            json.dumps(quality, sort_keys=True).encode()
        ).hexdigest(),
        "quality_report": str(quality_path),
    }
    (REPORTS / "phase8h_final_perception_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    (REPORTS / "phase8h_final_readiness.md").write_text(
        "# Phase 8H final readiness\n\n"
        f"Status: **{gate['status']}**\n\n"
        f"`perception_ready`: `{str(gate['perception_ready']).lower()}`  \n"
        "`phase8h_blind_run_count`: `1`  \n"
        "`phase8g_blind_consumed`: `true`\n"
    )
    print(json.dumps(gate, indent=2))
    if quality["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
