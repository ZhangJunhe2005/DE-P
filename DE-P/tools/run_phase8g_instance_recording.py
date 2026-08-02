#!/usr/bin/env python3
"""Parallel, atomic Phase 8G depth+instance recorder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import threading
import uuid

import numpy as np
from ruamel.yaml import YAML

try:
    from run_phase8c_formal_recording import run_worker
except ModuleNotFoundError:  # Imported as tools.run_phase8g_instance_recording.
    from tools.run_phase8c_formal_recording import run_worker


ROOT = Path(__file__).resolve().parents[1]


def validate_scenario_identity_schema(rows):
    """Fail before ROS startup if scenario or 32SC1 identity values overflow."""
    yaml = YAML(typ="safe")
    seen = set()
    for row in rows:
        document = yaml.load(Path(row["scenario_file"]))
        scenario = document["dynamic_scenario"]
        seed = int(scenario["seed"])
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError(f"{row['sequence_id']}: scenario seed exceeds uint32")
        for actor in scenario.get("actors", []):
            actor_id = int(actor["id"])
            if not 1 <= actor_id <= 2**31 - 1:
                raise ValueError(
                    f"{row['sequence_id']}: actor ID {actor_id} exceeds ROS 32SC1"
                )
            if actor_id in seen:
                raise ValueError(f"duplicate actor ID across matrix: {actor_id}")
            seen.add(actor_id)


def validate(staging, rows):
    yaml = YAML(typ="safe")
    alignment_errors = []
    for row in rows:
        directory = staging / "sequences" / row["sequence_id"]
        metadata = yaml.load(directory / "metadata.yaml")
        if metadata.get("dataset_version") != "dep_dynamic_instance_sequence_v2":
            raise ValueError("Phase 8G sequence lacks instance schema")
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        if len(frames) != 60:
            raise ValueError(f"{row['sequence_id']}: expected 60 frames")
        for frame in frames:
            depth = np.load(directory / frame["depth_path"], allow_pickle=False)
            instance = np.load(directory / frame["instance_path"], allow_pickle=False)
            if depth.shape != (90, 160) or depth.dtype != np.float32:
                raise ValueError("depth schema mismatch")
            if instance.shape != depth.shape or instance.dtype != np.int32:
                raise ValueError("instance schema mismatch")
            if float(frame["depth_instance_offset"]) != 0.0:
                raise ValueError("depth and instance stamps are not exact")
            objects = json.loads(
                (directory / frame["dynamic_objects_path"]).read_text()
            )
            known = {int(obj["object_id"]) for obj in objects}
            mask_ids = {int(x) for x in np.unique(instance) if int(x)}
            if not mask_ids <= known:
                raise ValueError("instance image contains unknown actor ID")
            for obj in objects:
                error = abs(
                    int(np.count_nonzero(instance == int(obj["object_id"])))
                    - int(obj["rendered_pixel_count"])
                )
                alignment_errors.append(error)
                if error:
                    raise ValueError("instance count disagrees with shared z-buffer GT")
    return {
        "status": "PASS", "sequences": len(rows),
        "frames": len(rows) * 60,
        "instance_mask_depth_alignment_error": max(alignment_errors, default=0),
        "runtime_input": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-catalog", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ros-master-port-base", type=int, default=13100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenario-type")
    parser.add_argument("--phase-label", default="phase8g")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    logs = staging / "logs"
    logs.mkdir()
    scenarios = staging / "scenario_configs"
    import subprocess
    subprocess.run([
        str(Path(os.sys.executable)),
        str(ROOT / "tools/generate_phase8g_scenario_matrix.py"),
        "--catalog", str(args.static_catalog), "--output", str(scenarios),
        "--split", args.split, "--phase-label", args.phase_label,
    ], check=True)
    with (scenarios / "matrix.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if args.scenario_type:
        rows = [row for row in rows if row["scenario_type"] == args.scenario_type]
    if args.limit is not None:
        rows = rows[:args.limit]
    validate_scenario_identity_schema(rows)
    worker_count = min(args.workers, len(rows))
    buckets = [rows[index::worker_count] for index in range(worker_count)]
    progress = [0, len(rows)]
    lock, cancel = threading.Lock(), threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_worker, worker, args.ros_master_port_base + worker,
                    bucket, staging, logs, progress, lock, cancel,
                )
                for worker, bucket in enumerate(buckets)
            ]
            for future in as_completed(futures):
                future.result()
        result = validate(staging, rows)
        (staging / "instance_validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "splits").mkdir()
        (staging / "splits" / f"{args.split}.txt").write_text(
            "".join(f"{row['sequence_id']}\n" for row in rows), encoding="utf-8"
        )
        manifest = {
            "dataset_version": "dep_dynamic_instance_sequence_v2",
            "split": args.split, "sequences": len(rows),
            "instance_mask_evaluator_only": True,
            "completion_status": "complete",
        }
        YAML().dump(manifest, staging / "dataset_manifest.yaml")
        os.replace(staging, output)
        print(json.dumps({**result, "output": str(output)}, indent=2))
    except BaseException:
        (staging / "INCOMPLETE").write_text("recording did not commit\n")
        raise


if __name__ == "__main__":
    main()
