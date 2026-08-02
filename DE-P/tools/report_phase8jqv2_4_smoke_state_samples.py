#!/usr/bin/env python3
"""Emit human-auditable consecutive V2 smoke states without changing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_frame(frame: dict) -> dict:
    return {
        "frame_index": frame["frame_index"],
        "timestamp_ns": frame["timestamp_ns"],
        "position_world": frame["position_world"],
        "velocity_world": frame["velocity_world"],
        "acceleration_world": frame["acceleration_world"],
        "quaternion_world_from_body_wxyz":
            frame["quaternion_world_from_body"],
        "goal_world": frame["goal_world"],
        "goal_body": frame["goal_body"],
        "actionability": frame["actionability"],
        "static_depth_frame_hash": frame["static_depth_frame_hash"],
    }


def sequence_sample(dataset: Path, sequence_manifest: dict) -> dict:
    sequence_path = dataset / sequence_manifest["path"]
    sequence = json.loads(sequence_path.read_text())
    base = (
        dataset / sequence["suite"] / sequence["split"]
        / sequence["sequence_id"]
    )
    frames = [
        json.loads(line)
        for line in (base / "frames.jsonl").read_text().splitlines()
    ]
    positions = np.asarray(
        [frame["position_world"] for frame in frames], dtype=np.float64
    )
    return {
        "sequence_id": sequence["sequence_id"],
        "split": sequence["split"],
        "scenario": sequence["scenario"],
        "position_span_m": float(np.linalg.norm(np.ptp(positions, axis=0))),
        "unique_static_depth_frames": len({
            frame["static_depth_frame_hash"] for frame in frames
        }),
        "render_diagnostics": json.loads(
            (base / "render_diagnostics.json").read_text()
        ),
        "consecutive_frames": [
            compact_frame(frame) for frame in frames[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=ROOT / "data/phase8_authoritative_v2_smoke",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=ROOT / "reports/phase8jqv2_4_smoke_state_samples.json",
        type=Path,
    )
    args = parser.parse_args()
    manifest_path = args.dataset / "manifests/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sequence_records = [
        (row, json.loads((args.dataset / row["path"]).read_text()))
        for row in manifest["sequences"]
    ]
    selected = []
    for wanted in ("normal_progress", "near_boundary_recovery_stress"):
        selected.append(next(
            row for row, sequence in sequence_records
            if sequence["split"] == "train"
            and sequence["scenario"] == wanted
        ))
    samples = [sequence_sample(args.dataset, row) for row in selected]
    result = {
        "status": "PASS" if all(
            sample["position_span_m"] > 0
            and sample["unique_static_depth_frames"] > 1
            and sample["render_diagnostics"].get("camera_pose_source")
            == "persisted_uav_state_v2"
            for sample in samples
        ) else "FAIL",
        "dataset_version": manifest["dataset_version"],
        "root_manifest_hash": manifest_hash(manifest_path),
        "samples": samples,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
