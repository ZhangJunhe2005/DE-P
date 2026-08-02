#!/usr/bin/env python3
"""Generate strictly causal estimated contexts for the frozen R0 selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import (
    CameraModel,
    DynamicPerceptionConfig,
    Pose,
)


DATASET = ROOT / "data/phase8_authoritative_v2"
SELECTION = ROOT / "artifacts/phase8jqv2_5/r0_selection_manifest.json"
EXPECTED_ROOT_HASH = \
    "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
# The authoritative renderer is not the ROS optical convention: its persisted
# image ray is body [1, +(u-cx)/fx, +(v-cy)/fy].  Therefore optical
# [+X right,+Y down,+Z forward] maps to body [+Y,+Z,+X].
CAMERA_BODY_FROM_OPTICAL = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perception_config():
    return replace(
        DynamicPerceptionConfig.from_global_config(),
        enabled=True,
        foreground_mode="range_image_hybrid",
        camera_width=160,
        camera_height=96,
        camera_fx=80.0,
        camera_fy=80.0,
        camera_cx=80.0,
        camera_cy=45.0,
    )


def camera_model():
    return CameraModel(
        width=160, height=96, fx=80.0, fy=80.0, cx=80.0, cy=45.0,
        depth_scale=1.0, min_depth=0.1, max_depth=20.0,
    )


def body_rotation(frame):
    w, x, y, z = frame["quaternion_world_from_body"]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def track_payload(track):
    return {
        "track_id": int(track.track_id),
        "position_world": np.asarray(track.position_world).tolist(),
        "velocity_world": np.asarray(track.velocity_world).tolist(),
        "state_covariance": np.asarray(track.state_covariance).tolist(),
        "confidence": float(track.confidence),
        "active": bool(
            track.is_confirmed and track.is_dynamic
            and track.attention_authorized
        ),
        "is_confirmed": bool(track.is_confirmed),
        "is_dynamic": bool(track.is_dynamic),
        "attention_authorized": bool(track.attention_authorized),
        "timestamp": float(track.timestamp),
        "age": int(track.age),
        "missed_count": int(track.missed_count),
    }


def process_sequence(arguments):
    sequence, frames_to_save, output = arguments
    base = (
        DATASET / sequence["suite"] / "valid" / sequence["sequence_id"]
    )
    frames = [
        json.loads(line)
        for line in (base / "frames.jsonl").read_text().splitlines()
    ]
    depth = np.load(base / "depth.npy", mmap_mode="r", allow_pickle=False)
    perception = DynamicPerception(
        perception_config(), (3, 5), attention_device="cpu"
    )
    model = camera_model()
    records = []
    selected = set(frames_to_save)
    sequence_output = Path(output) / "entries" / sequence["sequence_id"]
    sequence_output.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames):
        rotation = body_rotation(frame)
        timestamp = float(frame["timestamp_ns"]) / 1e9
        pose = Pose(
            np.asarray(frame["position_world"], dtype=np.float64),
            rotation @ CAMERA_BODY_FROM_OPTICAL,
            timestamp,
        )
        result = perception.update_depth(
            np.asarray(depth[frame_index], dtype=np.float32),
            pose, timestamp, model,
        )
        if frame_index not in selected:
            continue
        payload = {
            "cache_format_version": "phase8jqv2_5_estimated_r0_v1",
            "dataset_root_manifest_hash": EXPECTED_ROOT_HASH,
            "sequence_id": sequence["sequence_id"],
            "frame_index": frame_index,
            "timestamp": timestamp,
            "attention": result.attention_map.cpu(),
            "tracks": [track_payload(track) for track in result.dynamic_tracks],
            "input_source": "depth_only_strictly_causal",
            "foreground_mode": "range_image_hybrid",
            "future_actor_metadata_used": False,
        }
        path = sequence_output / f"{frame_index:04d}.pt"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        records.append({
            "sequence_id": sequence["sequence_id"],
            "frame_index": frame_index,
            "path": str(path.relative_to(output)),
            "sha256": sha256(path),
            "track_count": len(payload["tracks"]),
            "attention_nonzero": int(torch.count_nonzero(payload["attention"])),
        })
    if len(records) != len(selected):
        raise RuntimeError(
            f"cache selection mismatch for {sequence['sequence_id']}"
        )
    return sequence["sequence_id"], records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="artifacts/phase8jqv2_5/estimated_r0_cache"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    selection = json.loads(SELECTION.read_text())
    if selection["root_manifest_hash"] != EXPECTED_ROOT_HASH:
        raise RuntimeError("R0 selection root hash mismatch")
    if sha256(DATASET / "manifests/dataset_manifest.json") != EXPECTED_ROOT_HASH:
        raise RuntimeError("Formal V2 root manifest changed")
    requested = {}
    for row in selection["rows"]:
        requested.setdefault(row["sequence_id"], []).append(row["frame_index"])
    manifest = json.loads(
        (DATASET / "manifests/dataset_manifest.json").read_text()
    )
    sequences = {}
    for record in manifest["sequences"]:
        value = json.loads((DATASET / record["path"]).read_text())
        if value["sequence_id"] in requested:
            sequences[value["sequence_id"]] = value
    identifiers = sorted(requested)
    if args.max_sequences:
        identifiers = identifiers[:args.max_sequences]
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (sequences[identifier], requested[identifier], str(output))
        for identifier in identifiers
    ]
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_sequence, task): task[0]["sequence_id"]
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            identifier, records = future.result()
            entries.extend(records)
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({
                    "status": "RUNNING",
                    "sequences_complete": completed,
                    "sequences_total": len(tasks),
                    "entries_complete": len(entries),
                    "last_sequence": identifier,
                }))
    entries.sort(key=lambda row: (row["sequence_id"], row["frame_index"]))
    config = perception_config()
    cache = {
        "status": "PASS",
        "cache_format_version": "phase8jqv2_5_estimated_r0_v1",
        "dataset_version": "phase8_authoritative_v2",
        "dataset_root_manifest_hash": EXPECTED_ROOT_HASH,
        "selection_manifest_hash": selection["selection_manifest_hash"],
        "perception_config": asdict(config),
        "perception_config_hash": canonical_hash(asdict(config)),
        "renderer_camera_semantics":
            "body_ray=[1,+image_right,+image_down]",
        "camera_body_from_optical": CAMERA_BODY_FROM_OPTICAL.tolist(),
        "strictly_causal": True,
        "future_actor_metadata_used": False,
        "production_test_used": False,
        "blind_used": False,
        "sequence_count": len(identifiers),
        "entry_count": len(entries),
        "complete_selection": (
            not args.max_sequences and len(entries) == selection["frame_count"]
        ),
        "entries": entries,
    }
    cache["cache_manifest_hash"] = canonical_hash(cache)
    (output / "index.json").write_text(json.dumps(cache, indent=2) + "\n")
    print(json.dumps({
        key: value for key, value in cache.items() if key != "entries"
    }, indent=2))


if __name__ == "__main__":
    main()
