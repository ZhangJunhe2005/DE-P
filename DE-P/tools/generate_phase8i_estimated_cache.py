#!/usr/bin/env python3
"""Generate the frozen Phase-8H causal perception cache for Phase 8I.

Only depth, timestamps, camera poses, and camera calibration are read. Actor
annotations, instance masks, point clouds, and static maps are deliberately
outside this program's input surface.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, Pose


CACHE_FORMAT_VERSION = "phase8i_phase8h_context_v1"
ALLOWED_SPLITS = ("train", "valid")
FORBIDDEN_INPUT_TOKENS = ("dynamic_objects", "instance", ".ply", "esdf", "pointcloud")
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def phase8h_perception_config():
    """Return the effective config frozen by the Phase 8H blind evaluator."""
    return replace(
        DynamicPerceptionConfig.from_global_config(),
        foreground_mode="range_image_hybrid",
    )


def depth_path(sequence_dir, frame):
    relative = str(frame["depth_path"])
    lowered = relative.lower()
    if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
        raise RuntimeError(f"forbidden estimated-cache input path: {relative}")
    return sequence_dir / relative


def sequence_content_hash(sequence_dir, frames):
    digest = hashlib.sha256()
    for name in ("metadata.yaml", "frames.csv"):
        digest.update(name.encode())
        digest.update((sequence_dir / name).read_bytes())
    for frame in frames:
        path = depth_path(sequence_dir, frame)
        digest.update(str(path.relative_to(sequence_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def camera_model_from(metadata):
    intrinsics = metadata["camera_intrinsics"]
    return CameraModel(
        width=int(metadata["raw_image_width"]),
        height=int(metadata["raw_image_height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        depth_scale=float(metadata["depth_scale"]),
        min_depth=float(intrinsics["min_depth"]),
        max_depth=float(intrinsics["max_depth"]),
    )


def pose_from(metadata, frame):
    rotation_body = Rotation.from_quat([
        float(frame["camera_qx"]), float(frame["camera_qy"]),
        float(frame["camera_qz"]), float(frame["camera_qw"]),
    ]).as_matrix()
    body_position = np.asarray([
        frame["camera_x"], frame["camera_y"], frame["camera_z"],
    ], dtype=float)
    body_from_camera = np.asarray(metadata["camera_rotation_body_from_camera"], dtype=float)
    camera_offset = np.asarray(metadata["camera_position_body"], dtype=float)
    timestamp = float(frame["timestamp"])
    return Pose(
        body_position + rotation_body @ camera_offset,
        rotation_body @ body_from_camera,
        timestamp,
    )


def track_payload(track):
    return {
        "track_id": int(track.track_id),
        "position_world": torch.as_tensor(track.position_world, dtype=torch.float32),
        "velocity_world": torch.as_tensor(track.velocity_world, dtype=torch.float32),
        "state": torch.as_tensor(
            np.concatenate((track.position_world, track.velocity_world)), dtype=torch.float32
        ),
        "state_covariance": torch.as_tensor(track.state_covariance, dtype=torch.float32),
        "confidence": float(track.confidence),
        "active": bool(track.is_confirmed and track.is_dynamic and track.attention_authorized),
        "is_confirmed": bool(track.is_confirmed),
        "is_dynamic": bool(track.is_dynamic),
        "attention_authorized": bool(track.attention_authorized),
        "timestamp": float(track.timestamp),
        "age": int(track.age),
        "missed_count": int(track.missed_count),
        "provenance": {
            "birth_observation_id": int(track.birth_observation_id),
            "birth_frame": int(track.birth_frame),
            "last_observation_id": int(track.last_observation_id),
            "last_direct_observation_frame": int(track.last_direct_observation_frame),
            "direct_observation_count": int(track.direct_observation_count),
            "ever_directly_observed": bool(track.ever_directly_observed),
            "visibility_state": str(track.visibility_state),
            "dynamic_reason": str(track.dynamic_reason),
        },
    }


def tensor_content_hash(payload):
    digest = hashlib.sha256()
    digest.update(payload["cache_key"].encode())
    attention = payload["attention"].detach().cpu().contiguous().numpy()
    digest.update(attention.dtype.str.encode())
    digest.update(np.asarray(attention.shape, dtype=np.int64).tobytes())
    digest.update(attention.tobytes())
    for track in payload["tracks"]:
        for name in ("track_id", "confidence", "active", "is_confirmed", "is_dynamic",
                     "attention_authorized", "timestamp", "age", "missed_count"):
            digest.update(repr(track[name]).encode())
        for name in ("position_world", "velocity_world", "state", "state_covariance"):
            value = track[name].detach().cpu().contiguous().numpy()
            digest.update(value.dtype.str.encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        digest.update(json.dumps(track["provenance"], sort_keys=True).encode())
    return digest.hexdigest()


def process_sequence(task):
    (dataset_root_text, cache_root_text, split, sequence_id, dataset_manifest_hash,
     perception_hash, perception_config_hash, resume) = task
    dataset_root = Path(dataset_root_text)
    cache_root = Path(cache_root_text)
    sequence_dir = dataset_root / "sequences" / sequence_id
    metadata = load_yaml(sequence_dir / "metadata.yaml")
    with (sequence_dir / "frames.csv").open("r", newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    sequence_hash = sequence_content_hash(sequence_dir, frames)
    timestamp_hash = canonical_hash([frame["timestamp"] for frame in frames])
    camera_description = {
        "intrinsics": metadata["camera_intrinsics"],
        "depth_scale": metadata["depth_scale"],
        "raw_shape": [metadata["raw_image_height"], metadata["raw_image_width"]],
        "camera_position_body": metadata["camera_position_body"],
        "camera_rotation_body_from_camera": metadata["camera_rotation_body_from_camera"],
        "world_frame": metadata["world_frame"],
        "camera_optical_frame": metadata["camera_optical_frame"],
    }
    camera_hash = canonical_hash(camera_description)
    # This is the exact explicit override used by the frozen Phase 8H blind
    # evaluator. The YAML default remains temporal_voxel for legacy callers.
    config = phase8h_perception_config()
    perception = DynamicPerception(config, (cfg["vertical_num"], cfg["horizon_num"]))
    camera_model = camera_model_from(metadata)
    history = int(cfg["dynamic_training"]["history_length"])
    entries = {}
    hits = misses = 0
    for offset, frame in enumerate(frames):
        depth = np.load(depth_path(sequence_dir, frame), allow_pickle=False)
        timestamp = float(frame["timestamp"])
        result = perception.update_depth(
            depth, pose_from(metadata, frame), timestamp, camera_model
        )
        if offset < history - 1:
            continue
        frame_index = int(frame["frame_index"])
        key_fields = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "perception_implementation_hash": perception_hash,
            "perception_config_hash": perception_config_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "sequence_content_hash": sequence_hash,
            "timestamp_hash": timestamp_hash,
            "camera_model_hash": camera_hash,
            "split": split,
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "timestamp": frame["timestamp"],
        }
        cache_key = canonical_hash(key_fields)
        relative = Path(split) / sequence_id / f"{frame_index:06d}-{cache_key}.pt"
        output = cache_root / relative
        if resume and output.is_file():
            old = torch.load(output, map_location="cpu", weights_only=False)
            if old.get("cache_key") == cache_key and old.get("content_hash"):
                entries[f"{split}:{sequence_id}:{frame_index}"] = {
                    "path": str(relative), "cache_key": cache_key,
                    "content_hash": old["content_hash"],
                }
                hits += 1
                continue
        tracks = [track_payload(track) for track in result.all_tracks]
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": cache_key,
            "key_fields": key_fields,
            "attention": result.attention_map[0].detach().cpu().to(torch.float32),
            "target_positions": torch.stack(
                [item["position_world"] for item in tracks]
            ) if tracks else torch.empty((0, 3), dtype=torch.float32),
            "target_velocities": torch.stack(
                [item["velocity_world"] for item in tracks]
            ) if tracks else torch.empty((0, 3), dtype=torch.float32),
            "confidence": torch.tensor(
                [item["confidence"] for item in tracks], dtype=torch.float32
            ),
            "active_mask": torch.tensor(
                [item["active"] for item in tracks], dtype=torch.bool
            ),
            "timestamps": torch.tensor(
                [item["timestamp"] for item in tracks], dtype=torch.float64
            ),
            "tracks": tracks,
            "provenance_summary": {
                "input_source": "depth",
                "strictly_causal": True,
                "actor_ground_truth_used": False,
                "instance_mask_used": False,
                "static_map_used": False,
                "frames_consumed": offset + 1,
                "latest_timestamp": timestamp,
                "all_track_count": len(result.all_tracks),
                "dynamic_track_count": len(result.dynamic_tracks),
                "projected_track_count": len(result.projected_dynamic_tracks),
            },
        }
        payload["content_hash"] = tensor_content_hash(payload)
        atomic_torch(output, payload)
        entries[f"{split}:{sequence_id}:{frame_index}"] = {
            "path": str(relative), "cache_key": cache_key,
            "content_hash": payload["content_hash"],
        }
        misses += 1
    return {
        "split": split,
        "sequence_id": sequence_id,
        "frame_count": len(frames),
        "expected_window_count": max(0, len(frames) - history + 1),
        "sequence_content_hash": sequence_hash,
        "timestamp_hash": timestamp_hash,
        "camera_model_hash": camera_hash,
        "entries": entries,
        "cache_hits": hits,
        "cache_misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/phase8_dynamic_production"))
    parser.add_argument("--output", default=str(ROOT / "cache/phase8i_phase8h_perception"))
    parser.add_argument("--report", default=str(ROOT / "reports/phase8i_estimated_cache_validation.json"))
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--determinism-sequence", default="")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    cache_root = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if cache_root == dataset_root or dataset_root in cache_root.parents:
        pass
    manifest_path = dataset_root / "dataset_manifest.yaml"
    manifest = load_yaml(manifest_path)
    dataset_manifest_hash = sha256_file(manifest_path)
    entry_gate = json.loads(
        (ROOT / "reports/phase8i_entry_gate.json").read_text(encoding="utf-8")
    )
    if entry_gate.get("status") != "PASS" or not entry_gate.get("perception_ready"):
        raise RuntimeError("Phase 8I entry Gate is not PASS")
    perception_hash = entry_gate["frozen_hashes"]["perception_implementation_sha256"]
    locked_config_hash = entry_gate["frozen_hashes"]["perception_config_sha256"]
    if sha256_file(ROOT / "config/traj_opt.yaml") != locked_config_hash:
        raise RuntimeError("frozen Phase 8H perception config changed after entry Gate")
    perception_config = asdict(phase8h_perception_config())
    perception_runtime_config_hash = canonical_hash(perception_config)
    perception_config_hash = perception_runtime_config_hash

    tasks = []
    split_counts = {}
    for split in ALLOWED_SPLITS:
        split_file = dataset_root / manifest["splits"][split]
        sequences = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        split_counts[split] = len(sequences)
        for sequence_id in sequences:
            tasks.append((
                str(dataset_root), str(cache_root), split, sequence_id,
                dataset_manifest_hash, perception_hash, perception_config_hash,
                not args.no_resume,
            ))
    start = time.perf_counter()
    context = mp.get_context("spawn")
    if args.workers == 1:
        results = [process_sequence(task) for task in tasks]
    else:
        with context.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(process_sequence, tasks))
    results.sort(key=lambda item: (item["split"], item["sequence_id"]))
    entries = {}
    sequences = {}
    for result in results:
        entries.update(result.pop("entries"))
        sequences[f"{result['split']}:{result['sequence_id']}"] = result
    expected = {
        split: sum(item["expected_window_count"] for item in results
                   if item["split"] == split)
        for split in ALLOWED_SPLITS
    }
    actual = {
        split: sum(key.startswith(f"{split}:") for key in entries)
        for split in ALLOWED_SPLITS
    }
    if actual != expected:
        raise RuntimeError(f"cache completeness failure: actual={actual}, expected={expected}")
    if any(key.startswith("test:") for key in entries):
        raise RuntimeError("production test cache generation is forbidden")

    # Recompute a fixed sequence in a temporary sibling directory and compare
    # every window's canonical tensor hash.
    determinism_sequence = args.determinism_sequence or sorted(
        key.split(":", 1)[1] for key in sequences if key.startswith("valid:")
    )[0]
    deterministic_task = next(
        task for task in tasks if task[2] == "valid" and task[3] == determinism_sequence
    )
    det_root = cache_root / ".determinism"
    det_task = list(deterministic_task)
    det_task[1] = str(det_root)
    det_task[-1] = False
    det_result = process_sequence(tuple(det_task))
    baseline = {
        key: value["content_hash"] for key, value in entries.items()
        if key.startswith(f"valid:{determinism_sequence}:")
    }
    repeated = {
        key: value["content_hash"] for key, value in det_result["entries"].items()
    }
    deterministic = baseline == repeated and len(baseline) > 0
    if not deterministic:
        raise RuntimeError("estimated cache determinism check failed")

    index = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "perception_implementation_hash": perception_hash,
        "perception_config_hash": perception_config_hash,
        "frozen_config_file_hash": locked_config_hash,
        "perception_runtime_config_hash": perception_runtime_config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "allowed_splits": list(ALLOWED_SPLITS),
        "production_test_generated": False,
        "entries": dict(sorted(entries.items())),
        "sequences": sequences,
    }
    index["index_content_hash"] = canonical_hash(index)
    atomic_json(cache_root / "index.json", index)
    indexed_file_count = len(entries)
    split_cache_file_count = sum(
        1 for split in ALLOWED_SPLITS
        for _ in (cache_root / split).rglob("*.pt")
    )
    report = {
        "status": "PASS",
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_root": str(cache_root),
        "index": str(cache_root / "index.json"),
        "index_sha256": sha256_file(cache_root / "index.json"),
        "index_content_hash": index["index_content_hash"],
        "perception_implementation_hash": perception_hash,
        "perception_config_hash": perception_config_hash,
        "frozen_config_file_hash": locked_config_hash,
        "perception_runtime_config_hash": perception_runtime_config_hash,
        "effective_foreground_mode": perception_config["foreground_mode"],
        "dataset_manifest_hash": dataset_manifest_hash,
        "strictly_causal": True,
        "future_frames_used": False,
        "actor_ground_truth_used": False,
        "instance_mask_used": False,
        "static_map_or_esdf_used": False,
        "production_test_generated": False,
        "split_sequence_counts": split_counts,
        "split_window_counts": actual,
        "total_windows": len(entries),
        "indexed_cache_file_count": indexed_file_count,
        "unindexed_stale_file_count": split_cache_file_count - indexed_file_count,
        "unindexed_stale_file_status": (
            "excluded_by_index; first temporal_voxel attempt is never loaded"
            if split_cache_file_count > indexed_file_count else "none"
        ),
        "cache_hits": sum(item["cache_hits"] for item in results),
        "cache_misses": sum(item["cache_misses"] for item in results),
        "determinism": {
            "sequence_id": determinism_sequence,
            "window_count": len(baseline),
            "content_hashes_identical": deterministic,
        },
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
