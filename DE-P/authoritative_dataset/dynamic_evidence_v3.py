"""Formal V3 evidence-chain primitives: deterministic, causal and no-pickle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np


CONTRACT_VERSION = "dynamic_evidence_formal_v3_generator_v1"
LABELS = (
    "DYNAMIC_SUPPORT", "STATIC_SUPPORT",
    "SENSOR_OR_SEGMENTATION_ARTIFACT", "UNKNOWN_AMBIGUOUS",
)
MOTION_THRESHOLD_MPS = 0.3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*identity):
    return int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:8], "big")


def motion_bucket(speed_mps):
    speed = float(speed_mps)
    if speed == 0.0:
        return "ZERO_MOTION"
    if 0.0 < speed < 0.5 * MOTION_THRESHOLD_MPS:
        return "LOW_SUBTHRESHOLD"
    if 0.5 * MOTION_THRESHOLD_MPS <= speed < MOTION_THRESHOLD_MPS:
        return "NEAR_THRESHOLD_NEGATIVE"
    if MOTION_THRESHOLD_MPS <= speed < 2.0 * MOTION_THRESHOLD_MPS:
        return "ACTIVE_DYNAMIC"
    return "HIGH_DYNAMIC"


def sampled_constant_velocity(start, velocity, frame_count, dt):
    times = np.arange(int(frame_count), dtype=np.float64) * float(dt)
    start = np.asarray(start, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    positions = start[None, :] + times[:, None] * velocity[None, :]
    actual = np.linalg.norm(np.diff(positions, axis=0), axis=1) / float(dt)
    return positions, actual


def split_identity(row):
    required = (
        "scene_family", "map_identity", "map_seed",
        "actor_trajectory_family", "actor_seed", "renderer_seed",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"split identity missing: {missing}")
    return sha256_bytes(canonical([row[key] for key in required]))


def find_leakage(rows):
    owners, leaks = {}, []
    for row in rows:
        identity = split_identity(row)
        split = row["split"]
        if identity in owners and owners[identity] != split:
            leaks.append({"identity": identity, "splits": sorted({owners[identity], split})})
        owners[identity] = split
    return leaks


def atomic_json(path, value, fail_stage=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if fail_stage == "before_write":
        raise OSError("injected before write")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if fail_stage == "after_write_before_rename":
        raise OSError("injected after write before rename")
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return sha256_file(path)


class ExclusiveRootLock:
    def __init__(self, root):
        self.path = Path(root) / "generation_state/formal_v3.lock"

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "time_ns": time.time_ns()}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("active or stale formal V3 lock requires explicit audit") from error
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def release(self):
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


def save_numeric_shard(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"pickle/object array forbidden: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"non-finite shard: {name}")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def load_numeric_shard(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("shard hash mismatch")
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key].copy() for key in value.files}


__all__ = [
    "CONTRACT_VERSION", "LABELS", "MOTION_THRESHOLD_MPS", "atomic_json",
    "ExclusiveRootLock", "find_leakage", "load_numeric_shard", "motion_bucket",
    "sampled_constant_velocity", "save_numeric_shard", "sha256_file",
    "split_identity", "stable_seed",
]
