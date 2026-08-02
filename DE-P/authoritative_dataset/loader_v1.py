"""Fail-closed read-only loader for Authoritative Dataset Protocol V1."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import (
    DATASET_VERSION, FORMAL_DATASET_VERSION, FORMAL_DATASET_VERSION_V2,
    PROTOCOL_VERSION,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuthoritativePilotDataset:
    def __init__(self, root, split="pilot"):
        self.root = Path(root)
        manifest_path = self.root / "manifests/dataset_manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("dataset protocol mismatch")
        if self.manifest["dataset_version"] != DATASET_VERSION:
            raise RuntimeError("dataset version mismatch")
        if split != "pilot":
            raise RuntimeError("test/train/valid access forbidden in pilot")
        for relative, expected in self.manifest["bound_files"].items():
            if sha256(self.root / relative) != expected:
                raise RuntimeError(f"manifest hash mismatch: {relative}")
        self.frames = []
        for sequence in self.manifest["sequences"]:
            sequence_path = self.root / sequence["manifest"]
            if sha256(sequence_path) != sequence["sha256"]:
                raise RuntimeError("sequence manifest mismatch")
            value = json.loads(sequence_path.read_text())
            for frame in value["frames"]:
                frame_path = self.root / frame["path"]
                if sha256(frame_path) != frame["sha256"]:
                    raise RuntimeError("frame manifest mismatch")
                self.frames.append(frame_path)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        value = json.loads(self.frames[index].read_text())
        if value["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("frame protocol mismatch")
        authority = Path(value["authority_artifact"])
        metadata = json.loads(
            (authority / "occupancy_metadata.json").read_text()
        )
        if (
            metadata["artifact_manifest_hash"]
            != value["authority_manifest_hash"]
            or metadata["occupancy_hash"] != value["occupancy_hash"]
            or metadata["map_uuid"] != value["map_uuid"]
        ):
            raise RuntimeError("frame authority mismatch")
        certificate = self.root / value["certificate_path"]
        if sha256(certificate) != value["certificate_hash"]:
            raise RuntimeError("certificate hash mismatch")
        depth_path = self.root / value["depth_path"]
        if sha256(depth_path) != value["depth_image_hash"]:
            raise RuntimeError("depth hash mismatch")
        depth = np.load(depth_path, allow_pickle=False)
        if not np.isfinite(depth).all():
            raise RuntimeError("non-finite depth")
        return {"metadata": value, "depth": depth}

    def semantic_hash(self, index):
        sample = self[index]
        digest = hashlib.sha256()
        digest.update(json.dumps(
            sample["metadata"], sort_keys=True,
            separators=(",", ":")).encode())
        digest.update(sample["depth"].tobytes())
        return digest.hexdigest()


class AuthoritativeFormalDataset(Dataset):
    """Lazy, fail-closed train/valid loader for formal authoritative data."""

    def __init__(
        self, root, split, *, expected_root_hash=None, cache_size=8,
        verify_files=True, max_depth_m=20.0,
    ):
        self.root = Path(root).expanduser().resolve()
        if split not in {"train", "valid"}:
            raise ValueError("formal loader permits only train or valid")
        self.split = split
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.verify_files = bool(verify_files)
        self.max_depth_m = float(max_depth_m)
        if self.max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        manifest_path = self.root / "manifests/dataset_manifest.json"
        self.root_manifest_hash = sha256(manifest_path)
        if (
            expected_root_hash is not None
            and self.root_manifest_hash != expected_root_hash
        ):
            raise RuntimeError("formal root manifest hash mismatch")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError("dataset protocol mismatch")
        if self.manifest["dataset_version"] not in {
            FORMAL_DATASET_VERSION, FORMAL_DATASET_VERSION_V2,
        }:
            raise RuntimeError("formal dataset version mismatch")
        if self.manifest.get("test_generated", False):
            raise RuntimeError("formal loader refuses a manifest with test data")
        if self.manifest.get("blind_access_count", 0):
            raise RuntimeError("formal loader refuses blind-used data")
        self._map_indices = self._build_map_indices()
        self._sequences = []
        self._cumulative_frames = []
        total = 0
        seen = set()
        for record in self.manifest["sequences"]:
            path = self.root / record["path"]
            if sha256(path) != record["sha256"]:
                raise RuntimeError(
                    f"sequence manifest mismatch: {record['path']}"
                )
            sequence = json.loads(path.read_text())
            if sequence["split"] != split:
                continue
            identifier = sequence["sequence_id"]
            if identifier in seen:
                raise RuntimeError(f"duplicate sequence id: {identifier}")
            seen.add(identifier)
            if sequence["dataset_version"] != self.manifest["dataset_version"]:
                raise RuntimeError("sequence dataset version mismatch")
            if sequence["frame_count"] <= 0:
                raise RuntimeError("empty sequence")
            self._sequences.append((record, sequence))
            total += int(sequence["frame_count"])
            self._cumulative_frames.append(total)
        if not self._sequences:
            raise RuntimeError(f"formal split is empty: {split}")
        self._length = total
        self._cache = OrderedDict()

    def _build_map_indices(self):
        records = []
        for record in self.manifest["maps"]:
            path = self.root / record["path"]
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"map manifest mismatch: {record['path']}")
            value = json.loads(path.read_text())
            records.append((value["split"], value["map_uuid"]))
        return {
            key: index for index, key in enumerate(sorted(records))
        }

    def __len__(self):
        return self._length

    def _location(self, index):
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        sequence_index = bisect_right(self._cumulative_frames, index)
        previous = (
            self._cumulative_frames[sequence_index-1]
            if sequence_index else 0
        )
        return sequence_index, index-previous

    def _package(self, sequence_index):
        record, sequence = self._sequences[sequence_index]
        identifier = sequence["sequence_id"]
        if identifier in self._cache:
            value = self._cache.pop(identifier)
            self._cache[identifier] = value
            return value
        base = (
            self.root / sequence["suite"] / sequence["split"] / identifier
        )
        if self.verify_files:
            for relative, expected in sequence["files"].items():
                if sha256(self.root / relative) != expected:
                    raise RuntimeError(f"sequence child hash mismatch: {relative}")
        frames = [
            json.loads(line)
            for line in (base / "frames.jsonl").read_text().splitlines()
        ]
        certificates = [
            json.loads(line)
            for line in (
                base / "certificates.jsonl"
            ).read_text().splitlines()
        ]
        if len(frames) != sequence["frame_count"] or len(certificates) != len(
            frames
        ):
            raise RuntimeError(f"incomplete sequence package: {identifier}")
        depth = np.load(base / "depth.npy", mmap_mode="r", allow_pickle=False)
        if depth.shape[0] != len(frames) or depth.dtype != np.float32:
            raise RuntimeError(f"depth package mismatch: {identifier}")
        diagnostic = json.loads((base / "render_diagnostics.json").read_text())
        value = (sequence, frames, certificates, depth, diagnostic)
        self._cache[identifier] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        sequence_index, frame_index = self._location(index)
        sequence, frames, certificates, depth, diagnostic = self._package(
            sequence_index
        )
        frame = frames[frame_index]
        certificate = certificates[frame_index]
        depth_m = np.asarray(depth[frame_index], dtype=np.float32).copy()
        if depth_m.shape != (96, 160) or not np.isfinite(depth_m).all():
            raise RuntimeError("formal depth must be finite [96,160]")
        if frame["frame_index"] != frame_index:
            raise RuntimeError("frame index mismatch")
        if frame["sequence_id"] != sequence["sequence_id"]:
            raise RuntimeError("frame sequence mismatch")
        observation = np.asarray(
            frame["velocity_body"]
            + frame["acceleration_body"]
            + frame["goal_body"],
            dtype=np.float32,
        )
        if observation.shape != (9,) or not np.isfinite(observation).all():
            raise RuntimeError("formal observation must be finite [9]")
        action = frame["actionability"]
        return {
            "depth_m": torch.from_numpy(depth_m[None]),
            "current_depth": torch.from_numpy(
                (depth_m / self.max_depth_m)[None]
            ),
            "observation_9d": torch.from_numpy(observation),
            "position_world": torch.tensor(
                frame["position_world"], dtype=torch.float32
            ),
            "velocity_world": torch.tensor(
                frame["velocity_world"], dtype=torch.float32
            ),
            "acceleration_world": torch.tensor(
                frame["acceleration_world"], dtype=torch.float32
            ),
            "quaternion_world_from_body_wxyz": torch.tensor(
                frame["quaternion_world_from_body"], dtype=torch.float32
            ),
            "goal_world": torch.tensor(
                frame["goal_world"], dtype=torch.float32
            ),
            "goal_body": torch.tensor(
                frame["goal_body"], dtype=torch.float32
            ),
            "map_index": torch.tensor(
                self._map_indices[(self.split, frame["map_uuid"])],
                dtype=torch.long,
            ),
            "stress": torch.tensor(bool(action["stress"])),
            "recoverable": torch.tensor(bool(action["recoverable"])),
            "unknown": torch.tensor(bool(action["feasibility_unknown"])),
            "actor_count": torch.tensor(
                len(frame["actor_metadata"]), dtype=torch.long
            ),
            "timestamp_ns": torch.tensor(
                frame["timestamp_ns"], dtype=torch.long
            ),
            "sequence_id": sequence["sequence_id"],
            "scenario": sequence["scenario"],
            "suite": sequence["suite"],
            "frame_index": frame_index,
            "certificate": certificate,
            "actor_metadata": frame["actor_metadata"],
            "renderer_version": diagnostic["renderer_version"],
        }

    def semantic_hash(self, index):
        sample = self[index]
        digest = hashlib.sha256()
        for name in (
            "depth_m", "current_depth", "observation_9d", "position_world",
            "velocity_world", "acceleration_world", "goal_world", "goal_body",
        ):
            digest.update(
                sample[name].detach().cpu().contiguous().numpy().tobytes()
            )
        digest.update(sample["sequence_id"].encode())
        digest.update(str(sample["frame_index"]).encode())
        return digest.hexdigest()


def authoritative_formal_collate(samples):
    if not samples:
        raise ValueError("cannot collate an empty formal batch")
    tensor_keys = (
        "depth_m", "current_depth", "observation_9d", "position_world",
        "velocity_world", "acceleration_world",
        "quaternion_world_from_body_wxyz", "goal_world", "goal_body",
        "map_index", "stress", "recoverable", "unknown", "actor_count",
        "timestamp_ns",
    )
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in tensor_keys
    }
    for key in (
        "sequence_id", "scenario", "suite", "frame_index", "certificate",
        "actor_metadata", "renderer_version",
    ):
        batch[key] = [sample[key] for sample in samples]
    return batch
