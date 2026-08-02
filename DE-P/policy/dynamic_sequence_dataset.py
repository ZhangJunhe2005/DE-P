"""Lazy loader for strictly validated continuous dynamic sequences."""

from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

from config.config import cfg
from policy.dynamic.attention import build_dynamic_attention
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, DynamicTrack, Pose
from policy.dynamic_training_config import DynamicTrainingConfig


MANIFEST_FIELDS = {
    "dataset_version", "sensor_source", "time_unit", "distance_unit", "splits"
}
METADATA_FIELDS = {
    "dataset_version", "sequence_id", "sensor_source", "source_topic",
    "depth_encoding", "depth_scale", "raw_image_width", "raw_image_height",
    "network_image_width", "network_image_height", "camera_intrinsics",
    "camera_position_body", "camera_rotation_body_from_camera", "world_frame",
    "body_frame", "camera_optical_frame", "pointcloud_frame", "simulator_commit",
    "dep_commit", "configuration_hash", "random_seed", "time_unit",
    "distance_unit", "acceleration_method",
}
FRAME_FIELDS = {
    "sequence_id", "frame_index", "timestamp", "depth_path", "pointcloud_path",
    "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz",
    "camera_qw", "velocity_x", "velocity_y", "velocity_z", "acceleration_x",
    "acceleration_y", "acceleration_z", "goal_x", "goal_y", "goal_z", "map_id",
    "dynamic_objects_path", "scenario_id", "seed", "depth_odom_offset",
    "depth_camera_info_offset", "depth_gt_offset",
}
OBJECT_FIELDS = {
    "object_id", "position_world", "velocity_world", "position_covariance",
    "radius", "type", "visibility", "occluded", "dynamic",
}


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def validate_estimated_cache_index(index, manifest_path, perception_config, split):
    if index.get("cache_format_version") != "phase8i_phase8h_context_v1":
        raise ValueError("unsupported indexed estimated-context cache format")
    if index.get("production_test_generated"):
        raise ValueError("estimated-context cache must not contain production test")
    if split not in set(index.get("allowed_splits", ())):
        raise ValueError(f"indexed estimated cache does not allow split {split}")
    declared_content_hash = index.get("index_content_hash")
    content = dict(index)
    content.pop("index_content_hash", None)
    if not declared_content_hash or _canonical_hash(content) != declared_content_hash:
        raise ValueError("indexed estimated-context index content hash mismatch")
    manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if index.get("dataset_manifest_hash") != manifest_hash:
        raise ValueError("indexed estimated-context dataset manifest hash mismatch")
    runtime_hash = _canonical_hash(asdict(perception_config))
    if index.get("perception_runtime_config_hash") != runtime_hash:
        raise ValueError(
            "indexed estimated-context foreground mode/perception config hash mismatch"
        )
    return index


def require_index_for_nonempty_estimated_cache(cache_root):
    cache_root = Path(cache_root)
    index_path = cache_root / "index.json"
    if index_path.is_file():
        return index_path
    if cache_root.exists() and any(cache_root.rglob("*.pt")):
        raise ValueError(
            "estimated cache contains unindexed files; unindexed cache is invisible"
        )
    return None


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def _require_exact_fields(data, required, label):
    missing = required - set(data)
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")


def validate_dataset_splits(root):
    root = Path(root)
    manifest = _load_yaml(root / "dataset_manifest.yaml")
    _require_exact_fields(manifest, MANIFEST_FIELDS, "dataset manifest")
    split_sequences = {}
    seen = set()
    seeds = {}
    maps_by_split = {split: set() for split in ("train", "valid", "test")}
    for split in ("train", "valid", "test"):
        split_path = root / manifest["splits"][split]
        sequences = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        overlap = seen.intersection(sequences)
        if overlap:
            raise ValueError(f"sequence split leakage: {sorted(overlap)}")
        seen.update(sequences)
        split_sequences[split] = sequences
        for sequence_id in sequences:
            metadata = _load_yaml(root / "sequences" / sequence_id / "metadata.yaml")
            seed = int(metadata["random_seed"])
            if seed in seeds and seeds[seed] != split:
                raise ValueError(
                    f"random seed {seed} leaks across {seeds[seed]} and {split}"
                )
            seeds[seed] = split
            frames_path = root / "sequences" / sequence_id / "frames.csv"
            with frames_path.open("r", newline="", encoding="utf-8") as stream:
                map_ids = {int(row["map_id"]) for row in csv.DictReader(stream)}
            if len(map_ids) != 1:
                raise ValueError(f"sequence {sequence_id} must use exactly one map_id")
            maps_by_split[split].update(map_ids)
    if "map_splits" in manifest:
        declared = {split: {int(value) for value in manifest["map_splits"][split]}
                    for split in ("train", "valid", "test")}
        for split in declared:
            if maps_by_split[split] != declared[split]:
                raise ValueError(f"declared map split does not match sequences for {split}")
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = declared[left] & declared[right]
            if overlap:
                raise ValueError(f"map split leakage: {sorted(overlap)}")
    return manifest, split_sequences


class DynamicSequenceDataset(Dataset):
    def __init__(self, root, split="train", training_config=None,
                 perception_config=None):
        self.root = Path(root).expanduser().resolve()
        self.training_config = training_config or DynamicTrainingConfig.from_global_config()
        self.perception_config = perception_config or DynamicPerceptionConfig.from_global_config()
        self.training_config.validate()
        self.perception_config.validate()
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")
        self.split = split
        self.manifest, split_sequences = validate_dataset_splits(self.root)
        if self.manifest["dataset_version"] != self.training_config.dataset_version:
            raise ValueError("dataset version does not match dynamic_training config")
        if self.manifest["sensor_source"] != "depth":
            raise ValueError("official phase-6 training source must be depth")
        self._depth_cache = OrderedDict()
        self._object_cache = OrderedDict()
        self._curriculum_context_source = self.training_config.context_source
        self._curriculum_ratio = 1.0
        self._curriculum_epoch = 0
        self._estimated_cache_index = None
        if self.training_config.estimated_cache_dir:
            cache_root = Path(
                self.training_config.estimated_cache_dir
            ).expanduser().resolve()
            index_path = require_index_for_nonempty_estimated_cache(
                cache_root
            )
            if index_path is not None:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                self._estimated_cache_index = validate_estimated_cache_index(
                    index, self.root / "dataset_manifest.yaml",
                    self.perception_config, self.split,
                )
        self._sequences = {}
        self._windows = []
        for sequence_id in split_sequences[split]:
            self._load_sequence_index(sequence_id)
        if not self._windows:
            raise ValueError(f"split {split!r} contains no complete windows")

    @property
    def cached_depth_count(self):
        return len(self._depth_cache)

    def _load_sequence_index(self, sequence_id):
        directory = self.root / "sequences" / sequence_id
        metadata = _load_yaml(directory / "metadata.yaml")
        _require_exact_fields(metadata, METADATA_FIELDS, f"metadata {sequence_id}")
        if metadata["sequence_id"] != sequence_id:
            raise ValueError("sequence_id does not match directory")
        if metadata["dataset_version"] != self.manifest["dataset_version"]:
            raise ValueError("sequence dataset version mismatch")
        if metadata["sensor_source"] != "depth" or metadata["source_topic"] == "/lidar_points":
            raise ValueError("legacy /lidar_points is forbidden in dynamic sequence data")
        if metadata["world_frame"] != self.perception_config.world_frame_id:
            raise ValueError("sequence world frame mismatch")
        if metadata["camera_optical_frame"] != self.perception_config.camera_frame_id:
            raise ValueError("sequence optical frame mismatch")
        with (directory / "frames.csv").open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != FRAME_FIELDS:
                raise ValueError("frames.csv schema mismatch")
            frames = list(reader)
        timestamps = np.asarray([float(frame["timestamp"]) for frame in frames])
        indices = np.asarray([int(frame["frame_index"]) for frame in frames])
        if len(frames) < self.training_config.history_length:
            return
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"timestamps are not strictly increasing in {sequence_id}")
        if not np.all(np.diff(indices) == 1):
            raise ValueError(f"missing or duplicate frame in {sequence_id}")
        for frame in frames:
            if frame["sequence_id"] != sequence_id:
                raise ValueError("frame sequence_id mismatch")
            for key in ("depth_path", "dynamic_objects_path"):
                if not (directory / frame[key]).is_file():
                    raise FileNotFoundError(directory / frame[key])
            if frame["pointcloud_path"] and "/lidar_points" in frame["pointcloud_path"]:
                raise ValueError("legacy lidar data is forbidden")
            offsets = [abs(float(frame[name])) for name in (
                "depth_odom_offset", "depth_camera_info_offset", "depth_gt_offset"
            )]
            if max(offsets) > float(metadata.get("sync_slop_seconds", 0.03)):
                raise ValueError("recorded frame exceeds synchronization slop")
        self._sequences[sequence_id] = (directory, metadata, frames)
        history = self.training_config.history_length
        for current in range(history - 1, len(frames), self.training_config.window_stride):
            scenario = str(metadata.get("scenario_type", metadata.get("scenario_id", ""))).lower()
            current_objects = self._objects(directory / frames[current]["dynamic_objects_path"])
            history_start = max(0, current - history + 1)
            seen = set()
            for frame in frames[history_start:current + 1]:
                seen.update(int(obj["object_id"]) for obj in self._objects(
                    directory / frame["dynamic_objects_path"]
                ) if self._visible(obj))
            observable = [obj for obj in current_objects if int(obj["object_id"]) in seen]
            if not observable:
                category = "no_target"
            elif len(observable) > 1 or "multi" in scenario:
                category = "multi_target"
            elif any(not self._visible(obj) for obj in observable):
                category = "occluded_but_tracked"
            elif "temporal" in scenario or "delay" in scenario:
                category = "temporally_separated"
            elif "cross" in scenario or "head" in scenario:
                camera_position = np.asarray([
                    frames[current]["camera_x"], frames[current]["camera_y"],
                    frames[current]["camera_z"],
                ], dtype=float)
                camera_velocity = np.asarray([
                    frames[current]["velocity_x"], frames[current]["velocity_y"],
                    frames[current]["velocity_z"],
                ], dtype=float)
                predicted_clearances = []
                for obj in observable:
                    relative_position = np.asarray(obj["position_world"], dtype=float) - camera_position
                    relative_velocity = np.asarray(obj["velocity_world"], dtype=float) - camera_velocity
                    speed_squared = float(relative_velocity @ relative_velocity)
                    closest_time = float(np.clip(
                        -(relative_position @ relative_velocity) / max(speed_squared, 1e-9),
                        0.0, float(cfg["sgm_time"]),
                    ))
                    predicted_clearances.append(float(np.linalg.norm(
                        relative_position + relative_velocity * closest_time
                    ) - float(obj["radius"]) - 0.3))
                category = (
                    "visible_high_risk" if min(predicted_clearances) < 0.5
                    else "visible_low_risk"
                )
            else:
                category = "visible_low_risk"
            self._windows.append((sequence_id, current, category))

    def __len__(self):
        return len(self._windows)

    @property
    def sample_categories(self):
        return [entry[2] for entry in self._windows]

    def set_curriculum(self, epoch, context_source, ratio):
        if context_source not in {"ground_truth", "noisy_ground_truth", "estimated"}:
            raise ValueError("invalid curriculum context source")
        if not 0 < float(ratio) <= 1:
            raise ValueError("curriculum ratio must be in (0,1]")
        self._curriculum_epoch = int(epoch)
        self._curriculum_context_source = context_source
        self._curriculum_ratio = float(ratio)

    def _load_depth(self, path):
        path = Path(path)
        key = str(path)
        if key in self._depth_cache:
            value = self._depth_cache.pop(key)
            self._depth_cache[key] = value
            return value.copy()
        if path.suffix == ".npy":
            value = np.load(path, allow_pickle=False).astype(np.float32)
        else:
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if value is None:
                raise ValueError(f"failed to read depth: {path}")
            value = value.astype(np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"depth must be finite 2-D: {path}")
        self._depth_cache[key] = value.copy()
        while len(self._depth_cache) > self.training_config.cache_size:
            self._depth_cache.popitem(last=False)
        return value

    def _objects(self, path):
        key = str(Path(path))
        if key in self._object_cache:
            return [dict(item) for item in self._object_cache[key]]
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("dynamic object annotation must be a list")
        for item in data:
            _require_exact_fields(item, OBJECT_FIELDS, "dynamic object")
        self._object_cache[key] = data
        while len(self._object_cache) > self.training_config.cache_size:
            self._object_cache.popitem(last=False)
        return [dict(item) for item in data]

    @staticmethod
    def _visible(obj):
        return not bool(obj["occluded"]) and float(obj["visibility"]) > 0

    def _observable_objects(self, directory, selected):
        history_objects = [self._objects(directory / frame["dynamic_objects_path"])
                           for frame in selected]
        last_visible = {}
        ever_observed = set()
        for frame, objects in zip(selected, history_objects):
            for obj in objects:
                if self._visible(obj):
                    actor_id = int(obj["object_id"])
                    ever_observed.add(actor_id)
                    last_visible[actor_id] = (dict(obj), float(frame["timestamp"]))
        current_by_id = {int(obj["object_id"]): dict(obj) for obj in history_objects[-1]}
        observable = []
        now = float(selected[-1]["timestamp"])
        for actor_id in sorted(ever_observed):
            current = current_by_id.get(actor_id)
            if current is None:
                continue
            last, last_timestamp = last_visible[actor_id]
            if not self._visible(current):
                age = now - last_timestamp
                if age > self.perception_config.max_dt * self.perception_config.max_missed_frames:
                    continue
            else:
                last, last_timestamp = current, now
            label = dict(current)
            label["context_position_world"] = (
                np.asarray(last["position_world"], dtype=float)
                + np.asarray(last["velocity_world"], dtype=float) * (now - last_timestamp)
            ).tolist()
            label["context_velocity_world"] = list(last["velocity_world"])
            label["track_timestamp"] = last_timestamp
            label["ever_observed_in_history"] = True
            label["observable"] = True
            label["supervision_confidence"] = max(
                0.05, float(last["visibility"]) * np.exp(-(now - last_timestamp))
            )
            observable.append(label)
        return observable

    def _future_ground_truth(self, directory, frames, current_index, objects, timestamp):
        points = int(cfg["dynamic_loss"]["eval_points"])
        offsets = np.linspace(cfg["sgm_time"] / points, cfg["sgm_time"], points)
        query_times = timestamp + offsets
        positions = np.zeros((len(objects), points, 3), dtype=np.float32)
        valid = np.zeros((len(objects), points), dtype=bool)
        visible = np.zeros_like(valid)
        samples_by_id = {int(obj["object_id"]): [] for obj in objects}
        # Reading future annotations is confined to label construction. Neither
        # DynamicContext nor the network input receives any value from this block.
        for frame in frames[current_index:]:
            frame_time = float(frame["timestamp"])
            for obj in self._objects(directory / frame["dynamic_objects_path"]):
                actor_id = int(obj["object_id"])
                if actor_id in samples_by_id:
                    samples_by_id[actor_id].append((frame_time, obj))
        for actor_index, obj in enumerate(objects):
            samples = samples_by_id[int(obj["object_id"])]
            sample_times = np.asarray([entry[0] for entry in samples])
            for future_index, query in enumerate(query_times):
                right = int(np.searchsorted(sample_times, query, side="left"))
                if right < len(samples) and abs(sample_times[right] - query) <= 1e-6:
                    positions[actor_index, future_index] = samples[right][1]["position_world"]
                    valid[actor_index, future_index] = True
                    visible[actor_index, future_index] = self._visible(samples[right][1])
                elif 0 < right < len(samples):
                    left_time, left = samples[right - 1]
                    right_time, right_obj = samples[right]
                    alpha = (query - left_time) / (right_time - left_time)
                    positions[actor_index, future_index] = (
                        (1.0 - alpha) * np.asarray(left["position_world"], dtype=float)
                        + alpha * np.asarray(right_obj["position_world"], dtype=float)
                    )
                    valid[actor_index, future_index] = True
                    visible[actor_index, future_index] = self._visible(left) and self._visible(right_obj)
        # Absolute ROS stamps can be ~1e9 seconds.  float32 has insufficient
        # resolution there and collapses the complete 1.7 s supervision grid.
        return positions, valid, visible, query_times.astype(np.float64)

    def _noise_rng(self, sequence_id, frame_index):
        token = f"{self.split}:{sequence_id}:{frame_index}:{self.training_config.noise_seed}"
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
        return np.random.default_rng(seed)

    def _ground_truth_attention(self, objects, camera_pose, camera_model, sequence_id,
                                frame_index, context_source):
        noisy = context_source == "noisy_ground_truth"
        rng = self._noise_rng(sequence_id, frame_index)
        tracks = []
        for obj in objects:
            position = np.asarray(obj["context_position_world"], dtype=float)
            velocity = np.asarray(obj["context_velocity_world"], dtype=float)
            if noisy:
                position += rng.normal(0, self.training_config.position_noise_std, 3)
                velocity += rng.normal(0, self.training_config.velocity_noise_std, 3)
            covariance = np.eye(6) * 0.01
            covariance[:3, :3] = np.asarray(obj["position_covariance"], dtype=float)
            tracks.append(DynamicTrack(
                track_id=int(obj["object_id"]), position_world=position,
                velocity_world=velocity, state_covariance=covariance,
                age=self.training_config.history_length, hit_count=self.training_config.history_length,
                missed_count=int(not self._visible(obj)), is_confirmed=True,
                is_dynamic=bool(obj["dynamic"]), timestamp=float(obj["track_timestamp"]),
                dynamic_reason="observable_ground_truth",
            ))
        attention = build_dynamic_attention(
            tracks, camera_pose, camera_model,
            (cfg["vertical_num"], cfg["horizon_num"]), self.perception_config,
        )[0]
        if noisy and self.training_config.attention_noise_std:
            attention = (attention + torch.from_numpy(rng.normal(
                0, self.training_config.attention_noise_std, tuple(attention.shape)
            )).float()).clamp(0, self.perception_config.attention_max)
        return attention

    def _estimated_attention(self, raw_depths, selected, metadata, camera_model,
                             sequence_id, frame_index):
        if self._estimated_cache_index is not None:
            lookup = f"{self.split}:{sequence_id}:{frame_index}"
            entry = self._estimated_cache_index["entries"].get(lookup)
            if entry is None:
                raise KeyError(f"indexed estimated-context cache miss: {lookup}")
            cache_root = Path(self.training_config.estimated_cache_dir).expanduser().resolve()
            cache_path = (cache_root / entry["path"]).resolve()
            if cache_root not in cache_path.parents:
                raise ValueError("estimated-context cache entry escapes cache root")
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if payload.get("cache_key") != entry["cache_key"]:
                raise ValueError(f"indexed estimated-context cache key mismatch: {lookup}")
            if payload.get("content_hash") != entry["content_hash"]:
                raise ValueError(f"indexed estimated-context content hash mismatch: {lookup}")
            attention = payload["attention"]
            expected = (1, cfg["vertical_num"], cfg["horizon_num"])
            if tuple(attention.shape) != expected or not bool(torch.isfinite(attention).all()):
                raise ValueError(f"invalid cached attention for {lookup}")
            return attention
        cache_path = None
        cache_key = hashlib.sha256(json.dumps({
            "tracker_version": "phase8f_causal_range_image_v1",
            "perception_config": asdict(self.perception_config),
            "split": self.split,
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "sequence_hash": metadata["configuration_hash"],
            "timestamps": [frame["timestamp"] for frame in selected],
        }, sort_keys=True).encode()).hexdigest()
        if self.training_config.estimated_cache_dir:
            cache_dir = Path(self.training_config.estimated_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{cache_key}.pt"
            if cache_path.is_file():
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                if payload.get("cache_key") == cache_key:
                    return payload["attention"]
        perception = DynamicPerception(
            self.perception_config, (cfg["vertical_num"], cfg["horizon_num"])
        )
        result = None
        body_from_camera = np.asarray(metadata["camera_rotation_body_from_camera"], dtype=float)
        camera_offset = np.asarray(metadata["camera_position_body"], dtype=float)
        for depth, frame in zip(raw_depths, selected):
            rotation_body = Rotation.from_quat([
                float(frame["camera_qx"]), float(frame["camera_qy"]),
                float(frame["camera_qz"]), float(frame["camera_qw"])
            ]).as_matrix()
            body_position = np.asarray([frame["camera_x"], frame["camera_y"], frame["camera_z"]], dtype=float)
            timestamp = float(frame["timestamp"])
            pose = Pose(body_position + rotation_body @ camera_offset,
                        rotation_body @ body_from_camera, timestamp)
            result = perception.update_depth(depth, pose, timestamp, camera_model)
        attention = result.attention_map[0]
        if cache_path is not None:
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            torch.save({"cache_key": cache_key, "attention": attention}, temporary)
            os.replace(temporary, cache_path)
        return attention

    def __getitem__(self, item):
        sequence_id, current_index, sample_category = self._windows[item]
        directory, metadata, frames = self._sequences[sequence_id]
        history_length = self.training_config.history_length
        selected = frames[current_index - history_length + 1:current_index + 1]
        raw_depths = [self._load_depth(directory / frame["depth_path"]) for frame in selected]
        expected_shape = (int(metadata["raw_image_height"]), int(metadata["raw_image_width"]))
        if any(depth.shape != expected_shape for depth in raw_depths):
            raise ValueError("raw depth shape does not match sequence metadata")
        network_depths = [
            cv2.resize(
                depth, (int(metadata["network_image_width"]), int(metadata["network_image_height"])),
                interpolation=cv2.INTER_NEAREST,
            ) / float(metadata["camera_intrinsics"]["max_depth"])
            for depth in raw_depths
        ]
        depth_history = torch.from_numpy(np.stack(network_depths)[:, None]).float()
        current = selected[-1]
        timestamp = float(current["timestamp"])
        position = np.asarray([current["camera_x"], current["camera_y"], current["camera_z"]], dtype=float)
        quaternion = np.asarray([
            current["camera_qx"], current["camera_qy"], current["camera_qz"], current["camera_qw"]
        ], dtype=float)
        rotation_world_from_body = Rotation.from_quat(quaternion).as_matrix()
        camera_rotation = rotation_world_from_body @ np.asarray(
            metadata["camera_rotation_body_from_camera"], dtype=float
        )
        camera_position = position + rotation_world_from_body @ np.asarray(
            metadata["camera_position_body"], dtype=float
        )
        camera_pose = Pose(camera_position, camera_rotation, timestamp)
        intrinsics = metadata["camera_intrinsics"]
        camera_model = CameraModel(
            width=int(metadata["raw_image_width"]), height=int(metadata["raw_image_height"]),
            fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]),
            depth_scale=float(metadata["depth_scale"]), min_depth=float(intrinsics["min_depth"]),
            max_depth=float(intrinsics["max_depth"]),
        )
        objects = self._observable_objects(directory, selected)
        future_positions, future_valid, future_visibility, future_timestamps = (
            self._future_ground_truth(directory, frames, current_index, objects, timestamp)
        )
        context_source = self._curriculum_context_source
        if self._curriculum_ratio < 1.0:
            token = f"curriculum:{self.split}:{sequence_id}:{current['frame_index']}:{self._curriculum_epoch}"
            draw = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") / 2**64
            if draw >= self._curriculum_ratio:
                context_source = "ground_truth"
        if context_source in {"ground_truth", "noisy_ground_truth"}:
            attention = self._ground_truth_attention(
                objects, camera_pose, camera_model, sequence_id,
                int(current["frame_index"]), context_source,
            )
        else:
            attention = self._estimated_attention(
                raw_depths, selected, metadata, camera_model,
                sequence_id, int(current["frame_index"]),
            )
        velocity_world = np.asarray([current["velocity_x"], current["velocity_y"], current["velocity_z"]], dtype=float)
        acceleration_world = np.asarray([
            current["acceleration_x"], current["acceleration_y"], current["acceleration_z"]
        ], dtype=float)
        goal_world = np.asarray([current["goal_x"], current["goal_y"], current["goal_z"]], dtype=float)
        rotation_body_from_world = rotation_world_from_body.T
        observation = np.concatenate((
            rotation_body_from_world @ velocity_world,
            rotation_body_from_world @ acceleration_world,
            rotation_body_from_world @ (goal_world - position),
        )).astype(np.float32)
        return {
            "depth_history": depth_history,
            "current_depth": depth_history[-1],
            "timestamps": torch.tensor([float(frame["timestamp"]) for frame in selected]),
            "camera_positions_world": torch.tensor([
                [float(frame["camera_x"]), float(frame["camera_y"]), float(frame["camera_z"])]
                for frame in selected
            ]),
            "observation_9d": torch.from_numpy(observation),
            "position_world": torch.from_numpy(position.astype(np.float32)),
            "rotation_world_from_body": torch.from_numpy(rotation_world_from_body.astype(np.float32)),
            "goal_world": torch.from_numpy(goal_world.astype(np.float32)),
            "map_id": int(current["map_id"]),
            "attention": attention,
            "objects": objects,
            "future_positions_world": torch.from_numpy(future_positions),
            "future_valid_mask": torch.from_numpy(future_valid),
            "future_visibility_mask": torch.from_numpy(future_visibility),
            "future_timestamps": torch.from_numpy(future_timestamps),
            "sample_timestamp": timestamp,
            "sequence_mask": torch.ones(history_length, dtype=torch.bool),
            "sequence_id": sequence_id,
            "frame_index": int(current["frame_index"]),
            "sample_category": sample_category,
        }
