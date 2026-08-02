#!/usr/bin/env python3
"""Generate deterministic schema-compatible sequences for smoke tests only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from ruamel.yaml import YAML


FRAME_FIELDS = [
    "sequence_id", "frame_index", "timestamp", "depth_path", "pointcloud_path",
    "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz",
    "camera_qw", "velocity_x", "velocity_y", "velocity_z", "acceleration_x",
    "acceleration_y", "acceleration_z", "goal_x", "goal_y", "goal_z", "map_id",
    "dynamic_objects_path", "scenario_id", "seed", "depth_odom_offset",
    "depth_camera_info_offset", "depth_gt_offset",
]


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def dump_yaml(path, data):
    yaml = YAML()
    yaml.default_flow_style = False
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.dump(data, stream)


def render_depth(camera_position, objects):
    height, width = 90, 160
    fx = fy = 80.0
    cx, cy = 80.0, 45.0
    depth = np.full((height, width), 20.0, dtype=np.float32)
    # body FLU to optical RDF: (x,y,z) -> (-y,-z,x)
    for obj in objects:
        relative = np.asarray(obj["position_world"]) - camera_position
        optical = np.asarray([-relative[1], -relative[2], relative[0]])
        if optical[2] <= 0:
            continue
        u = int(round(fx * optical[0] / optical[2] + cx))
        v = int(round(fy * optical[1] / optical[2] + cy))
        radius = max(1, int(round(fx * obj["radius"] / optical[2])))
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2
        depth[mask] = min(float(optical[2]), 20.0)
    return depth


def generate_sequence(root, sequence_id, seed, scenario, frame_count=10, dt=0.1,
                      map_id=0, static_map_sha256=None):
    directory = root / "sequences" / sequence_id
    (directory / "depth").mkdir(parents=True)
    (directory / "dynamic_objects").mkdir()
    frames = []
    for frame_index in range(frame_count):
        timestamp = frame_index * dt
        camera_position = np.asarray([0.02 * frame_index, 0.0, 2.0])
        objects = []
        if scenario != "empty":
            if scenario == "head_on_cylinder":
                position = [4.5 - 0.15 * frame_index, 0.0, 2.0]
                velocity = [-1.5, 0.0, 0.0]
            elif scenario == "waypoint_ping_pong":
                direction = 1.0 if frame_index < frame_count // 2 else -1.0
                phase = frame_index if direction > 0 else frame_count - frame_index
                position = [3.0, -1.0 + 0.15 * phase, 2.0]
                velocity = [0.0, 1.5 * direction, 0.0]
            elif scenario == "delayed_linear":
                active_step = max(4, frame_count // 3)
                phase = max(0, frame_index - active_step)
                position = [3.0, -1.0 + 0.15 * phase, 2.0]
                velocity = [0.0, 0.0 if frame_index < active_step else 1.5, 0.0]
            else:
                position = [3.0, -1.0 + 0.2 * frame_index, 2.0]
                velocity = [0.0, 2.0, 0.0]
            occluded = scenario == "crossing_occlusion" and frame_index in {5, 6}
            objects.append({
                "object_id": 1,
                "position_world": position,
                "velocity_world": velocity,
                "position_covariance": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
                "radius": 0.35,
                "type": ("synthetic_cylinder" if scenario == "head_on_cylinder"
                         else "synthetic_sphere"),
                "visibility": 0.0 if occluded else 1.0,
                "occluded": occluded,
                "dynamic": True,
            })
            if scenario == "multi_target":
                objects.append({
                    "object_id": 2,
                    "position_world": [4.0, 1.4 - 0.16 * frame_index, 2.2],
                    "velocity_world": [0.0, -1.6, 0.0],
                    "position_covariance": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
                    "radius": 0.3, "type": "synthetic_sphere", "visibility": 1.0,
                    "occluded": False, "dynamic": True,
                })
        depth_relative = Path("depth") / f"{frame_index:06d}.npy"
        objects_relative = Path("dynamic_objects") / f"{frame_index:06d}.json"
        visible_objects = [obj for obj in objects if not obj["occluded"]]
        np.save(directory / depth_relative, render_depth(camera_position, visible_objects))
        (directory / objects_relative).write_text(json.dumps(objects, indent=2), encoding="utf-8")
        frames.append({
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "timestamp": f"{timestamp:.6f}",
            "depth_path": str(depth_relative),
            "pointcloud_path": "",
            "camera_x": float(camera_position[0]), "camera_y": 0.0, "camera_z": 2.0,
            "camera_qx": 0.0, "camera_qy": 0.0, "camera_qz": 0.0, "camera_qw": 1.0,
            "velocity_x": 0.2, "velocity_y": 0.0, "velocity_z": 0.0,
            "acceleration_x": 0.0, "acceleration_y": 0.0, "acceleration_z": 0.0,
            "goal_x": 8.0, "goal_y": 0.0, "goal_z": 2.0,
            "map_id": int(map_id),
            "dynamic_objects_path": str(objects_relative),
            "scenario_id": scenario,
            "seed": seed,
            "depth_odom_offset": 0.0,
            "depth_camera_info_offset": 0.0,
            "depth_gt_offset": 0.0,
        })
    with (directory / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        writer.writerows(frames)
    configuration = {
        "scenario": scenario, "frame_count": frame_count, "dt": dt, "seed": seed,
        "map_id": int(map_id),
        "camera": [160, 90, 80.0, 80.0, 80.0, 45.0],
    }
    metadata = {
        "dataset_version": "dep_dynamic_sequence_v1",
        "sequence_id": sequence_id,
        "sensor_source": "depth",
        "source_topic": "/synthetic/depth",
        "depth_encoding": "32FC1",
        "depth_scale": 1.0,
        "raw_image_width": 160,
        "raw_image_height": 90,
        "network_image_width": 160,
        "network_image_height": 96,
        "camera_intrinsics": {"fx": 80.0, "fy": 80.0, "cx": 80.0, "cy": 45.0,
                              "min_depth": 0.1, "max_depth": 20.0},
        "camera_position_body": [0.0, 0.0, 0.0],
        "camera_rotation_body_from_camera": [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        "world_frame": "world",
        "body_frame": "quadrotor",
        "camera_optical_frame": "camera_optical",
        "pointcloud_frame": "none",
        "simulator_commit": git_revision("/home/zjh/YOPO/Simulator"),
        "dep_commit": git_revision(Path(__file__).resolve().parents[1]),
        "configuration_hash": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "random_seed": seed,
        "time_unit": "second",
        "distance_unit": "meter",
        "acceleration_method": "synthetic_ground_truth",
        "completion_status": "complete",
        "scenario_id": scenario,
        "scenario_type": scenario,
        "frame_count": frame_count,
        "record_rate_hz": 1.0 / dt,
        "sync_slop_seconds": 0.0,
        "map_id": int(map_id),
        "static_map_sha256": static_map_sha256 or "synthetic_unknown",
    }
    dump_yaml(directory / "metadata.yaml", metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    (output / "splits").mkdir(parents=True)
    definitions = {
        "train": [("sequence_000001", 101, "crossing"),
                  ("sequence_000002", 102, "empty")],
        "valid": [("sequence_000003", 201, "head_on_cylinder")],
        "test": [("sequence_000004", 301, "crossing_occlusion")],
    }
    for split, sequences in definitions.items():
        (output / "splits" / f"{split}.txt").write_text(
            "".join(f"{sequence_id}\n" for sequence_id, _, _ in sequences), encoding="utf-8"
        )
        for sequence_id, seed, scenario in sequences:
            generate_sequence(output, sequence_id, seed, scenario, args.frames)
    dump_yaml(output / "dataset_manifest.yaml", {
        "dataset_version": "dep_dynamic_sequence_v1",
        "sensor_source": "depth",
        "time_unit": "second",
        "distance_unit": "meter",
        "splits": {split: f"splits/{split}.txt" for split in definitions},
    })
    print(json.dumps({
        "status": "PASS", "dataset_source": "synthetic_smoke_only",
        "output": str(output), "sequence_count": 4,
    }, indent=2))


if __name__ == "__main__":
    main()
