#!/usr/bin/env python3
"""Diagnose every dynamic track emitted in formal no-target sequences.

The static-map nearest-neighbour oracle in this tool is diagnostic only.  It is
never imported by the runtime perception pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from ruamel.yaml import YAML
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.pointcloud import camera_points_to_world
from policy.dynamic.projection import project_world_points_to_image
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, Pose
from policy.dynamic_sequence_dataset import validate_dataset_splits


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def camera_model(metadata):
    values = metadata["camera_intrinsics"]
    return CameraModel(
        width=int(metadata["raw_image_width"]), height=int(metadata["raw_image_height"]),
        fx=float(values["fx"]), fy=float(values["fy"]), cx=float(values["cx"]),
        cy=float(values["cy"]), depth_scale=float(metadata["depth_scale"]),
        min_depth=float(values["min_depth"]), max_depth=float(values["max_depth"]),
    )


def camera_pose(frame, metadata):
    rotation_body = Rotation.from_quat([
        float(frame[name]) for name in ("camera_qx", "camera_qy", "camera_qz", "camera_qw")
    ]).as_matrix()
    position_body = np.asarray([
        frame["camera_x"], frame["camera_y"], frame["camera_z"]
    ], dtype=float)
    offset = np.asarray(metadata["camera_position_body"], dtype=float)
    body_from_camera = np.asarray(metadata["camera_rotation_body_from_camera"], dtype=float)
    return Pose(position_body + rotation_body @ offset,
                rotation_body @ body_from_camera, float(frame["timestamp"]))


def voxel_signature(points, size=0.25):
    return {tuple(row) for row in np.floor(np.asarray(points) / size).astype(np.int64)}


def overlap_ratio(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / max(min(len(left), len(right)), 1)


def geometry_class(extent, eigenvalues, depth_range, bbox):
    extent = np.asarray(extent)
    eigenvalues = np.maximum(np.asarray(eigenvalues), 1e-12)
    if depth_range[1] >= 19.5:
        return "depth_boundary"
    if bbox[0][2] < 0.35 and extent[0] > 1.0 and extent[1] > 1.0:
        return "ground"
    if eigenvalues[0] / eigenvalues[-1] < 0.03 and extent.max() > 1.0:
        return "wall_or_large_plane"
    if extent.max() < 0.4:
        return "compact_or_sparse"
    return "other_surface"


def mechanism(entry):
    ratio = entry.get("previous_current_cluster_size_ratio")
    iou = entry.get("bbox_overlap")
    if ratio is not None and (ratio < 0.4 or ratio > 2.5):
        return "cluster_split_merge"
    if iou is not None and iou < 0.01 and entry.get("association_distance", 0) > 0.3:
        return "wrong_association"
    if entry["geometry_class"] in {"ground", "wall_or_large_plane"}:
        return "viewpoint_centroid_drift"
    if entry.get("innovation_norm", 0) > 0.75:
        return "innovation_outlier"
    return "kf_velocity_response"


def render_example(path, depth, points_world, cluster_world, history, velocity, attention,
                   camera_pose_value, camera_value):
    figure = plt.figure(figsize=(13, 8))
    ax = figure.add_subplot(2, 2, 1)
    ax.imshow(depth, cmap="viridis")
    if len(cluster_world):
        pixels = project_world_points_to_image(cluster_world, camera_pose_value, camera_value)
        if len(pixels):
            ax.scatter(pixels[:, 0], pixels[:, 1], s=2, c="red")
    ax.set_title("depth + false cluster")
    ax = figure.add_subplot(2, 2, 2, projection="3d")
    sample = points_world[::max(1, len(points_world) // 3000)]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1, alpha=0.15)
    if len(cluster_world):
        ax.scatter(cluster_world[:, 0], cluster_world[:, 1], cluster_world[:, 2], s=2, c="red")
    ax.set_title("world-frame cluster")
    ax = figure.add_subplot(2, 2, 3)
    history_array = np.asarray(history)
    if len(history_array):
        ax.plot(history_array[:, 0], history_array[:, 1], "o-")
        ax.quiver(history_array[-1, 0], history_array[-1, 1], velocity[0], velocity[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("centroid history + estimated velocity")
    ax = figure.add_subplot(2, 2, 4)
    ax.imshow(attention, vmin=0, vmax=1, cmap="magma")
    ax.set_title("attention")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--map-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--max-visualizations", type=int, default=12)
    args = parser.parse_args()
    root = args.dataset.resolve()
    _manifest, splits = validate_dataset_splits(root)
    catalog = load_yaml(args.map_catalog)
    map_paths = {int(row["map_id"]): Path(row["static_ply"]) for row in catalog["maps"]}
    config = DynamicPerceptionConfig.from_global_config()
    records, sequence_stats = [], {}
    trees = {}
    histories = defaultdict(list)
    previous_signatures = {}
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    visual_count = 0
    for split, sequences in splits.items():
        for sequence in sequences:
            directory = root / "sequences" / sequence
            metadata = load_yaml(directory / "metadata.yaml")
            if metadata.get("scenario_type") != "no_target":
                continue
            with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
                frames = list(csv.DictReader(stream))
            model = camera_model(metadata)
            perception = DynamicPerception(config, (cfg["vertical_num"], cfg["horizon_num"]))
            map_id = int(frames[0]["map_id"])
            if map_id not in trees:
                cloud = o3d.io.read_point_cloud(str(map_paths[map_id]))
                static_points = np.asarray(cloud.points)
                trees[map_id] = cKDTree(static_points)
            false_frames = 0
            for frame in frames:
                depth = np.load(directory / frame["depth_path"], allow_pickle=False)
                pose_value = camera_pose(frame, metadata)
                depth_frame = make_depth_frame(
                    depth, model, pose_value, float(frame["timestamp"]), config.depth_stride
                )
                points_camera = depth_frame.points_camera
                points_world = camera_points_to_world(points_camera, pose_value)
                observations, labels = perception.clustering.cluster(
                    points_world, points_camera, float(frame["timestamp"])
                )
                result = perception.update_depth(
                    depth, pose_value, float(frame["timestamp"]), model
                )
                match_by_track = {
                    int(item["track_id"]): item
                    for item in result.diagnostics["track_manager"].get("match_details", ())
                }
                observation_by_index = {index: value for index, value in enumerate(observations)}
                if result.dynamic_tracks:
                    false_frames += 1
                projected = {item.track_id: item for item in result.projected_dynamic_tracks}
                for track in result.dynamic_tracks:
                    match = match_by_track.get(track.track_id, {})
                    observation = observation_by_index.get(match.get("observation_index"))
                    if observation is None:
                        cluster_world = np.empty((0, 3))
                        cluster_camera = np.empty((0, 3))
                        extent = np.zeros(3)
                        covariance_eigenvalues = np.zeros(3)
                        bbox = np.stack((track.position_world, track.position_world))
                        point_count = 0
                    else:
                        mask = labels == observation.temporary_cluster_id
                        cluster_world, cluster_camera = points_world[mask], points_camera[mask]
                        bbox = observation.bounding_box_world
                        extent = bbox[1] - bbox[0]
                        covariance_eigenvalues = np.linalg.eigvalsh(
                            np.cov(cluster_world, rowvar=False)
                            if len(cluster_world) > 1 else np.zeros((3, 3))
                        )
                        point_count = observation.point_count
                    signature = voxel_signature(cluster_world)
                    prior_signature = previous_signatures.get(track.track_id, set())
                    point_overlap = overlap_ratio(signature, prior_signature)
                    previous_signatures[track.track_id] = signature
                    histories[(sequence, track.track_id)].append(track.position_world.tolist())
                    oracle_distance = trees[map_id].query(
                        cluster_world[::max(1, len(cluster_world) // 500)], workers=1
                    )[0] if len(cluster_world) else np.asarray([])
                    projected_track = projected.get(track.track_id)
                    velocity_std = float(np.sqrt(max(
                        np.max(np.diag(track.state_covariance)[3:]), 0.0
                    )))
                    depth_range = (
                        [float(cluster_camera[:, 2].min()), float(cluster_camera[:, 2].max())]
                        if len(cluster_camera) else [None, None]
                    )
                    geometry = geometry_class(
                        extent, covariance_eigenvalues,
                        [0.0, 0.0] if depth_range[0] is None else depth_range, bbox,
                    )
                    record = {
                        "split": split, "sequence_id": sequence,
                        "frame_index": int(frame["frame_index"]),
                        "track_id": int(track.track_id), "map_id": map_id,
                        "cluster_point_count": point_count,
                        "centroid_world": track.position_world.tolist(),
                        "bounding_box_world": bbox.tolist(), "extent": extent.tolist(),
                        "covariance_eigenvalues": covariance_eigenvalues.tolist(),
                        "planarity": float(1.0 - covariance_eigenvalues[0] /
                                           max(covariance_eigenvalues[-1], 1e-12)),
                        "linearity": float((covariance_eigenvalues[-1] - covariance_eigenvalues[-2]) /
                                           max(covariance_eigenvalues[-1], 1e-12)),
                        "compactness": float(covariance_eigenvalues.sum()),
                        "depth_range": depth_range,
                        "pixel": projected_track.pixel.tolist() if projected_track else None,
                        "feature_position": (
                            projected_track.feature_coordinate.tolist() if projected_track else None
                        ),
                        "track_age": track.age, "hit_count": track.hit_count,
                        "missed_count": track.missed_count,
                        "speed": float(np.linalg.norm(track.velocity_world)),
                        "velocity_world": track.velocity_world.tolist(),
                        "velocity_covariance_std": velocity_std,
                        "innovation": match.get("innovation"),
                        "innovation_norm": float(np.linalg.norm(match.get("innovation", [0, 0, 0]))),
                        "association_distance": match.get("association_distance"),
                        "association_mahalanobis_sq": match.get("association_mahalanobis_sq"),
                        "previous_current_cluster_size_ratio": match.get(
                            "previous_current_cluster_size_ratio"
                        ),
                        "bbox_overlap": match.get("bbox_iou"),
                        "point_overlap_ratio": point_overlap,
                        "dynamic_reason": track.dynamic_reason,
                        "attention_weight": (
                            projected_track.attention_weight if projected_track else 0.0
                        ),
                        "static_oracle_centroid_distance_m": float(
                            trees[map_id].query(track.position_world)[0]
                        ),
                        "static_oracle_point_distance_mean_m": (
                            float(oracle_distance.mean()) if len(oracle_distance) else None
                        ),
                        "static_oracle_point_fraction_within_0_25m": (
                            float((oracle_distance <= 0.25).mean()) if len(oracle_distance) else None
                        ),
                        "geometry_class": geometry,
                    }
                    record["root_mechanism"] = mechanism(record)
                    records.append(record)
                    if visual_count < args.max_visualizations and observation is not None:
                        render_example(
                            args.visual_dir / f"{sequence}_f{record['frame_index']:03d}_t{track.track_id}.png",
                            depth, points_world, cluster_world,
                            histories[(sequence, track.track_id)], track.velocity_world,
                            result.attention_map[0, 0].cpu().numpy(), pose_value, model,
                        )
                        visual_count += 1
            sequence_stats[sequence] = {
                "split": split, "frames": len(frames), "false_dynamic_frames": false_frames,
                "false_dynamic_frame_fraction": false_frames / len(frames),
            }
    geometry_counts = Counter(item["geometry_class"] for item in records)
    mechanism_counts = Counter(item["root_mechanism"] for item in records)
    oracle_values = [item["static_oracle_point_fraction_within_0_25m"] for item in records
                     if item["static_oracle_point_fraction_within_0_25m"] is not None]
    payload = {
        "status": "PASS", "diagnostic_only_static_map_oracle": True,
        "runtime_uses_static_map_oracle": False,
        "no_target_sequence_count": len(sequence_stats),
        "false_track_record_count": len(records),
        "sequence_stats": sequence_stats,
        "geometry_counts": dict(geometry_counts),
        "geometry_fractions": {key: value / max(len(records), 1)
                               for key, value in geometry_counts.items()},
        "root_mechanism_counts": dict(mechanism_counts),
        "root_mechanism_fractions": {key: value / max(len(records), 1)
                                     for key, value in mechanism_counts.items()},
        "static_oracle_false_points_within_0_25m_mean_fraction": (
            float(np.mean(oracle_values)) if oracle_values else None
        ),
        "visualization_count": visual_count,
        "visualization_directory": str(args.visual_dir.resolve()),
        "false_tracks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in {"false_tracks", "sequence_stats"}}, indent=2))


if __name__ == "__main__":
    main()
