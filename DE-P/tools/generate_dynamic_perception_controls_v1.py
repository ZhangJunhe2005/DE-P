#!/usr/bin/env python3
"""Generate development-only physical controls with the canonical CUDA renderer."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import (  # noqa: E402
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.dynamic_perception_control_schema_v1 import (  # noqa: E402
    CONTRACT_VERSION, SUITE_VERSION, canonical_hash, file_hash,
    finalize_manifest, validate_control,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa: E402
from authoritative_dataset.warp_visibility_reference_v1 import (  # noqa: E402
    classify_pair, summarize,
)
from geometry_authority.static_v1 import build_authority_artifact  # noqa: E402


CONFIG_PATH = (
    ROOT / "configs/dynamic_perception_physical_control_contract_v1.yaml"
)
DEFAULT_OUTPUT = ROOT / "data/phase8_dynamic_perception_controls_v1"
NAMESPACE = uuid.UUID("2ce11e58-c938-57af-8fca-78f481d4d138")


def atomic_json(path, value):
    path = Path(path)
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ) + "\n"
    if path.exists():
        if path.read_text() == data:
            return
        raise FileExistsError(f"refusing to overwrite different file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data)
    os.replace(temporary, path)


def save_npy(path, value):
    path = Path(path)
    array = np.asarray(value)
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        if loaded.dtype == array.dtype and np.array_equal(
            loaded, array, equal_nan=True
        ):
            return
        raise FileExistsError(f"refusing to overwrite different array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def grid(values):
    return np.asarray(list(values), dtype=np.float32)


def plane_x(x, y_min, y_max, z_min, z_max, step=.1):
    yy, zz = np.meshgrid(
        grid(np.arange(y_min+.5*step, y_max, step)),
        grid(np.arange(z_min+.5*step, z_max, step)), indexing="ij",
    )
    return np.stack(
        (
            np.full(yy.size, x+.5*step, dtype=np.float32),
            yy.ravel(), zz.ravel(),
        ),
        axis=1,
    )


def plane_y(y, x_min, x_max, z_min, z_max, step=.1):
    xx, zz = np.meshgrid(
        grid(np.arange(x_min+.5*step, x_max, step)),
        grid(np.arange(z_min+.5*step, z_max, step)), indexing="ij",
    )
    return np.stack(
        (
            xx.ravel(),
            np.full(xx.size, y+.5*step, dtype=np.float32),
            zz.ravel(),
        ),
        axis=1,
    )


def plane_z(z, x_min, x_max, y_min, y_max, step=.1):
    xx, yy = np.meshgrid(
        grid(np.arange(x_min+.5*step, x_max, step)),
        grid(np.arange(y_min+.5*step, y_max, step)), indexing="ij",
    )
    return np.stack(
        (
            xx.ravel(), yy.ravel(),
            np.full(xx.size, z+.5*step, dtype=np.float32),
        ),
        axis=1,
    )


def cylinder(x, y, radius, z_min=-2.5, z_max=2.5):
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    zz = np.arange(z_min, z_max+1e-6, .1)
    aa, heights = np.meshgrid(angles, zz, indexing="ij")
    return np.stack((
        x+radius*np.cos(aa).ravel(),
        y+radius*np.sin(aa).ravel(),
        heights.ravel(),
    ), axis=1).astype(np.float32)


def authority_cloud(name):
    background = plane_x(8.0, -11.5, 11.5, -6.5, 6.5)
    if name == "distant_plane":
        values = [background]
    elif name == "near_plane":
        values = [plane_x(4.0, -6.0, 6.0, -4.0, 4.0)]
    elif name == "sparse_natural":
        values = [
            background, cylinder(5.0, -3.2, .18),
            cylinder(6.0, 3.4, .22),
        ]
    elif name == "cave_like":
        values = [
            background, plane_y(-5.0, 1.0, 9.0, -3.5, 3.5),
            plane_y(5.0, 1.0, 9.0, -3.5, 3.5),
            plane_z(-3.5, 1.0, 9.0, -5.0, 5.0),
            plane_z(3.5, 1.0, 9.0, -5.0, 5.0),
        ]
    elif name == "forest_like":
        values = [
            background, cylinder(4.5, -3.0, .25),
            cylinder(5.5, 2.8, .28), cylinder(6.5, -4.2, .20),
        ]
    elif name == "room_wall_like":
        values = [
            plane_x(6.0, -8.0, 8.0, -5.0, 5.0),
            plane_y(-6.0, 0.0, 7.0, -4.0, 4.0),
            plane_y(6.0, 0.0, 7.0, -4.0, 4.0),
        ]
    elif name == "pillar_scene":
        values = [background, cylinder(3.5, 2.2, .35)]
    elif name == "tree_scene":
        values = [background, cylinder(4.0, -2.5, .28)]
    elif name == "disocclusion_scene":
        values = [background, cylinder(3.0, 0.0, .45)]
    elif name == "partial_occlusion_scene":
        values = [background, cylinder(1.0, .22, .18)]
    elif name == "finite_wall":
        values = [plane_x(6.0, -3.5, 3.5, -2.5, 2.5)]
    else:
        raise KeyError(name)
    return np.concatenate(values).astype("<f4")


def sensor_from_config(config):
    sensor = dict(config["sensor"])
    sensor["frame_period_ns"] = int(
        round(float(sensor.pop("frame_period_s"))*1e9)
    )
    return sensor


def ray_for_pixel(sensor, pixel):
    fx, fy, cx, cy = [float(x) for x in sensor["intrinsics"]]
    u, v = pixel
    ray = np.asarray([1.0, (u-cx)/fx, (v-cy)/fy], dtype=np.float64)
    return ray/np.linalg.norm(ray)


def rotate_yaw(vector, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(vector) @ rotation


def camera_trajectory(kind, frames, config):
    positions = np.zeros((frames, 3), dtype=np.float64)
    yaws = np.zeros(frames, dtype=np.float64)
    motion = config["camera_motion"]
    if kind in ("yaw_lateral", "yaw_forward"):
        row = motion["combined"][kind]
        yaws = np.arange(frames)*float(row["yaw_rad"])
        positions = (
            np.arange(frames)[:, None]
            * np.asarray(row["translation_m"], dtype=np.float64)
        )
    elif kind.startswith("yaw_"):
        direction = -1.0 if "left" in kind else 1.0
        level = (
            "small" if "small" in kind else
            "contract_valid_large" if "large" in kind else "medium"
        )
        yaws = np.arange(frames)*direction*float(
            motion["yaw_rad_per_frame"][level]
        )
    elif kind in ("lateral", "forward", "backward", "vertical_up"):
        key = "vertical" if kind == "vertical_up" else kind
        delta = np.asarray(
            motion["translation_m_per_frame"][key], dtype=np.float64
        )
        positions = np.arange(frames)[:, None]*delta
    elif kind == "vertical_down":
        delta = -np.asarray(
            motion["translation_m_per_frame"]["vertical"],
            dtype=np.float64,
        )
        positions = np.arange(frames)[:, None]*delta
    elif kind != "static":
        raise KeyError(kind)
    return positions, yaws


def actor_trajectory(
    sensor, camera_positions, camera_yaws, radius, distance,
    pixel, motion, speed, reference_frame=0,
):
    frames = len(camera_positions)
    reference_ray_body = ray_for_pixel(sensor, pixel)
    reference_ray_world = rotate_yaw(
        reference_ray_body, camera_yaws[reference_frame]
    )
    center = (
        camera_positions[reference_frame]
        + reference_ray_world*float(distance)
    )
    horizontal = np.asarray(
        [-reference_ray_world[1], reference_ray_world[0], 0.0]
    )
    horizontal /= max(np.linalg.norm(horizontal), 1e-12)
    if motion == "radial_approach":
        velocity = -reference_ray_world*float(speed)
    elif motion == "radial_departure":
        velocity = reference_ray_world*float(speed)
    elif motion == "pure_tangential":
        velocity = horizontal*float(speed)
    elif motion == "radial_tangential":
        velocity = (
            -.45*reference_ray_world+.8930285549745876*horizontal
        )*float(speed)
    else:
        raise KeyError(motion)
    timestamps = np.arange(frames, dtype=np.float64)*.1
    reference_time = timestamps[reference_frame]
    centers = center+(timestamps-reference_time)[:, None]*velocity
    return centers[:, None, :], velocity


def control_record(
    control_id, semantic_class, role, expected_outcome, split,
    authority, sensor_hash, renderer_hash, camera_positions, camera_yaws,
    actors, radii, seed, generator_hash, validator_hash,
    provenance_expectation, physical,
):
    timestamps = np.arange(len(camera_positions), dtype=np.float64)*.1
    actor_value = (
        None if actors is None else {
            "positions_world": np.asarray(actors).tolist(),
            "radii_m": [float(x) for x in radii],
        }
    )
    value = {
        "control_id": control_id,
        "contract_version": CONTRACT_VERSION,
        "suite_version": SUITE_VERSION,
        "semantic_class": semantic_class,
        "role": role,
        "expected_outcome": expected_outcome,
        "split": split,
        "authority": authority,
        "sensor_hash": sensor_hash,
        "renderer_hash": renderer_hash,
        "camera_trajectory": {
            "positions_world": np.asarray(camera_positions).tolist(),
            "yaws_rad": np.asarray(camera_yaws).tolist(),
            "timestamps": timestamps.tolist(),
        },
        "actor_trajectory": actor_value,
        "deterministic_seed": int(seed),
        "generator_hash": generator_hash,
        "validator_hash": validator_hash,
        "runtime_inputs": {
            "depth_file": "depth.npy",
            "camera_positions": "camera_positions.npy",
            "camera_yaws": "camera_yaws.npy",
            "intrinsics": "manifest.sensor.intrinsics",
            "timestamps": "timestamps.npy",
        },
        "offline_ground_truth": {
            "actor_id": (
                [] if actors is None else list(range(np.asarray(actors).shape[1]))
            ),
            "actor_mask": "nearest_actor_owner.npy",
            "expected_classification": semantic_class,
            "authority_correspondence": "provenance_summary.json",
        },
        "provenance_expectation": provenance_expectation,
        "physical": physical,
        "manual_depth_overwrite": False,
        "detector_output_used_for_label": False,
        "future_input_used": False,
    }
    validate_control(value)
    return value


def render_and_write(
    output, definition, backends, renderer, sensor,
    sensor_hash, generator_hash, validator_hash,
):
    control_id = definition["control_id"]
    directory = output / "controls" / control_id
    backend = backends[definition["authority_name"]]
    positions, yaws = camera_trajectory(
        definition["camera_motion"], definition.get("frames", 8),
        definition["config"],
    )
    actors = definition.get("actors")
    radii = definition.get("radii", [])
    result = renderer.render_with_actor_diagnostics(
        backend, positions, yaws,
        (
            np.asarray(actors, dtype=np.float64)
            if actors is not None else
            np.empty((len(positions), 0, 3), dtype=np.float64)
        ),
        radii,
        return_owner_map=True, return_actor_near_depth=True,
    )
    timestamps = np.arange(len(positions), dtype=np.float64)*.1
    save_npy(directory / "depth.npy", result["composed_depth"])
    save_npy(directory / "static_depth.npy", result["static_depth"])
    save_npy(directory / "camera_positions.npy", positions)
    save_npy(directory / "camera_yaws.npy", yaws)
    save_npy(directory / "timestamps.npy", timestamps)
    save_npy(directory / "nearest_actor_owner.npy",
             result["nearest_actor_owner"])
    save_npy(directory / "actor_near_depth.npy",
             result["actor_near_depth"])
    if actors is not None:
        save_npy(directory / "actor_positions.npy", actors)
    summaries = []
    for index in range(1, len(positions)):
        reference = classify_pair(
            result["static_depth"][index-1],
            result["static_depth"][index],
            positions[index-1], yaws[index-1],
            positions[index], yaws[index],
            sensor,
            definition["config"]["validation"][
                "warp_depth_consistency_tolerance_m"
            ],
        )
        summaries.append({"frame": index, **summarize(reference)})
    atomic_json(directory / "provenance_summary.json", {
        "reference_version": "warp_visibility_reference_v1",
        "pairs": summaries,
    })
    authority = {
        "map_uuid": backend.map.metadata["map_uuid"],
        "authority_hash": backend.map.metadata["artifact_manifest_hash"],
        "occupancy_hash": backend.map.metadata["occupancy_hash"],
        "raw_geometry_hash": backend.map.metadata["source_cloud_hash"],
        "scope": "development_control",
        "root": str(backend.map.root),
    }
    record = control_record(
        control_id, definition["semantic_class"], definition["role"],
        definition["expected_outcome"], definition["split"],
        authority, sensor_hash, hashlib.sha256(
            (ROOT / "authoritative_dataset/cuda_renderer_v1.py").read_bytes()
        ).hexdigest(),
        positions, yaws, actors, radii, definition["seed"],
        generator_hash, validator_hash,
        definition["provenance_expectation"],
        definition["physical"],
    )
    record["render_diagnostics"] = {
        key: np.asarray(value).tolist()
        for key, value in result.items()
        if key.startswith("per_actor_") or key == "actor_pixel_count"
    }
    record["files"] = {
        path.name: file_hash(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "control.json"
    }
    record["control_hash"] = canonical_hash(record)
    atomic_json(directory / "control.json", record)
    return record


def definitions(config, sensor, split, seed):
    frames = 8
    positions = config["physics"]["image_positions"]
    backgrounds = config["physics"]["backgrounds"]
    rows = []
    actor_specs = [
        # id, r, d, pixel key, motion, speed, background
        ("actor_min_full_r020_d180_center", .2, 1.8, "center",
         "radial_approach", .9, "distant_plane"),
        ("actor_r030_d180_center", .3, 1.8, "center",
         "radial_approach", .9, "near_plane"),
        ("actor_r020_d100_radial_departure", .2, 1.0, "center",
         "radial_departure", .9, "sparse_natural"),
        ("actor_r030_d100_radial_departure", .3, 1.0, "center",
         "radial_departure", 1.05, "cave_like"),
        ("actor_r020_d140_tangential_min", .2, 1.4, "center",
         "pure_tangential", 1.45, "forest_like"),
        ("actor_r030_d140_tangential_mid", .3, 1.4, "center",
         "pure_tangential", 1.60, "room_wall_like"),
        ("actor_r020_d140_tangential_max", .2, 1.4,
         "left_stable_interior", "pure_tangential", 1.75,
         "distant_plane"),
        ("actor_r030_d140_mixed", .3, 1.4, "right_stable_interior",
         "radial_tangential", 1.60, "near_plane"),
        ("actor_r020_upper_stable", .2, 1.8, "upper_stable_interior",
         "radial_approach", 1.05, "cave_like"),
        ("actor_r020_lower_stable", .2, 1.8, "lower_stable_interior",
         "radial_approach", 1.20, "forest_like"),
        ("actor_r020_near_left_full", .2, 1.8, "near_left_edge_full",
         "radial_approach", .9, "room_wall_like"),
        ("actor_r020_near_right_full", .2, 1.8, "near_right_edge_full",
         "radial_approach", 1.05, "sparse_natural"),
    ]
    for index, (control_id, radius, distance, pixel_name, motion,
                speed, authority_name) in enumerate(actor_specs):
        cameras, yaws = camera_trajectory("static", frames, config)
        actors, velocity = actor_trajectory(
            sensor, cameras, yaws, radius, distance,
            positions[pixel_name], motion, speed, reference_frame=0,
        )
        rows.append({
            "control_id": control_id,
            "semantic_class": "required_in_contract_positive",
            "role": "hard_positive",
            "expected_outcome": "physical_actor_present",
            "split": split, "seed": seed+index,
            "authority_name": authority_name,
            "camera_motion": "static", "frames": frames,
            "actors": actors, "radii": [radius], "config": config,
            "provenance_expectation": {
                "actor_full_projection_reference_frame": 0,
                "no_static_occlusion_reference_frame": True,
            },
            "physical": {
                "radius_m": radius,
                "requested_center_distance_m": distance,
                "reference_frame": 0,
                "motion": motion, "speed_mps": speed,
                "velocity_world_mps": velocity.tolist(),
                "image_position": pixel_name,
                "background": authority_name,
                "actor_acceleration_mps2": 0.0,
            },
        })
    cameras, yaws = camera_trajectory("static", frames, config)
    actors, velocity = actor_trajectory(
        sensor, cameras, yaws, .3, 1.8, positions["center"],
        "pure_tangential", 1.45, reference_frame=0,
    )
    rows.append({
        "control_id": "actor_partial_static_occlusion",
        "semantic_class": "bounded_latency_positive",
        "role": "partial_occlusion_positive",
        "expected_outcome": "physical_partial_static_occlusion",
        "split": split, "seed": seed+90,
        "authority_name": "partial_occlusion_scene",
        "camera_motion": "static", "frames": frames,
        "actors": actors, "radii": [.3], "config": config,
        "provenance_expectation": {
            "projected_greater_than_visible_reference_frame": True,
            "actor_remains_partially_visible_reference_frame": True,
        },
        "physical": {
            "radius_m": .3, "requested_center_distance_m": 1.8,
            "reference_frame": 0, "motion": "pure_tangential",
            "speed_mps": 1.45, "velocity_world_mps": velocity.tolist(),
            "image_position": "center",
            "background": "partial_occlusion_scene",
            "actor_acceleration_mps2": 0.0,
        },
    })
    fov_specs = [
        ("static_fov_yaw_left_wall", "distant_plane", "yaw_left_medium",
         "newly_visible_from_image_fov"),
        ("static_fov_yaw_right_wall", "distant_plane", "yaw_right_medium",
         "newly_visible_from_image_fov"),
        ("static_fov_yaw_left_small", "room_wall_like", "yaw_left_small",
         "newly_visible_from_image_fov"),
        ("static_fov_yaw_right_large", "room_wall_like", "yaw_right_large",
         "newly_visible_from_image_fov"),
        ("static_fov_vertical_up", "cave_like", "vertical_up",
         "newly_visible_from_image_fov"),
        ("static_fov_vertical_down", "cave_like", "vertical_down",
         "newly_visible_from_image_fov"),
        ("static_fov_lateral_translation", "forest_like", "lateral",
         "newly_visible_from_image_fov"),
        ("static_fov_forward_translation", "distant_plane", "forward",
         "stable_overlap"),
        ("static_fov_backward_translation", "distant_plane", "backward",
         "newly_visible_from_image_fov"),
        ("static_fov_yaw_lateral", "pillar_scene", "yaw_lateral",
         "newly_visible_from_image_fov"),
        ("static_tree_entry", "tree_scene", "yaw_right_medium",
         "newly_visible_from_image_fov"),
        ("static_pillar_entry", "pillar_scene", "yaw_left_medium",
         "newly_visible_from_image_fov"),
        ("static_wall_edge_entry", "finite_wall", "yaw_lateral",
         "max_depth_transition"),
        ("static_disocclusion", "disocclusion_scene", "lateral",
         "newly_visible_from_static_disocclusion"),
        ("max_depth_background_entry", "finite_wall", "yaw_right_large",
         "max_depth_transition"),
    ]
    offset = len(rows)
    for index, (control_id, authority_name, camera_motion, expected) in enumerate(
        fov_specs
    ):
        rows.append({
            "control_id": control_id,
            "semantic_class": "required_hard_negative",
            "role": "hard_negative",
            "expected_outcome":
                "no_measurement_no_track_birth_no_attention",
            "split": split, "seed": seed+offset+index,
            "authority_name": authority_name,
            "camera_motion": camera_motion, "frames": frames,
            "actors": None, "radii": [], "config": config,
            "provenance_expectation": {
                "required_channel": expected,
                "minimum_pixels":
                    config["validation"]["minimum_provenance_pixels"],
                "actor_count": 0,
            },
            "physical": {
                "camera_motion": camera_motion,
                "static_geometry": authority_name,
                "dynamic_gt": False,
                "manual_depth_overwrite": False,
            },
        })
    return rows


def edge_definitions(config, sensor, split, seed):
    frames = 10
    specs = [
        ("edge_actor_enter_left_r020", .2, 1.4, [-12., 45.], 1.45,
         "distant_plane", "static"),
        ("edge_actor_enter_right_r020", .2, 1.4, [172., 45.], -1.45,
         "near_plane", "static"),
        ("edge_actor_upper_r030", .3, 1.4, [80., -8.], 1.45,
         "cave_like", "static"),
        ("edge_actor_lower_r030", .3, 1.4, [80., 98.], -1.45,
         "forest_like", "static"),
        ("edge_actor_moving_camera", .2, 1.4, [-8., 45.], 1.60,
         "room_wall_like", "yaw_lateral"),
        ("edge_actor_small_to_stable", .2, 1.8, [-6., 45.], 1.45,
         "sparse_natural", "static"),
        ("edge_actor_default_to_stable", .3, 1.8, [166., 45.], -1.45,
         "distant_plane", "yaw_right_small"),
    ]
    rows = []
    for index, (control_id, radius, distance, pixel, signed_speed,
                authority_name, camera_motion) in enumerate(specs):
        cameras, yaws = camera_trajectory(camera_motion, frames, config)
        ray = ray_for_pixel(sensor, pixel)
        center = cameras[0]+rotate_yaw(ray, yaws[0])*distance
        if pixel[0] < 0 or pixel[0] > sensor["width"]-1:
            tangent = np.asarray([0.0, np.sign(signed_speed), 0.0])
        else:
            tangent = np.asarray([0.0, 0.0, np.sign(signed_speed)])
        velocity = tangent*abs(signed_speed)
        times = np.arange(frames)*.1
        actors = (center+times[:, None]*velocity)[:, None, :]
        rows.append({
            "control_id": control_id,
            "semantic_class": "bounded_latency_positive",
            "role": "paired_edge_positive",
            "expected_outcome": "measurement_after_stable_overlap_k",
            "split": split, "seed": seed+index,
            "authority_name": authority_name,
            "camera_motion": camera_motion, "frames": frames,
            "actors": actors, "radii": [radius], "config": config,
            "provenance_expectation": {
                "newly_visible_only_may_delay_birth": True,
                "stable_overlap_support_frames_k":
                    config["latency"]["causal_support_frames_k"],
                "blanket_border_crop_forbidden": True,
            },
            "physical": {
                "radius_m": radius, "requested_center_distance_m": distance,
                "reference_frame": 0, "motion": "edge_entry",
                "speed_mps": abs(signed_speed),
                "velocity_world_mps": velocity.tolist(),
                "image_entry_side": control_id.split("_")[3],
                "background": authority_name,
                "actor_acceleration_mps2": 0.0,
            },
        })
    return rows


def diagnostic_definitions(config, sensor, split, seed):
    frames = 6
    specs = [
        ("diagnostic_radius_r010", .1, 1.4, 1.0),
        ("diagnostic_distance_d400", .2, 4.0, 1.0),
        ("diagnostic_legacy_equivalent_r010_d400", .1, 4.0, 1.0),
        ("diagnostic_speed_above_contract", .2, 1.4, 2.5),
    ]
    rows = []
    for index, (control_id, radius, distance, speed) in enumerate(specs):
        cameras, yaws = camera_trajectory("static", frames, config)
        actors, velocity = actor_trajectory(
            sensor, cameras, yaws, radius, distance,
            [80, 45], "radial_approach", speed, reference_frame=0,
        )
        rows.append({
            "control_id": control_id,
            "semantic_class": "diagnostic_out_of_contract",
            "role": "diagnostic_only",
            "expected_outcome": "report_only_excluded_from_hard_gate",
            "split": split, "seed": seed+index,
            "authority_name": "distant_plane",
            "camera_motion": "static", "frames": frames,
            "actors": actors, "radii": [radius], "config": config,
            "provenance_expectation": {"hard_gate_eligible": False},
            "physical": {
                "radius_m": radius,
                "requested_center_distance_m": distance,
                "reference_frame": 0, "motion": "radial_approach",
                "speed_mps": speed,
                "velocity_world_mps": velocity.tolist(),
                "actor_acceleration_mps2": 0.0,
            },
        })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = DEFAULT_OUTPUT.resolve()
    if output != allowed and not (
        args.smoke and str(output).startswith("/tmp/")
    ):
        raise RuntimeError(f"CCR1 output must be exactly {allowed}")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["contract_version"] != CONTRACT_VERSION:
        raise RuntimeError("control config version mismatch")
    sensor = sensor_from_config(config)
    generator_hash = file_hash(Path(__file__))
    validator_path = (
        ROOT / "tools/validate_dynamic_perception_controls_v1.py"
    )
    if not validator_path.is_file():
        raise RuntimeError("independent validator source must exist first")
    validator_hash = file_hash(validator_path)
    config_hash = file_hash(CONFIG_PATH)
    sensor_hash = canonical_hash(sensor)
    authority_names = [
        "distant_plane", "near_plane", "sparse_natural", "cave_like",
        "forest_like", "room_wall_like", "pillar_scene", "tree_scene",
        "disocclusion_scene", "partial_occlusion_scene", "finite_wall",
    ]
    authority_roots = {}
    for index, name in enumerate(authority_names):
        map_uuid = str(uuid.uuid5(NAMESPACE, f"authority:{name}"))
        root = output / "authority" / map_uuid
        metadata = build_authority_artifact(
            root, authority_cloud(name), map_uuid=map_uuid,
            generator_seed=840000+index, map_id=f"ccr1-{name}",
            generator_source_hash=generator_hash,
            generator_config_hash=config_hash,
            parent_git_commit="ccr1-development-control",
            map_namespace="dynamic_perception_control_authority_v1",
            bounds_min=[-3.0, -12.0, -7.0],
            bounds_max=[12.0, 12.0, 7.0],
        )
        authority_roots[name] = {
            "root": root, "metadata": metadata,
        }
    backends = {
        name: ExactAuthorityBVH(value["root"])
        for name, value in authority_roots.items()
    }
    renderer = CudaAuthorityRenderer(sensor, f"cuda:{args.device}")
    rows = []
    split_counts = {}
    started = time.perf_counter()
    split_rows = (
        ("development", config["split"]["development_seed"]),
    ) if args.smoke else (
        ("development", config["split"]["development_seed"]),
        ("sealed_control_holdout", config["split"]["sealed_holdout_seed"]),
    )
    for split, seed in split_rows:
        controls = (
            definitions(config, sensor, split, seed)
            + edge_definitions(config, sensor, split, seed+100)
            + diagnostic_definitions(config, sensor, split, seed+200)
        )
        # Holdout has distinct IDs, seeds, trajectories (mirrored yaw/velocity)
        # and the same physical contract. No detector is run.
        if split == "sealed_control_holdout":
            for definition in controls:
                definition["control_id"] += "_holdout"
                if definition["actors"] is not None:
                    definition["actors"] = np.asarray(
                        definition["actors"]
                    ).copy()
                    definition["actors"][..., 1] *= -1
                    velocity = definition["physical"].get(
                        "velocity_world_mps"
                    )
                    if velocity is not None:
                        velocity = list(velocity)
                        velocity[1] *= -1
                        definition["physical"][
                            "velocity_world_mps"
                        ] = velocity
                definition["camera_motion"] = {
                    "yaw_left_medium": "yaw_right_medium",
                    "yaw_right_medium": "yaw_left_medium",
                    "yaw_left_small": "yaw_right_small",
                    "yaw_right_large": "yaw_left_large",
                }.get(definition["camera_motion"],
                      definition["camera_motion"])
        split_counts[split] = len(controls)
        for definition in controls:
            rows.append(render_and_write(
                output, definition, backends, renderer, sensor,
                sensor_hash, generator_hash, validator_hash,
            ))
            print(json.dumps({
                "status": "RENDERED",
                "control_id": definition["control_id"],
                "split": split,
            }))
    infeasible = [
        {
            "radius_m": radius, "distance_m": .6,
            "minimum_center_clearance_m": radius+.3+.1,
            "classification": "physically_infeasible",
            "reason": "actor-UAV clearance contract",
        }
        for radius in (.2, .3)
    ]
    manifest = finalize_manifest({
        "contract_version": CONTRACT_VERSION,
        "suite_version": SUITE_VERSION,
        "scope": config["scope"],
        "config_hash": config_hash,
        "sensor": sensor,
        "sensor_hash": sensor_hash,
        "renderer_version": RENDERER_VERSION,
        "renderer_hash": hashlib.sha256(
            (ROOT / "authoritative_dataset/cuda_renderer_v1.py").read_bytes()
        ).hexdigest(),
        "generator_hash": generator_hash,
        "validator_hash": validator_hash,
        "authority": {
            name: {
                "root": str(value["root"]),
                "map_uuid": value["metadata"]["map_uuid"],
                "authority_hash":
                    value["metadata"]["artifact_manifest_hash"],
                "occupancy_hash": value["metadata"]["occupancy_hash"],
                "raw_geometry_hash":
                    value["metadata"]["source_cloud_hash"],
            }
            for name, value in authority_roots.items()
        },
        "control_ids": [row["control_id"] for row in rows],
        "split_counts": split_counts,
        "physically_infeasible_matrix_entries": infeasible,
        "manual_depth_overwrite": False,
        "detector_executed": False,
        "architecture_executed": False,
        "tf1_holdout_accessed": False,
        "test_accessed": False,
        "blind_accessed": False,
    })
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "generation_complete.json", {
        "status": "PASS", "manifest_hash": manifest["manifest_hash"],
        "control_count": len(rows),
        "elapsed_seconds": time.perf_counter()-started,
    })
    print(json.dumps({
        "status": "PASS", "control_count": len(rows),
        "manifest_hash": manifest["manifest_hash"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
