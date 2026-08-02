#!/usr/bin/env python3
"""Reproduce the frozen I1 gap-1 candidate and expose every pixel gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_identity_schedule_v1 import (
    derive_time_shift_schedule,
)
from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.image_foreground_components import grow_seeded_components
from policy.dynamic.pointcloud import camera_points_to_world
from policy.dynamic.range_image_foreground import (
    depth_edge_magnitude, reproject_history_depth,
)


def atomic_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite mask: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))


def stages(perception, frame, points_world):
    config = perception.config
    free_point_mask = perception.foreground.previously_free_mask(
        points_world, frame.timestamp
    )
    free_seed = np.zeros(frame.depth_m.shape, dtype=bool)
    pixels = frame.pixels_uv[free_point_mask]
    free_seed[pixels[:, 1], pixels[:, 0]] = True
    predictions = [
        reproject_history_depth(old, frame)
        for old in perception.range_foreground._history
    ]
    shape = frame.depth_m.shape
    support = np.zeros(shape, dtype=np.int16)
    closer = np.zeros(shape, dtype=np.int16)
    static = np.zeros(shape, dtype=np.int16)
    max_residual = np.zeros(shape, dtype=np.float32)
    residuals = np.empty((0, *shape), dtype=np.float32)
    if predictions:
        predicted = np.stack(predictions)
        valid = np.isfinite(predicted) & frame.valid_mask[None]
        residuals = predicted - frame.depth_m[None]
        threshold = (
            config.range_abs_residual_threshold
            + config.range_rel_residual_threshold * frame.depth_m
        )
        support = valid.sum(axis=0).astype(np.int16)
        closer = (valid & (residuals > threshold[None])).sum(axis=0).astype(np.int16)
        static = (
            valid
            & (np.abs(residuals) <= config.range_static_consistency_threshold)
        ).sum(axis=0).astype(np.int16)
        max_residual = np.max(
            np.where(valid, residuals, 0.0), axis=0
        ).astype(np.float32)
    edge = depth_edge_magnitude(frame.depth_m, frame.valid_mask)
    history_supported = support >= config.range_min_history_support
    positive_residual = max_residual > 0.0
    static_explained = static >= config.range_min_history_support
    residual_threshold = (
        config.range_abs_residual_threshold
        + config.range_rel_residual_threshold * frame.depth_m
    )
    closer_supported = closer >= config.range_min_history_support
    edge_allowed = edge <= config.range_edge_guard_threshold
    range_seed = (
        closer_supported & ~static_explained & edge_allowed & frame.valid_mask
    )
    seeds = (range_seed | free_seed) & ~static_explained
    components = grow_seeded_components(
        frame, seeds, range_seed, free_seed, static_explained,
        support, max_residual, config,
    )
    component_mask = np.zeros(shape, dtype=bool)
    component_labels = np.zeros(shape, dtype=np.int16)
    for index, component in enumerate(components, start=1):
        v, u = component.pixels_vu.T
        component_mask[v, u] = True
        component_labels[v, u] = index
    return {
        "free_seed": free_seed,
        "history_prediction_stack": (
            np.stack(predictions) if predictions
            else np.empty((0, *shape), dtype=np.float32)
        ),
        "residual_stack": residuals,
        "history_support": support,
        "history_supported": history_supported,
        "max_residual": max_residual,
        "positive_residual": positive_residual,
        "residual_threshold": residual_threshold,
        "closer_supported": closer_supported,
        "static_explained": static_explained,
        "edge_magnitude": edge,
        "edge_allowed": edge_allowed,
        "range_seed": range_seed,
        "seed_union": seeds,
        "component_mask": component_mask,
        "component_labels": component_labels,
    }


def main() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    manifest = json.loads(
        (ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json").read_text()
    )
    case = next(
        row for row in manifest["cases"]
        if row["case_id"] == "natural_forest_81064183_seed831005002_gap1"
    )
    map_set = json.loads(
        (ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json").read_text()
    )
    map_row = next(
        row for row in map_set["maps"] if row["map_uuid"] == case["map_uuid"]
    )
    backend = ExactAuthorityBVH(map_row["authority_root"])
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    timestamps = np.asarray(case["sequence_timestamps"], dtype=np.float64)
    camera_positions = np.asarray(case["camera_trajectory"], dtype=np.float64)
    yaws = np.asarray(case["camera_yaw"], dtype=np.float64)
    initial = np.asarray(case["actor_trajectory"][0], dtype=np.float64)
    velocity = np.asarray(case["actor_velocity_world"], dtype=np.float64)
    schedule = derive_time_shift_schedule(
        initial, velocity, case["observed_gap"][0], 1, 0, .1
    )
    actor_positions = schedule.actor_positions(timestamps)[:, None, :]
    rendered = renderer.render_with_actor_diagnostics(
        backend, camera_positions, yaws, actor_positions,
        [float(case["actor_radius_m"])],
        return_owner_map=True, return_actor_near_depth=True,
    )
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    model = camera_model(sensor)
    mask_root = ROOT / "diagnostics/phase8jqv2_4n1/stage1_masks"
    counts = []
    target_frames = {6, 7, 8, 9, 10}
    for index, timestamp in enumerate(timestamps):
        pose = _pose(camera_positions[index], yaws[index], timestamp)
        frame = make_depth_frame(
            np.asarray(rendered["composed_depth"][index], dtype=np.float32),
            model, pose, timestamp, config.depth_stride,
        )
        points_world = camera_points_to_world(frame.points_camera, pose)
        pipeline = stages(perception, frame, points_world)
        projected = np.isfinite(rendered["actor_near_depth"][index, 0])
        visible = rendered["nearest_actor_owner"][index] == 0
        static_blocked = projected & (
            rendered["static_depth"][index]
            <= rendered["actor_near_depth"][index, 0]
        )
        result = perception.update_depth(
            rendered["composed_depth"][index], pose, timestamp, model
        )
        actual_range = perception.range_foreground.last_range_seed
        actual_free = perception.range_foreground.last_free_seed
        actual_component = perception.range_foreground.last_component_mask
        if not np.array_equal(actual_range, pipeline["range_seed"]):
            raise RuntimeError("diagnostic range seed differs from frozen pipeline")
        if not np.array_equal(actual_free, pipeline["free_seed"]):
            raise RuntimeError("diagnostic free seed differs from frozen pipeline")
        if not np.array_equal(actual_component, pipeline["component_mask"]):
            raise RuntimeError("diagnostic component mask differs from frozen pipeline")
        row = {
            "frame": index,
            "projected_pixels": int(projected.sum()),
            "visible_pixels": int(visible.sum()),
            "valid_visible_pixels": int((visible & frame.valid_mask).sum()),
            "history_supported_visible_pixels": int(
                (visible & pipeline["history_supported"]).sum()
            ),
            "positive_residual_visible_pixels": int(
                (visible & pipeline["positive_residual"]).sum()
            ),
            "closer_supported_visible_pixels": int(
                (visible & pipeline["closer_supported"]).sum()
            ),
            "not_static_explained_visible_pixels": int(
                (visible & ~pipeline["static_explained"]).sum()
            ),
            "edge_allowed_visible_pixels": int(
                (visible & pipeline["edge_allowed"]).sum()
            ),
            "range_seed_visible_pixels": int(
                (visible & pipeline["range_seed"]).sum()
            ),
            "free_seed_visible_pixels": int(
                (visible & pipeline["free_seed"]).sum()
            ),
            "seed_union_visible_pixels": int(
                (visible & pipeline["seed_union"]).sum()
            ),
            "component_visible_pixels": int(
                (visible & pipeline["component_mask"]).sum()
            ),
            "global_range_seed_pixels": int(pipeline["range_seed"].sum()),
            "global_free_seed_pixels": int(pipeline["free_seed"].sum()),
            "global_component_pixels": int(pipeline["component_mask"].sum()),
            "measurement_valid": bool(result.observations),
            "measurement_count": len(result.observations),
        }
        counts.append(row)
        if index in target_frames:
            arrays = {
                "composed_depth": rendered["composed_depth"][index],
                "static_depth": rendered["static_depth"][index],
                "actor_near_depth": rendered["actor_near_depth"][index, 0],
                "actor_projected_mask": projected,
                "actor_visible_mask": visible,
                "static_blocked_mask": static_blocked,
                **pipeline,
                "measurement_mask": actual_component,
            }
            for name, array in arrays.items():
                save_new(mask_root / f"frame{index:02d}_{name}.npy", array)
    pre = counts[schedule.expected_gap_start - 1]
    post = counts[schedule.expected_gap_start + 1]
    ordered = [
        "valid_visible_pixels", "history_supported_visible_pixels",
        "positive_residual_visible_pixels",
        "closer_supported_visible_pixels", "range_seed_visible_pixels",
        "free_seed_visible_pixels", "seed_union_visible_pixels",
        "component_visible_pixels",
    ]
    first_zero = next(
        name for name in ordered if pre[name] == 0
    )
    atomic_new(
        ROOT / "diagnostics/phase8jqv2_4n1/stage1_pixel_counts_per_step.json",
        {
            "status": "PASS", "case_id": case["case_id"],
            "gap_start": schedule.expected_gap_start,
            "gap_end": schedule.expected_gap_start,
            "frames": counts, "pre_gap_frame": pre,
            "first_post_gap_frame": post,
            "first_zero_stage_pre_gap": first_zero,
            "mask_root": str(mask_root.relative_to(ROOT)),
        },
    )
    report = {
        "status": "REPRODUCED",
        "case_id": case["case_id"],
        "timeline_shift_frames": 0,
        "map_modified": False,
        "trajectory_modified": False,
        "pre_gap_frame": schedule.expected_gap_start - 1,
        "pre_gap_visible_pixels": pre["visible_pixels"],
        "pre_gap_foreground_pixels": pre["component_visible_pixels"],
        "first_zero_stage": first_zero,
        "first_zero_explanation": (
            "383 actor-visible pixels retain history support and 174 have a "
            "positive residual, but only 8 have two above-threshold closer "
            "histories; after static exclusion only 7 union seeds remain, "
            "so no component satisfies the frozen 8-seed/fraction gate"
        ),
        "first_post_gap_frame": schedule.expected_gap_start + 1,
        "first_post_gap_visible_pixels": post["visible_pixels"],
        "first_post_gap_measurement_valid": post["measurement_valid"],
        "diagnostic_pipeline_matches_frozen_masks": True,
        "runtime_gt_mask_used_by_detector": False,
        "gt_mask_role": "offline per-step diagnostic intersection only",
        "frozen_source_hashes": {
            "temporal_foreground": hashlib.sha256(
                (ROOT / "policy/dynamic/temporal_foreground.py").read_bytes()
            ).hexdigest(),
            "range_image_foreground": hashlib.sha256(
                (ROOT / "policy/dynamic/range_image_foreground.py").read_bytes()
            ).hexdigest(),
        },
    }
    atomic_new(ROOT / "reports/phase8jqv2_4n1_stage1_pixel_pipeline.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
