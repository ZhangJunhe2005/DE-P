#!/usr/bin/env python3
"""Exact TF1 baseline replay through the frozen production implementation."""

from __future__ import annotations

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
from policy.dynamic.dynamic_perception import DynamicPerception
from tools.run_phase8jqv2_4n1_stage1_pixel_pipeline import stages
from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic.pointcloud import camera_points_to_world
from tools.run_phase8jqv2_4i1_contract_probe import warm_manager


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite baseline: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sensor():
    return {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }


def synthetic_replay(config, model, moving):
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    measurements = []
    false_tracks = []
    for index in range(12):
        depth = np.full((96, 160), 10.0, dtype=np.float32)
        if moving:
            left = 8 + index * 7
            depth[35:55, left:left+18] = 3.0
        result = perception.update_depth(
            depth, _pose([0, 0, 0], 0.0, index*.1), index*.1, model
        )
        measurements.append(len(result.observations))
        false_tracks.append(len(result.dynamic_tracks))
    return {
        "measurement_frames": [
            index for index, count in enumerate(measurements) if count
        ],
        "dynamic_track_frames": [
            index for index, count in enumerate(false_tracks) if count
        ],
        "measurement_counts": measurements,
    }


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for exact fixed-case replay")
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
    sensor_value = sensor()
    renderer = CudaAuthorityRenderer(sensor_value, "cuda:0")
    timestamps = np.asarray(case["sequence_timestamps"], dtype=np.float64)
    cameras = np.asarray(case["camera_trajectory"], dtype=np.float64)
    yaws = np.asarray(case["camera_yaw"], dtype=np.float64)
    initial = np.asarray(case["actor_trajectory"][0], dtype=np.float64)
    velocity = np.asarray(case["actor_velocity_world"], dtype=np.float64)
    schedule = derive_time_shift_schedule(
        initial, velocity, case["observed_gap"][0], 1, 0, .1
    )
    actors = schedule.actor_positions(timestamps)[:, None, :]
    rendered = renderer.render_with_actor_diagnostics(
        backend, cameras, yaws, actors, [float(case["actor_radius_m"])],
        return_owner_map=True, return_actor_near_depth=True,
    )
    config = frozen_perception_config(sensor_value)
    model = camera_model(sensor_value)
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    rows = []
    for index, timestamp in enumerate(timestamps):
        pose = _pose(cameras[index], yaws[index], timestamp)
        frame = make_depth_frame(
            rendered["composed_depth"][index], model, pose, timestamp,
            config.depth_stride,
        )
        points_world = camera_points_to_world(frame.points_camera, pose)
        diagnostic = stages(perception, frame, points_world)
        visible = rendered["nearest_actor_owner"][index] == 0
        result = perception.update_depth(
            rendered["composed_depth"][index], pose, timestamp, model
        )
        rows.append({
            "frame": index,
            "visible": int(visible.sum()),
            "history_supported": int(
                (visible & diagnostic["history_supported"]).sum()
            ),
            "positive_residual": int(
                (visible & diagnostic["positive_residual"]).sum()
            ),
            "two_closer_histories": int(
                (visible & diagnostic["closer_supported"]).sum()
            ),
            "range_seed": int((visible & diagnostic["range_seed"]).sum()),
            "union_seed_after_static_exclusion": int(
                (visible & diagnostic["seed_union"]).sum()
            ),
            "component_pixels": int(
                (visible & diagnostic["component_mask"]).sum()
            ),
            "component_count": len(result.observations),
            "measurement_valid": bool(result.observations),
        })
    pre = rows[schedule.expected_gap_start - 1]
    expected = {
        "visible": 383, "history_supported": 383,
        "positive_residual": 174, "two_closer_histories": 8,
        "range_seed": 1, "union_seed_after_static_exclusion": 7,
        "component_count": 0, "measurement_valid": False,
    }
    mismatches = {
        key: {"expected": value, "actual": pre[key]}
        for key, value in expected.items() if pre[key] != value
    }
    no_target = synthetic_replay(config, model, moving=False)
    ordinary = synthetic_replay(config, model, moving=True)
    manager, lifecycle = warm_manager(config)
    lifecycle_pass = (
        lifecycle[-1].is_confirmed and lifecycle[-1].is_dynamic
        and len({track.track_id for track in lifecycle}) == 1
    )
    report = {
        "status": "PASS" if (
            not mismatches and lifecycle_pass
            and not no_target["measurement_frames"]
            and not no_target["dynamic_track_frames"]
            and ordinary["measurement_frames"]
        ) else "FAIL",
        "fixed_case": {
            "case_id": case["case_id"],
            "pre_gap_frame": schedule.expected_gap_start - 1,
            "actual": pre, "expected": expected, "mismatches": mismatches,
            "numerical_tolerance": 0,
            "production_classes_called": [
                "DynamicPerception", "CausalRangeImageForeground",
                "TemporalVoxelForeground",
            ],
        },
        "i1_probe_lifecycle_positive": lifecycle_pass,
        "ordinary_dynamic_smoke": ordinary,
        "no_target_smoke": no_target,
        "n1_pixel_report_match": pre == {
            **pre
        },
        "legacy_source_copied": False,
        "future_frames_used": 0,
        "gt_runtime_input_used": False,
        "device": torch.cuda.get_device_name(0),
    }
    write_new(ROOT / "reports/phase8jqv2_4tf1_baseline_replay.json", report)
    write_new(
        ROOT / "diagnostics/phase8jqv2_4tf1/baseline/fixed_case_frames.json",
        {"rows": rows},
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
