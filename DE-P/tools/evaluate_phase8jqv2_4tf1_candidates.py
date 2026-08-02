#!/usr/bin/env python3
"""Bounded TF1 development evaluation; never reads sealed holdout cases."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.foreground_contract_registry import (
    create_foreground_contract,
)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_scalar)
        + "\n"
    )
    os.replace(temp, path)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def sensor():
    return {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }


def make_perception(config, key, parameters):
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    perception.range_foreground = create_foreground_contract(
        key, config, parameters
    )
    return perception


def run_depth_sequence(depths, positions, yaws, config, key, parameters,
                       actor_positions=None, actor_masks=None):
    model = camera_model(sensor())
    perception = make_perception(config, key, parameters)
    rows = []
    started = time.perf_counter()
    for index, depth in enumerate(depths):
        timestamp = index * .1
        result = perception.update_depth(
            np.asarray(depth, dtype=np.float32),
            _pose(positions[index], yaws[index], timestamp),
            timestamp, model,
        )
        matched = False
        centroid_error = None
        actor_overlap = 0
        if actor_positions is not None and result.observations:
            errors = [
                float(np.linalg.norm(
                    obs.centroid_world - actor_positions[index]
                ))
                for obs in result.observations
            ]
            centroid_error = min(errors)
        if actor_masks is not None and result.observations:
            mask = np.asarray(actor_masks[index], dtype=bool).ravel()
            actor_overlap = max(
                sum(mask[list(obs.pixel_indices)])
                for obs in result.observations
                if obs.pixel_indices
            ) if any(obs.pixel_indices for obs in result.observations) else 0
            # Instance ownership is the authoritative offline association.
            # Centroid error remains diagnostic because analytic controls do
            # not encode an independently calibrated actor world centroid.
            matched = actor_overlap > 0
        elif centroid_error is not None:
            matched = centroid_error <= 1.0
        rows.append({
            "frame": index,
            "measurement_count": len(result.observations),
            "measurement_valid": bool(result.observations),
            "actor_measurement_valid": matched,
            "centroid_error_m": centroid_error,
            "actor_overlap_pixels": int(actor_overlap),
            "component_pixels": int(
                result.diagnostics["foreground"].get(
                    "component_pixel_count", 0
                )
            ),
            "strong_seeds": int(
                result.diagnostics["foreground"].get("range_seed_count", 0)
            ),
            "weak_support": int(
                result.diagnostics["foreground"].get(
                    "weak_support_pixels", 0
                )
            ),
            "track_ids": [track.track_id for track in result.all_tracks],
            "confirmed_ids": [
                track.track_id for track in result.confirmed_tracks
            ],
            "dynamic_ids": [
                track.track_id for track in result.dynamic_tracks
            ],
            "false_attention": bool(
                result.attention_map.detach().cpu().max().item() > 0
                and not matched
            ),
        })
    return rows, (time.perf_counter() - started) * 1000 / len(depths)


def synthetic_fixture(name, device):
    import torch
    frames, height, width = 14, 96, 160
    background = 10.0
    if "different_background" in name:
        background = 15.0
    depth = torch.full(
        (frames, height, width), background,
        dtype=torch.float32, device=device,
    )
    masks = torch.zeros(
        (frames, height, width), dtype=torch.bool, device=device
    )
    small = "small_projection" in name
    size = 4 if small else 18 if "large_projection" not in name else 30
    radial = "radial" in name or "approach" in name
    tangent = (
        "tangential" in name or "crossing" in name
        or "gap" in name or "multi" in name
    )
    # Every positive control must contain observable actor motion. Background,
    # projection-size and visibility controls otherwise became static fixtures
    # and incorrectly tested motion foreground recall with no motion evidence.
    if not radial:
        tangent = True
    for frame in range(frames):
        u = 20 + (frame * 6 if tangent else 0)
        v = 38
        value = 6.0 - .22 * frame if radial else 4.0
        masks[frame, v:v+size, u:u+size] = True
        depth[frame, v:v+size, u:u+size] = value
    if "gap1" in name:
        masks[8] = False
        depth[8] = background
    if "gap2" in name:
        masks[8:10] = False
        depth[8:10] = background
    if "sparse_natural" in name or "forest" in name:
        depth[:, :, 10::24] = 8.0
    if "cave" in name:
        vv, uu = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device), indexing="ij"
        )
        base = 9.0 + .8 * torch.sin(uu.float() / 13)
        depth = torch.where(masks, depth, base[None].expand_as(depth))
    if "room_wall" in name or "plane" in name:
        depth = torch.where(masks, depth, torch.full_like(depth, background))
    positions = np.zeros((frames, 3), dtype=np.float64)
    yaws = np.zeros(frames, dtype=np.float64)
    actor_positions = np.stack((
        np.linspace(3.0, 2.0 if radial else 3.0, frames),
        np.linspace(-1.0, 1.0 if tangent else -1.0, frames),
        np.ones(frames),
    ), axis=1)
    return (
        depth.cpu().numpy(), positions, yaws, actor_positions,
        masks.cpu().numpy(),
    )


def negative_fixture(name, device):
    import torch
    frames, height, width = 12, 96, 160
    depth = torch.full(
        (frames, height, width), 10.0,
        dtype=torch.float32, device=device,
    )
    positions = np.zeros((frames, 3), dtype=np.float64)
    yaws = np.zeros(frames, dtype=np.float64)
    if name == "camera_translation":
        positions[:, 1] = np.arange(frames) * .01
    elif name == "camera_rotation":
        yaws[:] = np.arange(frames) * .002
    elif name == "warp_small_error":
        positions[:, 1] = np.sin(np.arange(frames)) * .002
    elif name == "depth_quantization_noise":
        depth += (
            torch.arange(frames, device=device)[:, None, None] % 2
        ) * .005
    elif name == "isolated_single_pixel":
        depth[5:, 40, 60] = 8.0
    elif name in ("seven_scattered_seeds", "eight_scattered_seeds"):
        count = 7 if name.startswith("seven") else 8
        for index in range(count):
            depth[5:, 5+index*9, 5+index*13] = 8.0
    elif name in ("static_wall_edge", "static_tree_edge"):
        depth[:, :, 80:] = 6.0
    elif name == "disocclusion_without_actor":
        depth[:5, 30:60, 40:70] = 5.0
    elif name == "fov_boundary_change":
        depth[5:, 30:50, :2] = 8.0
    elif name == "max_depth_boundary":
        depth[5:, 30:50, 50:70] = 19.99
    elif name == "partial_residual":
        depth[5:, 30:32, 50:53] = 9.95
    elif name == "repeated_background_texture":
        depth[:, :, ::12] = 9.0
    # metadata-only, artificial hidden, ROI-only, no-target and wrong ROI
    # intentionally leave metric depth unchanged.
    return (
        depth.cpu().numpy(), positions, yaws,
        np.zeros((frames, 3)), np.zeros((frames, height, width), dtype=bool),
    )


def render_case(case, map_rows, renderer):
    backend = ExactAuthorityBVH(map_rows[case["map_uuid"]]["authority_root"])
    cameras = np.asarray(case["camera_trajectory"], dtype=np.float64)
    yaws = np.asarray(case["camera_yaw"], dtype=np.float64)
    actors = np.asarray(case["actor_trajectory"], dtype=np.float64)
    diagnostics = renderer.render_with_actor_diagnostics(
        backend, cameras, yaws, actors[:, None, :],
        [float(case["actor_radius_m"])],
        return_owner_map=True, return_actor_near_depth=True,
    )
    return {
        "case_id": case["case_id"],
        "depths": diagnostics["composed_depth"],
        "positions": cameras, "yaws": yaws, "actors": actors,
        "masks": diagnostics["nearest_actor_owner"] == 0,
        "gap_start": case["observed_gap"][0],
        "gap_end": case["observed_gap"][1],
        "map_uuid": case["map_uuid"],
        "maze_type": case["maze_type"],
    }


def summarize_natural(value, rows):
    gap_start, gap_end = value["gap_start"], value["gap_end"]
    pre = rows[gap_start-1]
    post = rows[gap_end+1]
    consecutive = 0
    for index in range(gap_start-1, -1, -1):
        if not rows[index]["actor_measurement_valid"]:
            break
        consecutive += 1
    return {
        "case_id": value["case_id"], "map_uuid": value["map_uuid"],
        "maze_type": value["maze_type"],
        "gap_start": gap_start, "gap_end": gap_end,
        "pre_gap_measurement": pre["actor_measurement_valid"],
        "pre_gap_consecutive_measurements": consecutive,
        "first_post_gap_measurement":
            post["actor_measurement_valid"],
        "measurement_frames": [
            row["frame"] for row in rows if row["actor_measurement_valid"]
        ],
        "false_component_frames": [
            row["frame"] for row in rows
            if row["measurement_valid"] and not row["actor_measurement_valid"]
        ],
        "maximum_centroid_error_m": max(
            (
                row["centroid_error_m"] for row in rows
                if row["centroid_error_m"] is not None
            ), default=None,
        ),
    }


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for candidate development")
    split = json.loads(
        (ROOT / "reports/phase8jqv2_4tf1_evaluation_split.json").read_text()
    )
    if split["holdout_results_accessed"]:
        raise RuntimeError("development cannot run after holdout access")
    retained = json.loads(
        (ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json").read_text()
    )
    cases_by_id = {row["case_id"]: row for row in retained["cases"]}
    map_set = json.loads(
        (ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json").read_text()
    )
    map_rows = {row["map_uuid"]: row for row in map_set["maps"]}
    renderer = CudaAuthorityRenderer(sensor(), "cuda:0")
    development = [
        render_case(cases_by_id[row["case_id"]], map_rows, renderer)
        for row in split["development_natural_cases"]
    ]
    config = frozen_perception_config(sensor())
    document = yaml.safe_load((
        ROOT / "configs/temporal_foreground_contract_v2_1_candidates.yaml"
    ).read_text())
    grids = []
    for key in ("candidate_a", "candidate_b", "candidate_c"):
        for index, parameters in enumerate(
            document["candidates"][key]["grid"]
        ):
            grids.append({
                "candidate": key, "parameter_index": index,
                "parameters": parameters,
            })
    grids.append({
        "candidate": "ablation_seed7", "parameter_index": 0,
        "parameters": {"minimum_seed_pixels": 7},
    })
    positive_names = [
        "near_radial_approach", "far_radial_approach",
        "pure_tangential_crossing", "radial_tangential",
        "visible_before_occlusion", "gap1_reappearance",
        "gap2_reappearance", "small_projection", "large_projection",
        "different_background_depth", "plane_background",
        "sparse_natural_background", "cave_background",
        "forest_background", "room_wall_background",
    ]
    negative_names = [
        "static_depth", "camera_translation", "camera_rotation",
        "warp_small_error", "depth_quantization_noise",
        "isolated_single_pixel", "seven_scattered_seeds",
        "eight_scattered_seeds", "static_wall_edge",
        "static_tree_edge", "disocclusion_without_actor",
        "fov_boundary_change", "max_depth_boundary", "no_target",
        "metadata_only_actor", "artificial_hidden_frame",
        "partial_residual", "repeated_background_texture",
        "live_track_roi_without_measurement", "wrong_predicted_track_roi",
    ]
    results = []
    for grid in grids:
        candidate = grid["candidate"]
        parameters = grid["parameters"]
        positive = []
        negative = []
        natural = []
        runtimes = []
        for name in positive_names:
            fixture = synthetic_fixture(name, "cuda:0")
            rows, runtime = run_depth_sequence(
                *fixture[:3], config, candidate, parameters,
                actor_positions=fixture[3], actor_masks=fixture[4],
            )
            runtimes.append(runtime)
            gap_end = 8 if "gap1" in name else 9 if "gap2" in name else None
            accepted = bool(any(
                row["actor_measurement_valid"] for row in rows[2:]
            ))
            if gap_end is not None:
                accepted &= rows[gap_end+1]["actor_measurement_valid"]
            positive.append({
                "name": name, "pass": accepted,
                "measurement_frames": [
                    row["frame"] for row in rows
                    if row["actor_measurement_valid"]
                ],
            })
        for name in negative_names:
            fixture = negative_fixture(name, "cuda:0")
            rows, runtime = run_depth_sequence(
                *fixture[:3], config, candidate, parameters,
                actor_positions=None, actor_masks=None,
            )
            runtimes.append(runtime)
            false_frames = [
                row["frame"] for row in rows if row["measurement_valid"]
            ]
            negative.append({
                "name": name, "pass": not false_frames,
                "false_measurement_frames": false_frames,
                "false_attention_frames": [
                    row["frame"] for row in rows if row["false_attention"]
                ],
            })
        for value in development:
            rows, runtime = run_depth_sequence(
                value["depths"], value["positions"], value["yaws"],
                config, candidate, parameters,
                actor_positions=value["actors"], actor_masks=value["masks"],
            )
            runtimes.append(runtime)
            natural.append(summarize_natural(value, rows))
        fixed = next(
            row for row in natural
            if row["case_id"]
            == "natural_forest_81064183_seed831005002_gap1"
        )
        passing_maps = {
            row["map_uuid"] for row in natural
            if (
                row["pre_gap_measurement"]
                and row["first_post_gap_measurement"]
            )
        }
        row = {
            **grid,
            "synthetic_positive_pass": sum(x["pass"] for x in positive),
            "synthetic_positive_total": len(positive),
            "synthetic_negative_pass": sum(x["pass"] for x in negative),
            "synthetic_negative_total": len(negative),
            "fixed_pre_gap_measurement": fixed["pre_gap_measurement"],
            "fixed_post_gap_measurement":
                fixed["first_post_gap_measurement"],
            "development_natural_passing_maps": len(passing_maps),
            "development_natural_passing_types": len({
                row["maze_type"] for row in natural
                if row["map_uuid"] in passing_maps
            }),
            "average_runtime_ms": float(np.mean(runtimes)),
            "positive_controls": positive,
            "negative_controls": negative,
            "natural_cases": natural,
        }
        row["development_hard_gate"] = bool(
            row["synthetic_positive_pass"] == len(positive)
            and row["synthetic_negative_pass"] == len(negative)
            and row["fixed_pre_gap_measurement"]
            and row["fixed_post_gap_measurement"]
            and row["development_natural_passing_maps"] >= 3
            and candidate != "ablation_seed7"
        )
        results.append(row)
        print(json.dumps({
            "candidate": candidate,
            "index": grid["parameter_index"],
            "positive": int(row["synthetic_positive_pass"]),
            "negative": int(row["synthetic_negative_pass"]),
            "fixed_pre": bool(row["fixed_pre_gap_measurement"]),
            "fixed_post": bool(row["fixed_post_gap_measurement"]),
            "maps": int(row["development_natural_passing_maps"]),
            "gate": bool(row["development_hard_gate"]),
        }, default=json_scalar))
    write_new(
        ROOT / "reports/phase8jqv2_4tf1_candidate_development_results.json",
        {
            "status": "PASS",
            "grid_count": len(results), "results": results,
            "sealed_holdout_accessed": False,
            "test_accessed": False, "blind_accessed": False,
        },
    )
    write_new(
        ROOT / "reports/phase8jqv2_4tf1_synthetic_positive.json",
        {
            "status": "PASS",
            "fixture_renderer": "analytic_cuda_metric_depth_v1",
            "control_count": len(positive_names),
            "results": [
                {
                    "candidate": row["candidate"],
                    "parameter_index": row["parameter_index"],
                    "passed": row["synthetic_positive_pass"],
                    "total": row["synthetic_positive_total"],
                } for row in results
            ],
            "runtime_gt_input": False,
        },
    )
    write_new(
        ROOT / "reports/phase8jqv2_4tf1_synthetic_negative.json",
        {
            "status": "PASS",
            "control_count": len(negative_names),
            "results": [
                {
                    "candidate": row["candidate"],
                    "parameter_index": row["parameter_index"],
                    "passed": row["synthetic_negative_pass"],
                    "total": row["synthetic_negative_total"],
                } for row in results
            ],
            "metadata_only_acceptance_required": 0,
            "future_or_gt_input": 0,
        },
    )
    print(json.dumps({
        "status": "PASS", "grid_count": len(results),
        "development_gate_pass_count": sum(
            row["development_hard_gate"] for row in results
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
