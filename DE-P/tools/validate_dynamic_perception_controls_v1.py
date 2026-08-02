#!/usr/bin/env python3
"""Independent physical/semantic validator for CCR1 control artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa: E402
from authoritative_dataset.dynamic_perception_control_schema_v1 import (  # noqa: E402
    CONTRACT_VERSION, file_hash, validate_control,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa: E402
from authoritative_dataset.warp_visibility_reference_v1 import (  # noqa: E402
    classify_pair, summarize,
)


CONFIG_PATH = (
    ROOT / "configs/dynamic_perception_physical_control_contract_v1.yaml"
)
DEFAULT_DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"


def analytic_projection(focal, radius, distance):
    radius_pixels = focal*radius/math.sqrt(distance**2-radius**2)
    return {
        "diameter_pixels": 2*radius_pixels,
        "area_pixels2": math.pi*radius_pixels**2,
    }


def analytic_raster_mask(sensor, center_body, radius):
    height, width = int(sensor["height"]), int(sensor["width"])
    fx, fy, cx, cy = [float(x) for x in sensor["intrinsics"]]
    vv, uu = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64), indexing="ij",
    )
    rays = np.stack((
        np.ones_like(uu), (uu-cx)/fx, (vv-cy)/fy,
    ), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    center = np.asarray(center_body, dtype=np.float64)
    projection = np.sum(rays*center, axis=-1)
    perpendicular2 = np.sum(center*center)-projection*projection
    near = projection-np.sqrt(np.maximum(radius**2-perpendicular2, 0))
    return (
        (projection > 0) & (perpendicular2 <= radius**2)
        & (near > 0) & (near <= float(sensor["max_depth_m"]))
    )


def world_to_body(vector, yaw):
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    world_from_body = np.asarray(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(vector, dtype=np.float64) @ world_from_body.T


def load_control(directory):
    value = json.loads((directory / "control.json").read_text())
    validate_control(value)
    for name, expected in value["files"].items():
        actual = file_hash(directory / name)
        if actual != expected:
            raise RuntimeError(f"{value['control_id']}/{name} hash mismatch")
    return value


def max_motion(positions, yaws):
    translation = (
        float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max())
        if len(positions) > 1 else 0.0
    )
    yaw = (
        float(np.abs(np.diff(yaws)).max()) if len(yaws) > 1 else 0.0
    )
    return translation, yaw


def validate_one(directory, manifest, config, renderer, backend_cache):
    value = load_control(directory)
    control_id = value["control_id"]
    depths = np.load(directory / "depth.npy", allow_pickle=False)
    static = np.load(directory / "static_depth.npy", allow_pickle=False)
    positions = np.load(
        directory / "camera_positions.npy", allow_pickle=False
    )
    yaws = np.load(directory / "camera_yaws.npy", allow_pickle=False)
    timestamps = np.load(directory / "timestamps.npy", allow_pickle=False)
    owner = np.load(
        directory / "nearest_actor_owner.npy", allow_pickle=False
    )
    actors = (
        np.load(directory / "actor_positions.npy", allow_pickle=False)
        if (directory / "actor_positions.npy").is_file() else
        np.empty((len(positions), 0, 3), dtype=np.float64)
    )
    radii = (
        value["actor_trajectory"]["radii_m"]
        if value["actor_trajectory"] is not None else []
    )
    authority_root = Path(value["authority"]["root"])
    key = str(authority_root)
    backend = backend_cache.setdefault(key, ExactAuthorityBVH(authority_root))
    rerender = renderer.render_with_actor_diagnostics(
        backend, positions, yaws, actors, radii,
        return_owner_map=True, return_actor_near_depth=True,
    )
    depth_difference = float(np.max(np.abs(
        rerender["composed_depth"]-depths
    )))
    static_difference = float(np.max(np.abs(
        rerender["static_depth"]-static
    )))
    owner_equal = bool(np.array_equal(
        rerender["nearest_actor_owner"], owner
    ))
    errors = []
    tolerance = config["validation"]
    if depth_difference > tolerance["rerender_depth_abs_tolerance_m"]:
        errors.append("rerender_depth_mismatch")
    if static_difference > tolerance["rerender_depth_abs_tolerance_m"]:
        errors.append("rerender_static_depth_mismatch")
    if not owner_equal:
        errors.append("rerender_owner_mismatch")
    if value["manual_depth_overwrite"]:
        errors.append("manual_depth_overwrite")
    translation, yaw_delta = max_motion(positions, yaws)
    provenance = []
    for index in range(1, len(positions)):
        reference = classify_pair(
            static[index-1], static[index],
            positions[index-1], yaws[index-1],
            positions[index], yaws[index], manifest["sensor"],
            tolerance["warp_depth_consistency_tolerance_m"],
        )
        provenance.append({"frame": index, **summarize(reference)})
    totals = {
        key: sum(row.get(key, 0) for row in provenance)
        for key in provenance[0] if key != "frame"
    } if provenance else {}
    changed_pixels = int(np.any(
        np.abs(np.diff(static, axis=0)) > 1e-6, axis=0
    ).sum()) if len(static) > 1 else 0
    expected = value["provenance_expectation"]
    required_channel = expected.get("required_channel")
    if required_channel:
        channel_key = f"{required_channel}_pixels"
        if totals.get(channel_key, 0) < int(expected["minimum_pixels"]):
            errors.append(f"missing_provenance:{required_channel}")
    if value["semantic_class"] == "required_hard_negative":
        if actors.shape[1] != 0 or np.any(owner >= 0):
            errors.append("static_negative_contains_actor")
        if translation == 0 and yaw_delta == 0:
            errors.append("static_motion_control_has_zero_camera_motion")
        if changed_pixels < tolerance["minimum_changed_pixels"]:
            errors.append("static_motion_control_has_no_depth_change")
    camera_motion = value.get("physical", {}).get("camera_motion")
    if camera_motion in ("yaw_lateral", "yaw_forward"):
        if translation == 0 or yaw_delta == 0:
            errors.append("combined_camera_motion_component_missing")
    if value["control_id"].startswith("edge_actor_moving_camera"):
        if translation == 0 or yaw_delta == 0:
            errors.append("paired_actor_camera_motion_component_missing")

    actor_report = []
    if actors.shape[1]:
        if value["actor_trajectory"]["positions_world"] != actors.tolist():
            errors.append("metadata_render_actor_position_mismatch")
        dt = np.diff(timestamps)
        velocity = np.diff(actors, axis=0)/dt[:, None, None]
        expected_velocity = np.asarray(
            value["physical"]["velocity_world_mps"], dtype=np.float64
        )
        velocity_error = float(np.max(np.abs(
            velocity[:, 0]-expected_velocity
        )))
        acceleration = (
            np.diff(velocity[:, 0], axis=0)/dt[1:, None]
            if len(velocity) > 1 else np.zeros((0, 3))
        )
        acceleration_max = float(np.max(np.linalg.norm(
            acceleration, axis=1
        ))) if len(acceleration) else 0.0
        if velocity_error > tolerance["velocity_abs_tolerance_mps"]:
            errors.append("metadata_velocity_mismatch")
        if (
            acceleration_max
            > tolerance["acceleration_abs_tolerance_mps2"]
        ):
            errors.append("actor_acceleration_contract")
        reference_frame = int(value["physical"].get("reference_frame", 0))
        for actor_index, radius in enumerate(radii):
            relative = (
                actors[:, actor_index]-positions
            )
            distances = np.linalg.norm(relative, axis=1)
            projected = rerender[
                "per_actor_projected_pixel_count"
            ][:, actor_index]
            visible = rerender[
                "per_actor_visible_pixel_count"
            ][:, actor_index]
            near = rerender["actor_near_depth"][:, actor_index]
            projected_mask = np.isfinite(near[reference_frame])
            vv, uu = np.nonzero(projected_mask)
            bbox = (
                [int(uu.min()), int(vv.min()), int(uu.max()), int(vv.max())]
                if len(uu) else None
            )
            analytic = analytic_projection(
                float(manifest["sensor"]["intrinsics"][0]),
                float(radius), float(distances[reference_frame]),
            )
            center_body = world_to_body(
                actors[reference_frame, actor_index]
                - positions[reference_frame],
                yaws[reference_frame],
            )
            cpu_raster = analytic_raster_mask(
                manifest["sensor"], center_body, float(radius)
            )
            cpu_v, cpu_u = np.nonzero(cpu_raster)
            cpu_bbox = (
                [
                    int(cpu_u.min()), int(cpu_v.min()),
                    int(cpu_u.max()), int(cpu_v.max()),
                ] if len(cpu_u) else None
            )
            diameter_raster = (
                max(bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1)
                if bbox is not None else 0
            )
            diameter_error = abs(
                diameter_raster-analytic["diameter_pixels"]
            )
            area_relative_error = abs(
                float(projected[reference_frame])-analytic["area_pixels2"]
            )/max(analytic["area_pixels2"], 1e-9)
            role = value["role"]
            if role == "hard_positive" and (
                int(cpu_raster.sum()) != int(projected[reference_frame])
                or cpu_bbox != bbox
            ):
                errors.append("cpu_cuda_sphere_raster_mismatch")
            is_center_axis = (
                abs(center_body[1]) <= 1e-9
                and abs(center_body[2]) <= 1e-9
            )
            if is_center_axis and role == "hard_positive":
                if (
                    diameter_error
                    > tolerance["projection_diameter_abs_tolerance_px"]
                ):
                    errors.append("analytic_raster_diameter_mismatch")
                if (
                    area_relative_error
                    > tolerance["projection_area_relative_tolerance"]
                ):
                    errors.append("analytic_raster_area_mismatch")
            if (
                role == "hard_positive"
                and visible[reference_frame] != projected[reference_frame]
            ):
                errors.append("full_projection_static_occlusion")
            minimum_actor_uav = (
                float(radius)
                + float(config["physics"]["uav_radius_m"])
                + float(config["physics"][
                    "actor_uav_clearance_margin_m"
                ])
            )
            if (
                value["semantic_class"]
                != "diagnostic_out_of_contract"
                and float(distances.min()) < minimum_actor_uav-1e-9
            ):
                errors.append("actor_uav_clearance")
            collision = [
                backend.query_one(center, float(radius))["collision"]
                for center in actors[:, actor_index]
            ]
            if (
                value["semantic_class"]
                != "diagnostic_out_of_contract" and any(collision)
            ):
                errors.append("actor_static_collision")
            actor_report.append({
                "actor_index": actor_index,
                "radius_m": float(radius),
                "distance_reference_m":
                    float(distances[reference_frame]),
                "distance_range_m": [
                    float(distances.min()), float(distances.max())
                ],
                "speed_mps": float(np.linalg.norm(expected_velocity)),
                "velocity_max_abs_error_mps": velocity_error,
                "acceleration_max_mps2": acceleration_max,
                "analytic_diameter_pixels": analytic["diameter_pixels"],
                "analytic_area_pixels2": analytic["area_pixels2"],
                "cuda_projected_pixels":
                    int(projected[reference_frame]),
                "cpu_analytic_raster_pixels": int(cpu_raster.sum()),
                "cpu_analytic_raster_bbox_uv": cpu_bbox,
                "center_axis_continuous_formula_applicable": is_center_axis,
                "cuda_visible_pixels": int(visible[reference_frame]),
                "raster_bbox_uv": bbox,
                "raster_diameter_pixels": diameter_raster,
                "diameter_abs_error_pixels": diameter_error,
                "area_relative_error": area_relative_error,
                "depth_range_m": [
                    float(np.min(near[np.isfinite(near)])),
                    float(np.max(near[np.isfinite(near)])),
                ],
                "static_collision": any(collision),
            })
        if value["role"] == "partial_occlusion_positive":
            reference = int(value["physical"]["reference_frame"])
            projected = rerender[
                "per_actor_projected_pixel_count"
            ][reference, 0]
            visible = rerender[
                "per_actor_visible_pixel_count"
            ][reference, 0]
            if not (0 < visible < projected):
                errors.append("partial_occlusion_not_realized")
        if value["role"] == "paired_edge_positive":
            visible_frames = np.flatnonzero(
                rerender["per_actor_visible_pixel_count"][:, 0] > 0
            )
            full_frames = []
            for index in visible_frames:
                near = rerender["actor_near_depth"][index, 0]
                vv, uu = np.nonzero(np.isfinite(near))
                if (
                    len(uu)
                    and uu.min() > 0 and uu.max() < depths.shape[2]-1
                    and vv.min() > 0 and vv.max() < depths.shape[1]-1
                ):
                    full_frames.append(int(index))
            k = int(config["latency"]["causal_support_frames_k"])
            consecutive = 0
            maximum = 0
            for index in range(len(positions)):
                if index in full_frames:
                    consecutive += 1
                    maximum = max(maximum, consecutive)
                else:
                    consecutive = 0
            if maximum < k:
                errors.append("edge_actor_lacks_stable_overlap_latency")
    return {
        "control_id": control_id,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "split": value["split"],
        "semantic_class": value["semantic_class"],
        "role": value["role"],
        "rerender_depth_max_abs_error_m": depth_difference,
        "rerender_static_max_abs_error_m": static_difference,
        "rerender_owner_equal": owner_equal,
        "camera_translation_max_m": translation,
        "camera_yaw_delta_max_rad": yaw_delta,
        "changed_pixels": changed_pixels,
        "provenance_totals": totals,
        "all_depth_valid": bool(np.all(
            np.isfinite(static)
            & (static < float(manifest["sensor"]["max_depth_m"])-1e-6)
        )),
        "actor_validation": actor_report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if dataset != DEFAULT_DATASET.resolve() and not (
        args.smoke and str(dataset).startswith("/tmp/")
    ):
        raise RuntimeError("unexpected CCR1 dataset root")
    manifest = json.loads((dataset / "manifest.json").read_text())
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise RuntimeError("manifest contract mismatch")
    if manifest["validator_hash"] != file_hash(Path(__file__)):
        raise RuntimeError("validator source hash mismatch")
    renderer = CudaAuthorityRenderer(
        manifest["sensor"], f"cuda:{args.device}"
    )
    backend_cache = {}
    rows = []
    started = time.perf_counter()
    for control_id in manifest["control_ids"]:
        row = validate_one(
            dataset / "controls" / control_id,
            manifest, config, renderer, backend_cache,
        )
        rows.append(row)
        print(json.dumps({
            "control_id": control_id, "status": row["status"],
            "errors": row["errors"],
        }))
    failures = [row for row in rows if row["status"] != "PASS"]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "contract_version": CONTRACT_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "control_count": len(rows),
        "passed": len(rows)-len(failures),
        "failed": len(failures),
        "failure_control_ids": [
            row["control_id"] for row in failures
        ],
        "controls": rows,
        "architecture_executed": False,
        "detector_executed": False,
        "tf1_holdout_accessed": False,
        "elapsed_seconds": time.perf_counter()-started,
    }
    output = dataset / "physical_validation.json"
    text = json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        default=lambda value: (
            value.item() if isinstance(value, np.generic) else
            (_ for _ in ()).throw(TypeError(
                f"{type(value).__name__} is not JSON serializable"
            ))
        ),
    )+"\n"
    if output.exists() and output.read_text() != text:
        raise FileExistsError("refusing to overwrite different validation")
    output.write_text(text)
    print(json.dumps({
        "status": result["status"], "passed": result["passed"],
        "failed": result["failed"],
    }, indent=2))
    if failures:
        raise RuntimeError(
            f"{len(failures)} physical controls failed validation"
        )


if __name__ == "__main__":
    main()
