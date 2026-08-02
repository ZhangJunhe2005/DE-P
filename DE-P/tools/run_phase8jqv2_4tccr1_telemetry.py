#!/usr/bin/env python3
"""Real development detector->TrackManager telemetry for TCCR1."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "diagnostics/phase8jqv2_4tccr1"
CONTROLS = ROOT / "data/phase8_dynamic_perception_controls_v1"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.dynamic_perception_architecture_registry import (
    create_architecture, load_architecture_config,
)
from policy.dynamic.types import CameraModel, Pose


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def attention_reason(track, config):
    reasons = []
    if not track.is_confirmed:
        reasons.append("not_confirmed")
    if not track.is_dynamic:
        reasons.append("not_dynamic")
    if not track.ever_directly_observed:
        reasons.append("never_directly_observed")
    if track.prediction_only_age > config.max_missed_frames:
        reasons.append("prediction_age_exceeded")
    if track.visibility_state == "clear_missing":
        reasons.append("clear_missing")
    if track.confidence < config.track_confidence_threshold:
        reasons.append("confidence_below_threshold")
    return "authorized" if not reasons else "+".join(reasons)


def telemetry_row(case_id, category, frame, timestamp, track,
                  before, manager, result, config, pose):
    managed = manager._tracks[track.track_id]
    components = dict(track.confidence_components)
    weighted = {
        "confirmation": .15*components.get("confirmation", 0.),
        "association": .25*components.get("association", 0.),
        "velocity_uncertainty":
            .15*components.get("uncertainty", 0.),
        "motion_consistency":
            .25*components.get("motion_consistency", 0.),
        "foreground_support":
            .10*components.get("foreground_support", 0.),
        "cluster_geometry":
            .10*components.get("cluster_geometry", 0.),
    }
    raw_base = sum(weighted.values())
    split_factor = (
        .75 if managed.split_merge_suspected and track.is_dynamic
        else .25 if managed.split_merge_suspected else 1.
    )
    base_after_split = raw_base*split_factor
    miss_factor = config.confidence_missed_decay**track.missed_count
    details = {
        int(row["track_id"]): row
        for row in result.diagnostics["track_manager"]["match_details"]
    }
    match = details.get(track.track_id)
    covariance = np.asarray(track.state_covariance)
    velocity_std = float(np.sqrt(max(
        np.diag(covariance)[3:].max(), 0.
    )))
    position_std = float(np.sqrt(max(
        np.diag(covariance)[:3].max(), 0.
    )))
    previous = before.get(track.track_id)
    return {
        "case_id": case_id,
        "category": category,
        "frame": frame,
        "timestamp": float(timestamp),
        "camera_position_world": pose.position_world.tolist(),
        "rotation_world_from_camera":
            pose.rotation_world_from_camera.tolist(),
        "track_id": int(track.track_id),
        "measurement_present": bool(
            track.last_direct_observation_frame == frame
        ),
        "association_result": "matched" if match else (
            "birth" if track.birth_frame == frame else "unmatched_prediction"
        ),
        "base_confidence_raw": raw_base,
        "base_confidence_after_split_merge": base_after_split,
        "confidence_before_update":
            None if previous is None else float(previous.confidence),
        "confidence_after_update": float(track.confidence),
        "miss_decay_factor": miss_factor,
        "miss_decay_contribution":
            float(track.confidence-base_after_split),
        "measurement_contribution": float(
            weighted["association"]
            + weighted["foreground_support"]
            + weighted["cluster_geometry"]
        ),
        "velocity_uncertainty_contribution":
            weighted["velocity_uncertainty"],
        "position_uncertainty_contribution": 0.0,
        "position_uncertainty_used_in_confidence": False,
        "confidence_components": components,
        "weighted_confidence_components": weighted,
        "innovation_residual":
            None if match is None else match["innovation"],
        "association_cost":
            None if match is None else match["association_distance"],
        "association_mahalanobis_sq":
            None if match is None else match["association_mahalanobis_sq"],
        "kalman_state": np.r_[
            track.position_world, track.velocity_world
        ].tolist(),
        "covariance_diagonal": np.diag(covariance).tolist(),
        "velocity_estimate": track.velocity_world.tolist(),
        "velocity_norm": float(np.linalg.norm(track.velocity_world)),
        "velocity_uncertainty": velocity_std,
        "position_uncertainty": position_std,
        "confirmed": bool(track.is_confirmed),
        "is_dynamic": bool(track.is_dynamic),
        "dynamic_reason": track.dynamic_reason,
        "attention_authorized": bool(track.attention_authorized),
        "attention_reason": attention_reason(track, config),
        "missed_count": int(track.missed_count),
        "visibility_state": track.visibility_state,
        "track_age": int(track.age),
        "hit_count": int(track.hit_count),
        "consecutive_hits": int(track.consecutive_direct_hits),
        "prediction_only_age": int(track.prediction_only_age),
        "last_direct_measurement_frame":
            int(track.last_direct_observation_frame),
        "last_direct_measurement_time":
            None if track.last_direct_observation_frame < 0 else
            float(timestamp) - (
                frame-track.last_direct_observation_frame
            )*.1,
        "algorithm_state_read_only": True,
    }


def make_perception(sensor):
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    architecture = load_architecture_config()
    key = "physical_control_residual_v1"
    perception.range_foreground = create_architecture(
        key, config, architecture["candidates"][key]
    )
    return perception, config


def run_case(case_id, category, depths, poses, timestamps, model,
             offline=None):
    sensor = {
        "height": model.height, "width": model.width,
        "intrinsics": [model.fx, model.fy, model.cx, model.cy],
        "min_depth_m": model.min_depth,
        "max_depth_m": model.max_depth,
    }
    perception, config = make_perception(sensor)
    rows, frames = [], []
    previous = {}
    ever_dynamic = set()
    started = time.perf_counter()
    for index, timestamp in enumerate(timestamps):
        result = perception.update_depth(
            np.asarray(depths[index], dtype=np.float32),
            poses[index], float(timestamp), model,
        )
        current = {track.track_id: track for track in result.all_tracks}
        for track in result.all_tracks:
            rows.append(telemetry_row(
                case_id, category, index, timestamp, track,
                previous, perception.track_manager, result, config,
                poses[index],
            ))
            if track.is_dynamic:
                ever_dynamic.add(track.track_id)
        frames.append({
            "frame": index,
            "timestamp": float(timestamp),
            "observation_count": len(result.observations),
            "track_ids": sorted(current),
            "confirmed_track_ids": [
                int(track.track_id) for track in result.confirmed_tracks
            ],
            "dynamic_track_ids": [
                int(track.track_id) for track in result.dynamic_tracks
            ],
            "attention_nonzero":
                int(result.attention_map.count_nonzero().item()),
            "deleted_track_ids": list(
                result.diagnostics["track_manager"]["deleted_track_ids"]
            ),
        })
        previous = current
    runtime_ms = (time.perf_counter()-started)*1000/len(timestamps)
    return {
        "case_id": case_id,
        "category": category,
        "frames": frames,
        "runtime_average_ms": runtime_ms,
        "unique_track_ids": sorted({
            row["track_id"] for row in rows
        }),
        "ever_dynamic_track_ids": sorted(ever_dynamic),
        "telemetry": rows,
        "offline_evaluation": offline,
        "runtime_gt_used": False,
    }


def development_controls():
    manifest = json.loads((CONTROLS / "manifest.json").read_text())
    rows = []
    for control_id in manifest["control_ids"]:
        root = CONTROLS / "controls" / control_id
        control = json.loads((root / "control.json").read_text())
        if control["split"] != "development":
            continue
        runtime = control["runtime_inputs"]
        depths = np.load(root / runtime["depth_file"])
        positions = np.load(root / runtime["camera_positions"])
        yaws = np.load(root / runtime["camera_yaws"])
        timestamps = np.load(root / runtime["timestamps"])
        model = camera_model(manifest["sensor"])
        poses = [
            _pose(position, yaw, timestamp)
            for position, yaw, timestamp in zip(
                positions, yaws, timestamps
            )
        ]
        rows.append(run_case(
            control_id, "physical_controls", depths, poses,
            timestamps, model,
            offline={
                "semantic_class": control["semantic_class"],
                "expected_outcome": control["expected_outcome"],
                "actor_radii_m": (
                    control["actor_trajectory"]["radii_m"]
                    if control.get("actor_trajectory") is not None else []
                ),
            },
        ))
    return rows


def production_train_case(sequence_id):
    root = (
        ROOT / "data/phase8_dynamic_production/sequences" / sequence_id
    )
    metadata = yaml.safe_load((root / "metadata.yaml").read_text())
    if not sequence_id.startswith("phase8c_train_"):
        raise RuntimeError("TCCR1 may read only existing train development rows")
    rotation_body_from_camera = np.asarray(
        metadata["camera_rotation_body_from_camera"], dtype=np.float64
    )
    frame_rows = list(csv.DictReader((root / "frames.csv").open()))
    depths, poses, times = [], [], []
    for row in frame_rows:
        timestamp = float(row["timestamp"])
        position = np.asarray([
            float(row["camera_x"]), float(row["camera_y"]),
            float(row["camera_z"]),
        ])
        quaternion = np.asarray([
            float(row["camera_qx"]), float(row["camera_qy"]),
            float(row["camera_qz"]), float(row["camera_qw"]),
        ])
        body_rotation = Rotation.from_quat(quaternion).as_matrix()
        poses.append(Pose(
            position, body_rotation @ rotation_body_from_camera, timestamp
        ))
        depths.append(np.load(root / row["depth_path"]))
        times.append(timestamp)
    model = CameraModel(
        width=int(metadata["raw_image_width"]),
        height=int(metadata["raw_image_height"]),
        fx=float(metadata["camera_intrinsics"]["fx"]),
        fy=float(metadata["camera_intrinsics"]["fy"]),
        cx=float(metadata["camera_intrinsics"]["cx"]),
        cy=float(metadata["camera_intrinsics"]["cy"]),
        depth_scale=1.,
        min_depth=float(metadata["camera_intrinsics"]["min_depth"]),
        max_depth=float(metadata["camera_intrinsics"]["max_depth"]),
    )
    return run_case(
        sequence_id, "ordinary_dynamic",
        np.asarray(depths), poses, np.asarray(times), model,
        offline={"scenario_type": metadata["scenario_type"]},
    )


def ordinary_dynamic_cases():
    selected = defaultdict(list)
    for path in sorted((
        ROOT / "data/phase8_dynamic_production/sequences"
    ).glob("phase8c_train_*/metadata.yaml")):
        metadata = yaml.safe_load(path.read_text())
        kind = metadata["scenario_type"]
        if kind in {
            "crossing", "head_on", "multi_target",
            "temporal_separation", "occluded_but_tracked",
        } and len(selected[kind]) < 3:
            selected[kind].append(path.parent.name)
    return [
        production_train_case(sequence_id)
        for kind in sorted(selected)
        for sequence_id in selected[kind]
    ]


def eosr1_cases():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("EOSR1 telemetry rerender requires host CUDA")
    manifest = json.loads((
        REPORTS / "phase8jqv2_4eosr1_proof_witness_manifest.json"
    ).read_text())
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80., 80., 80., 45.],
        "min_depth_m": .1, "max_depth_m": 20.,
        "ray_step_m": .1, "frame_period_ns": 100_000_000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    model = camera_model(sensor)
    cases = []
    for witness in manifest["witnesses"]:
        backend = ExactAuthorityBVH(witness["authority_root"])
        actor = np.asarray(witness["actor_positions"], dtype=np.float64)
        camera = np.asarray(witness["camera_positions"], dtype=np.float64)
        yaws = np.asarray(witness["camera_yaws"], dtype=np.float64)
        radius = float(witness["radius_m"])
        diagnostics = renderer.render_with_actor_diagnostics(
            backend, camera, yaws, actor[:, None, :], [radius],
            return_owner_map=True,
        )
        timestamps = np.arange(len(camera), dtype=np.float64)*.1
        poses = [
            _pose(position, yaw, timestamp)
            for position, yaw, timestamp in zip(camera, yaws, timestamps)
        ]
        full = (
            diagnostics["per_actor_projected_pixel_count"][:, 0] > 0
        ) & (
            diagnostics["per_actor_visible_pixel_count"][:, 0] == 0
        ) & (
            diagnostics["per_actor_static_blocked_pixel_count"][:, 0]
            == diagnostics["per_actor_projected_pixel_count"][:, 0]
        )
        cases.append(run_case(
            witness["witness_id"], "natural_eosr1_gap1",
            diagnostics["composed_depth"], poses, timestamps, model,
            offline={
                "radius_m": radius,
                "strict_full_occlusion_frames":
                    np.flatnonzero(full).astype(int).tolist(),
                "expected_gap": witness["accepted_run"],
            },
        ))
    return cases


def confidence_summary(cases):
    successful = set()
    dynamic = set()
    samples = []
    pre_gap = []
    negatives = []
    gap_events = []
    eosr_witness_runs = 0
    eosr_pre_gap_dynamic_events = 0
    for case in cases:
        case_key = case["case_id"]
        per_track = defaultdict(list)
        for row in case["telemetry"]:
            key = (case_key, row["track_id"])
            per_track[row["track_id"]].append(row)
            if row["confirmed"]:
                successful.add(key)
            if row["is_dynamic"]:
                dynamic.add(key)
                samples.append(row)
        for track_id, trace in per_track.items():
            trace.sort(key=lambda item: item["frame"])
            for previous, current in zip(trace, trace[1:]):
                if (
                    previous["is_dynamic"]
                    and current["missed_count"] == 1
                    and not current["measurement_present"]
                ):
                    pre_gap.append(previous)
                    gap_events.append({
                        "case_id": case_key,
                        "category": case["category"],
                        "track_id": track_id,
                        "pre_gap_frame": previous["frame"],
                        "gap_frame": current["frame"],
                        "pre_gap_confidence":
                            previous["confidence_after_update"],
                        "gap_confidence":
                            current["confidence_after_update"],
                        "same_id": True,
                        "track_exists": True,
                        "dynamic_survives": current["is_dynamic"],
                        "attention_survives":
                            current["attention_authorized"],
                    })
        if case["category"] == "natural_eosr1_gap1":
            eosr_witness_runs += 1
            gap = case["offline_evaluation"]["expected_gap"]
            gap_start = gap[0]
            before = [
                row for row in case["telemetry"]
                if row["frame"] < gap_start and row["is_dynamic"]
            ]
            eosr_pre_gap_dynamic_events += int(bool(before))
        if (
            case["category"] == "physical_controls"
            and not case["offline_evaluation"]["actor_radii_m"]
        ):
            negatives.append(case)
    values = np.asarray([
        row["confidence_after_update"] for row in samples
    ], dtype=np.float64)
    def distribution(rows):
        value = np.asarray([
            row["confidence_after_update"] for row in rows
        ], dtype=np.float64)
        if not len(value):
            return None
        quantiles = np.percentile(
            value, (0, 10, 25, 50, 75, 90, 95, 100)
        )
        return {
            key: float(number) for key, number in zip(
                ("minimum", "p10", "p25", "median", "p75", "p90",
                 "p95", "maximum"), quantiles
            )
        } | {
            "fraction_ge_0_95": float(np.mean(value >= .95)),
            "fraction_ge_0_98": float(np.mean(value >= .98)),
            "fraction_ge_1_00": float(np.mean(value >= 1.)),
            "sample_count": len(value),
        }
    return {
        "successful_confirmed_tracks": len(successful),
        "confirmed_dynamic_tracks": len(dynamic),
        "dynamic_frame_distribution": distribution(samples),
        "pre_gap_distribution": distribution(pre_gap),
        "pre_gap_rows": pre_gap,
        "gap_events": gap_events,
        "gap1_enter_events": len(gap_events),
        "natural_eosr1_witness_runs": eosr_witness_runs,
        "natural_eosr1_pre_gap_dynamic_events":
            eosr_pre_gap_dynamic_events,
        "negative_sequences": len(negatives),
        "negative_false_attention_frames": sum(
            bool(frame["attention_nonzero"])
            for case in negatives for frame in case["frames"]
        ),
    }


def main():
    import torch
    started = time.perf_counter()
    cases = development_controls()
    cases.extend(ordinary_dynamic_cases())
    cases.extend(eosr1_cases())
    summary = confidence_summary(cases)
    OUT.mkdir(parents=True, exist_ok=True)
    telemetry_path = OUT / "runtime_telemetry.jsonl"
    temporary = telemetry_path.with_name(
        f".{telemetry_path.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w") as target:
        for case in cases:
            for row in case["telemetry"]:
                target.write(json.dumps(row, sort_keys=True)+"\n")
    os.replace(temporary, telemetry_path)
    case_path = OUT / "case_summary.json"
    atomic_json(case_path, {
        "cases": [{
            key: value for key, value in case.items()
            if key != "telemetry"
        } for case in cases]
    })
    result = {
        "status": "PASS" if (
            summary["successful_confirmed_tracks"] >= 30
            and summary["confirmed_dynamic_tracks"] >= 20
            and summary["gap1_enter_events"] >= 6
            and summary["negative_sequences"] >= 10
        ) else "FAIL",
        **summary,
        "case_count": len(cases),
        "category_counts": dict(Counter(
            case["category"] for case in cases
        )),
        "telemetry_file": str(telemetry_path),
        "telemetry_sha256": sha(telemetry_path),
        "case_summary_file": str(case_path),
        "elapsed_seconds": time.perf_counter()-started,
        "device": torch.cuda.get_device_name(0),
        "detector_executed": True,
        "track_manager_executed": True,
        "confidence_artificially_set": False,
        "covariance_artificially_set": False,
        "track_id_artificially_set": False,
        "dynamic_artificially_set": False,
        "attention_artificially_set": False,
        "runtime_gt_used": False,
        "offline_gt_used_for_evaluation_only": True,
        "holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(
        REPORTS / "phase8jqv2_4tccr1_real_pre_gap_confidence_distribution.json",
        result,
    )
    print(json.dumps({
        key: result[key] for key in (
            "status", "successful_confirmed_tracks",
            "confirmed_dynamic_tracks", "gap1_enter_events",
            "natural_eosr1_witness_runs",
            "natural_eosr1_pre_gap_dynamic_events",
            "negative_sequences", "negative_false_attention_frames",
            "case_count", "category_counts", "elapsed_seconds", "device",
        )
    }, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("TCCR1 runtime telemetry minimums not reached")


if __name__ == "__main__":
    main()
