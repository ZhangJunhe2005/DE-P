#!/usr/bin/env python3
"""Bounded time-shift-only schedule stage for one retained gap class."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa
from authoritative_dataset.dynamic_motion_v2 import (  # noqa
    CONTRACT_VERSION_V2_1,
    _dense_times,
    load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa
from authoritative_dataset.occlusion_constructor_v2_1 import (  # noqa
    _trajectory_valid_exact,
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_identity_schedule_v1 import (  # noqa
    SCHEDULE_VERSION,
    derive_time_shift_schedule,
    verify_time_shift_only,
)
from tools.run_phase8jqv2_4i1_retained_audit import (  # noqa
    canonical_hash,
    trace_case,
)


MANIFEST = ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json"
MAP_SET = ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json"
REJECTION_ROOT = ROOT / "diagnostics/phase8jqv2_4i1"


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [convert(item) for item in value]
    return value


def atomic_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite stage evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        convert(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-candidates", type=int, default=33)
    parser.add_argument("--maximum-runtime-seconds", type=float, default=240.0)
    return parser.parse_args()


def matched_track(row):
    track_id = row["track"]["matched_track_id"]
    return next((
        track for track in row["track"]["tracks"]
        if track["track_id"] == track_id
    ), None)


def strict_identity(rows, gap_start, gap_end):
    reasons = []
    if gap_start < 2 or gap_end + 2 >= len(rows):
        return {"pass": False, "reasons": ["insufficient_sequence_margin"]}
    guard = matched_track(rows[gap_start - 2])
    pre = matched_track(rows[gap_start - 1])
    track_id = (
        None if pre is None else pre["track_id"]
    )
    for label, track in (("guard", guard), ("pre", pre)):
        if track is None:
            reasons.append(f"{label}_track_missing")
            continue
        if track["track_id"] != track_id:
            reasons.append(f"{label}_track_id_mismatch")
        if not track["confirmed"]:
            reasons.append(f"{label}_not_confirmed")
        if not track["is_dynamic"]:
            reasons.append(f"{label}_not_dynamic")
        if track["prediction_only_age"] != 0:
            reasons.append(f"{label}_not_direct")
        if not track["attention_authorized"]:
            reasons.append(f"{label}_attention_not_authorized")
    if not rows[gap_start - 1]["detection"]["measurement_valid"]:
        reasons.append("pre_measurement_invalid")
    gap_tracks = []
    for index in range(gap_start, gap_end + 1):
        row = rows[index]
        sensor = row["sensor"]
        track = matched_track(row)
        if not sensor["natural_occlusion_verified"]:
            reasons.append(f"gap_{index}_not_natural_occlusion")
        if track is None:
            reasons.append(f"gap_{index}_track_missing")
            continue
        gap_tracks.append(track["track_id"])
        if track["track_id"] != track_id:
            reasons.append(f"gap_{index}_track_id_mismatch")
        if not track["confirmed"]:
            reasons.append(f"gap_{index}_not_confirmed")
        if not track["is_dynamic"]:
            reasons.append(f"gap_{index}_not_dynamic")
        if track["prediction_only_age"] < 1:
            reasons.append(f"gap_{index}_not_prediction_only")
        if track["visibility_state"] != "occluded":
            reasons.append(f"gap_{index}_visibility_not_occluded")
        if not track["attention_authorized"]:
            reasons.append(f"gap_{index}_attention_not_authorized")
        if row["track"]["deleted_track_ids"]:
            reasons.append(f"gap_{index}_deletion")
        if row["track"]["created_track_ids"]:
            reasons.append(f"gap_{index}_new_track")
    post_ids = []
    for index in (gap_end + 1, gap_end + 2):
        row = rows[index]
        track = matched_track(row)
        if track is None:
            reasons.append(f"post_{index}_track_missing")
            continue
        post_ids.append(track["track_id"])
        if track["track_id"] != track_id:
            reasons.append(f"post_{index}_track_id_mismatch")
        if track["prediction_only_age"] != 0:
            reasons.append(f"post_{index}_not_direct")
        if row["track"]["replaced_this_frame"]:
            reasons.append(f"post_{index}_replacement")
        if not row["detection"]["measurement_valid"]:
            reasons.append(f"post_{index}_measurement_invalid")
    return {
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "track_id": track_id,
        "guard_frame": gap_start - 2,
        "pre_frame": gap_start - 1,
        "gap_track_ids": gap_tracks,
        "post_track_ids": post_ids,
    }


def candidate_case_order(cases):
    # Prefer the original case with greatest total visible support.
    trace_root = ROOT / "diagnostics/phase8jqv2_4i1/traces"
    support = {}
    for case in cases:
        path = trace_root / f"{case['case_id']}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        support[case["case_id"]] = sum(
            row["sensor"]["visible_pixel_count"] for row in rows
        )
    return sorted(
        cases, key=lambda case: (
            -support[case["case_id"]], case["case_id"]
        )
    )


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for schedule validation")
    manifest = json.loads(MANIFEST.read_text())
    cases = candidate_case_order([
        case for case in manifest["cases"]
        if case["requested_gap"] == args.gap
    ])
    if not cases:
        raise RuntimeError(f"no retained gap-{args.gap} case")
    map_rows = {
        row["map_uuid"]: row
        for row in json.loads(MAP_SET.read_text())["maps"]
    }
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    started = time.perf_counter()
    rejections = []
    accepted = None
    selected_trace = None
    evaluated = 0
    for case in cases:
        backend = ExactAuthorityBVH(
            map_rows[case["map_uuid"]]["authority_root"]
        )
        timestamps = np.asarray(
            case["sequence_timestamps"], dtype=np.float64
        )
        camera_positions = np.asarray(
            case["camera_trajectory"], dtype=np.float64
        )
        camera_yaw = np.asarray(case["camera_yaw"], dtype=np.float64)
        initial = np.asarray(
            case["actor_trajectory"][0], dtype=np.float64
        )
        velocity = np.asarray(
            case["actor_velocity_world"], dtype=np.float64
        )
        voxel_centers = (
            backend.map.origin[None, :]
            + (backend.occupied.astype(np.float64) + .5)
            * backend.map.resolution
        )
        occupancy_tree = cKDTree(voxel_centers)
        validation_times = _dense_times(
            float(timestamps[-1] + 1.7),
            float(np.linalg.norm(velocity)),
            float(contract["sampling"][
                "continuous_sample_spacing_m"
            ]),
        )
        for shift in range(args.maximum_candidates):
            if time.perf_counter() - started > args.maximum_runtime_seconds:
                rejections.append({
                    "case_id": case["case_id"],
                    "shift_frames": shift,
                    "reason": "maximum_runtime_exceeded",
                })
                break
            schedule = derive_time_shift_schedule(
                initial, velocity, case["observed_gap"][0],
                args.gap, shift, .1,
            )
            expected_end = (
                schedule.expected_gap_start + args.gap - 1
            )
            if expected_end + 2 >= len(timestamps):
                rejections.append({
                    "case_id": case["case_id"],
                    "shift_frames": shift,
                    "reason": "insufficient_post_gap_sequence",
                })
                continue
            invariants = verify_time_shift_only(
                schedule, initial, velocity, timestamps
            )
            actor_positions = schedule.actor_positions(timestamps)
            dense_positions = schedule.actor_positions(validation_times)
            camera_dense = np.stack([
                np.interp(
                    validation_times, timestamps, camera_positions[:, axis]
                )
                for axis in range(3)
            ], axis=1)
            if not _trajectory_valid_exact(
                backend, occupancy_tree, dense_positions, camera_dense,
                float(case["actor_radius_m"]),
                contract["sampling"], validation_times,
            ):
                rejections.append({
                    "case_id": case["case_id"],
                    "shift_frames": shift,
                    "reason": "continuous_motion_contract",
                    "invariants": invariants,
                })
                continue
            actor_frames = actor_positions[:, None, :]
            diagnostics = renderer.render_with_actor_diagnostics(
                backend, camera_positions, camera_yaw,
                actor_frames, [float(case["actor_radius_m"])],
                return_owner_map=False,
                return_actor_near_depth=False,
            )
            pattern = natural_occlusion_pattern(
                diagnostics, 0, contract
            )
            target = next((
                gap for gap in pattern["accepted_gaps"]
                if list(gap) == [
                    schedule.expected_gap_start, expected_end
                ]
            ), None)
            if target is None:
                rejections.append({
                    "case_id": case["case_id"],
                    "shift_frames": shift,
                    "reason": "exact_natural_gap_not_preserved",
                    "observed_accepted_gaps":
                        pattern["accepted_gaps"],
                    "invariants": invariants,
                })
                continue
            actor = {
                "actor_id": 0,
                "start": schedule.actor_initial_position,
                "velocity": schedule.actor_velocity,
                "radius_m": float(case["actor_radius_m"]),
                "motion_profile": "constant_velocity",
                "configured_speed_mps":
                    float(np.linalg.norm(schedule.actor_velocity)),
                "motion_contract_version": CONTRACT_VERSION_V2_1,
                "sampling_method": SCHEDULE_VERSION,
            }
            synthetic_construction = SimpleNamespace(
                actors=[actor], diagnostics=diagnostics
            )
            uav = SimpleNamespace(
                position_world=camera_positions, yaw=camera_yaw
            )
            scheduled_case = dict(case)
            scheduled_case["observed_gap"] = [
                schedule.expected_gap_start, expected_end
            ]
            rows = trace_case(
                scheduled_case, synthetic_construction, uav,
                actor_frames, timestamps, sensor,
            )
            identity = strict_identity(
                rows, schedule.expected_gap_start, expected_end
            )
            evaluated += 1
            candidate = {
                "case_id": case["case_id"],
                "shift_frames": shift,
                "gap_start": schedule.expected_gap_start,
                "gap_end": expected_end,
                "schedule_certificate": schedule.certificate,
                "invariants": invariants,
                "identity": identity,
                "trace_hash": canonical_hash(rows),
                "measurement_support": {
                    "pre_visible_pixels":
                        rows[schedule.expected_gap_start - 1][
                            "sensor"
                        ]["visible_pixel_count"],
                    "pre_foreground_pixels":
                        rows[schedule.expected_gap_start - 1][
                            "detection"
                        ]["foreground_pixel_count"],
                    "pre_measurement_valid":
                        rows[schedule.expected_gap_start - 1][
                            "detection"
                        ]["measurement_valid"],
                    "post_measurement_valid": [
                        rows[index]["detection"]["measurement_valid"]
                        for index in (
                            expected_end + 1, expected_end + 2
                        )
                    ],
                },
            }
            if identity["pass"]:
                accepted = candidate
                selected_trace = rows
                break
            rejections.append({
                **candidate,
                "reason": "frozen_identity_gate",
            })
        if accepted is not None:
            break
    output = ROOT / args.output
    trace_path = None
    if accepted is not None:
        trace_path = REJECTION_ROOT / (
            f"schedule_gap{args.gap}_{accepted['case_id']}.jsonl"
        )
        if trace_path.exists():
            raise FileExistsError(
                f"refusing to overwrite schedule trace: {trace_path}"
            )
        with trace_path.open("x") as stream:
            for row in selected_trace:
                stream.write(json.dumps(
                    row, sort_keys=True, allow_nan=False
                ) + "\n")
    report = {
        "status": "PASS" if accepted is not None else "FAIL",
        "stage_gap": args.gap,
        "schedule_version": SCHEDULE_VERSION,
        "case_order": [case["case_id"] for case in cases],
        "maximum_schedule_candidates": args.maximum_candidates,
        "maximum_runtime_seconds": args.maximum_runtime_seconds,
        "deterministic_ordering": "visible_support_desc_then_case_id_then_shift_asc",
        "candidates_with_full_perception_evaluation": evaluated,
        "accepted": accepted,
        "accepted_trace": (
            None if trace_path is None
            else str(trace_path.relative_to(ROOT))
        ),
        "rejection_count": len(rejections),
        "rejections": rejections,
        "elapsed_seconds": time.perf_counter() - started,
        "actor_speed_modified": False,
        "actor_direction_modified": False,
        "actor_acceleration_modified": False,
        "camera_geometry_modified": False,
        "frozen_perception_parameters_modified": False,
        "annex_used": False,
        "new_map_sweep_executed": False,
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_new_json(output, report)
    rejection_path = REJECTION_ROOT / (
        f"schedule_rejections_gap{args.gap}.json"
    )
    atomic_new_json(rejection_path, {
        "status": "PASS",
        "stage_status": report["status"],
        "rejections": rejections,
    })
    print(json.dumps({
        "status": report["status"],
        "gap": args.gap,
        "accepted": accepted,
        "rejection_count": len(rejections),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
