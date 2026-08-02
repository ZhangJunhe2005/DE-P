#!/usr/bin/env python3
"""Replay exactly the six retained natural cases and emit causal traces."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa
from authoritative_dataset.dynamic_motion_v2 import (  # noqa
    actor_position,
    load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa
from authoritative_dataset.occlusion_constructor_v2_2 import (  # noqa
    CONSTRUCTOR_VERSION,
    build_occluded_actor_specs_v2_2,
    sample_occlusion_uav_sequence_v2_2,
)
from authoritative_dataset.perception_probe_v2 import (  # noqa
    _pose,
    camera_model,
    frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception  # noqa


FIRST = ROOT / (
    "reports/occlusion_constructor_v2_2_"
    "natural_sweep_first_seed_all_types_gaps123.json"
)
REMAINING = ROOT / (
    "reports/occlusion_constructor_v2_2_"
    "natural_sweep_remaining_seeds_all_types_gaps123.json"
)
MAP_SET = ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json"
MANIFEST = ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json"
SUMMARY = ROOT / "reports/phase8jqv2_4i1_per_frame_trace_summary.json"
TRACE_ROOT = ROOT / "diagnostics/phase8jqv2_4i1/traces"


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [convert(item) for item in value]
    return value


def canonical_hash(value):
    raw = json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        convert(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    os.replace(temporary, path)


def retained_source_rows():
    rows = []
    for path in (FIRST, REMAINING):
        report = json.loads(path.read_text())
        for row in report["results"]:
            if row["geometry_status"] == "PASS":
                rows.append((path, row))
    if len(rows) != 6:
        raise RuntimeError(f"expected six retained cases, found {len(rows)}")
    return rows


def case_id(row):
    return (
        f"natural_{row['natural_type']}_{row['map_uuid'][:8]}"
        f"_seed{row['seed']}_gap{row['requested_gap_frames']}"
    )


def visibility_reason(projected, visible, blocked, outside, behind, beyond):
    if behind:
        return "behind_camera"
    if beyond:
        return "beyond_max_depth"
    if outside:
        return "outside_fov"
    if projected <= 0:
        return "not_projected"
    if visible > 0:
        return "directly_visible"
    if blocked == projected:
        return "full_static_occlusion"
    if blocked > 0:
        return "partial_static_occlusion"
    return "not_visible_unexplained"


def observation_row(observation):
    return {
        "observation_id": observation.observation_id,
        "temporary_cluster_id": observation.temporary_cluster_id,
        "centroid_world": observation.centroid_world,
        "centroid_camera": observation.centroid_camera,
        "depth": float(observation.centroid_camera[2]),
        "point_count": observation.point_count,
        "component_pixel_count": observation.component_pixel_count,
        "seed_pixel_count": observation.seed_pixel_count,
        "range_seed_count": observation.range_seed_count,
        "free_space_seed_count": observation.free_space_seed_count,
        "foreground_support": observation.foreground_support,
        "history_support": observation.history_support,
        "position_covariance": observation.position_covariance,
        "extent": observation.extent,
        "pixel_bbox": observation.pixel_bbox,
        "measurement_valid": True,
    }


def track_row(track, managed, match_detail, created, deleted):
    return {
        "track_id": track.track_id,
        "created_this_frame": bool(created),
        "deleted_this_frame": bool(track.track_id in deleted),
        "confirmed": track.is_confirmed,
        "confirmation_hit_count": track.hit_count,
        "consecutive_hit_count": track.consecutive_direct_hits,
        "missed_count": track.missed_count,
        "track_age": track.age,
        "prediction_only_age": track.prediction_only_age,
        "visibility_state": track.visibility_state,
        "kalman_position": track.position_world,
        "kalman_velocity": track.velocity_world,
        "estimated_speed": float(np.linalg.norm(track.velocity_world)),
        "covariance": track.state_covariance,
        "innovation": (
            None if match_detail is None else match_detail["innovation"]
        ),
        "innovation_covariance": (
            None if match_detail is None
            else match_detail["innovation_covariance"]
        ),
        "association_cost": (
            None if match_detail is None
            else match_detail["association_mahalanobis_sq"]
        ),
        "association_distance": (
            None if match_detail is None
            else match_detail["association_distance"]
        ),
        "association_accepted": match_detail is not None,
        "dynamic_consistency_counter":
            managed.motion_consistency_count,
        "is_dynamic": track.is_dynamic,
        "dynamic_reason": track.dynamic_reason,
        "attention_authorized": track.attention_authorized,
        "confidence": track.confidence,
        "confidence_components": track.confidence_components,
        "birth_frame": track.birth_frame,
        "last_direct_observation_frame":
            track.last_direct_observation_frame,
    }


def trace_case(case, construction, uav, actor_frames, frame_times, sensor):
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(
        config, (3, 5), attention_device="cpu"
    )
    model = camera_model(sensor)
    diagnostics = construction.diagnostics
    actor = construction.actors[0]
    gt_velocity = np.asarray(actor["velocity"], dtype=np.float64)
    rows = []
    previous_matched_id = None
    for index, timestamp in enumerate(frame_times):
        result = perception.update_depth(
            np.asarray(
                diagnostics["composed_depth"][index], dtype=np.float32
            ),
            _pose(uav.position_world[index], uav.yaw[index], timestamp),
            float(timestamp), model,
        )
        manager_diagnostics = result.diagnostics["track_manager"]
        match_by_track = {
            int(item["track_id"]): item
            for item in manager_diagnostics["match_details"]
        }
        deleted = {
            int(value)
            for value in manager_diagnostics["deleted_track_ids"]
        }
        track_rows = []
        for track in result.all_tracks:
            managed = perception.track_manager._tracks[track.track_id]
            track_rows.append(track_row(
                track, managed, match_by_track.get(track.track_id),
                track.birth_frame == index, deleted,
            ))
        matched_track_id = None
        matched_error = None
        if result.all_tracks:
            distances = np.asarray([
                np.linalg.norm(
                    track.position_world - actor_frames[index, 0]
                )
                for track in result.all_tracks
            ])
            selected = int(np.argmin(distances))
            if distances[selected] <= 1.0:
                matched_track_id = int(
                    result.all_tracks[selected].track_id
                )
                matched_error = float(distances[selected])
        observation_match = None
        observation_error = None
        if result.observations:
            distances = np.asarray([
                np.linalg.norm(
                    observation.centroid_world - actor_frames[index, 0]
                )
                for observation in result.observations
            ])
            selected = int(np.argmin(distances))
            observation_error = float(distances[selected])
            if observation_error <= 1.0:
                observation_match = int(selected)
        projected = int(
            diagnostics["per_actor_projected_pixel_count"][index, 0]
        )
        visible = int(
            diagnostics["per_actor_visible_pixel_count"][index, 0]
        )
        blocked = int(
            diagnostics[
                "per_actor_static_blocked_pixel_count"
            ][index, 0]
        )
        outside = bool(
            diagnostics["per_actor_outside_fov"][index, 0]
        )
        behind = bool(
            diagnostics["per_actor_behind_camera"][index, 0]
        )
        beyond = bool(
            diagnostics["per_actor_beyond_max_depth"][index, 0]
        )
        foreground = result.diagnostics["foreground"]
        measurement_valid = observation_match is not None
        if measurement_valid:
            rejection = None
        elif visible == 0:
            rejection = "actor_not_directly_visible"
        elif foreground.get("component_count", 0) == 0:
            rejection = "no_foreground_component"
        elif not result.observations:
            rejection = "foreground_component_rejected"
        else:
            rejection = "observation_not_actor_consistent"
        replaced = bool(
            previous_matched_id is not None
            and matched_track_id is not None
            and matched_track_id != previous_matched_id
        )
        if matched_track_id is not None:
            previous_matched_id = matched_track_id
        row = {
            "case_id": case["case_id"],
            "frame_index": index,
            "timestamp": float(timestamp),
            "ground_truth": {
                "actor_position": actor_frames[index, 0],
                "actor_velocity": gt_velocity,
                "camera_position": uav.position_world[index],
                "camera_yaw": float(uav.yaw[index]),
                "actor_camera_distance": float(np.linalg.norm(
                    actor_frames[index, 0] - uav.position_world[index]
                )),
            },
            "sensor": {
                "projected_pixel_count": projected,
                "visible_pixel_count": visible,
                "static_blocked_pixel_count": blocked,
                "inside_fov": not outside,
                "in_front_of_camera": not behind,
                "within_max_depth": not beyond,
                "occlusion_reason": visibility_reason(
                    projected, visible, blocked,
                    outside, behind, beyond,
                ),
                "natural_occlusion_verified": bool(
                    projected > 0 and visible == 0
                    and blocked == projected and not (
                        outside or behind or beyond
                    )
                ),
            },
            "detection": {
                "foreground_pixel_count":
                    int(foreground.get("component_pixel_count", 0)),
                "foreground_connected_components":
                    int(foreground.get("component_count", 0)),
                "foreground_diagnostics": foreground,
                "detection_count": len(result.observations),
                "detections": [
                    observation_row(value)
                    for value in result.observations
                ],
                "matched_detection_index": observation_match,
                "matched_detection_error_m": observation_error,
                "measurement_valid": measurement_valid,
                "measurement_rejection_reason": rejection,
            },
            "track": {
                "all_track_ids": [
                    int(value.track_id) for value in result.all_tracks
                ],
                "matched_track_id": matched_track_id,
                "matched_gt_actor_id":
                    0 if matched_track_id is not None else None,
                "matched_position_error_m": matched_error,
                "created_track_ids": [
                    value["track_id"] for value in track_rows
                    if value["created_this_frame"]
                ],
                "deleted_track_ids": sorted(deleted),
                "replaced_this_frame": replaced,
                "tracks": track_rows,
                "association_cost_matrix":
                    manager_diagnostics["association_cost_matrix"],
                "association_track_ids":
                    manager_diagnostics["association_track_ids"],
                "association_observation_indices":
                    manager_diagnostics[
                        "association_observation_indices"
                    ],
                "association_distance_threshold":
                    manager_diagnostics[
                        "association_distance_threshold"
                    ],
                "association_mahalanobis_threshold":
                    manager_diagnostics[
                        "association_mahalanobis_threshold"
                    ],
                "lifecycle_update_order":
                    manager_diagnostics["lifecycle_update_order"],
            },
            "runtime_gt_actor_input_used": False,
        }
        rows.append(convert(row))
    return rows


def trace_summary(case, rows):
    detections = [
        row["frame_index"] for row in rows
        if row["detection"]["measurement_valid"]
    ]
    track_frames = [
        row["frame_index"] for row in rows
        if row["track"]["matched_track_id"] is not None
    ]
    confirmed = []
    dynamic = []
    direct = []
    deleted = []
    replaced = []
    for row in rows:
        track_id = row["track"]["matched_track_id"]
        matched = next((
            track for track in row["track"]["tracks"]
            if track["track_id"] == track_id
        ), None)
        if matched is not None:
            if matched["confirmed"]:
                confirmed.append(row["frame_index"])
            if matched["is_dynamic"]:
                dynamic.append(row["frame_index"])
            if matched["prediction_only_age"] == 0:
                direct.append(row["frame_index"])
        if row["track"]["deleted_track_ids"]:
            deleted.append(row["frame_index"])
        if row["track"]["replaced_this_frame"]:
            replaced.append(row["frame_index"])
    gap_start, gap_end = case["observed_gap"]
    return {
        "case_id": case["case_id"],
        "first_detection_frame": detections[0] if detections else None,
        "first_track_frame": track_frames[0] if track_frames else None,
        "first_confirmed_frame": confirmed[0] if confirmed else None,
        "first_dynamic_frame": dynamic[0] if dynamic else None,
        "last_direct_frame_before_gap": max(
            (frame for frame in direct if frame < gap_start),
            default=None,
        ),
        "gap_start": gap_start,
        "gap_end": gap_end,
        "deleted_frames": deleted,
        "replacement_frames": replaced,
        "measurement_valid_frames": detections,
        "confirmed_frames": confirmed,
        "dynamic_frames": dynamic,
        "trace_hash": canonical_hash(rows),
    }


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for retained replay")
    if MANIFEST.exists() or SUMMARY.exists() or TRACE_ROOT.exists():
        raise FileExistsError(
            "I1 retained evidence already exists; refusing overwrite"
        )
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
    frame_times = np.arange(60, dtype=np.float64) * .1
    map_rows = {
        row["map_uuid"]: row
        for row in json.loads(MAP_SET.read_text())["maps"]
    }
    cases = []
    summaries = []
    pending_traces = {}
    for source_path, source in retained_source_rows():
        map_row = map_rows[source["map_uuid"]]
        backend = ExactAuthorityBVH(map_row["authority_root"])
        gap = int(source["requested_gap_frames"])
        seed = (
            2_200_000_000 + int(source["seed"]) * 10 + gap
        ) % (2**32)
        rng = np.random.default_rng(seed)
        uav = sample_occlusion_uav_sequence_v2_2(
            backend, rng, frame_times, gap
        )
        construction = build_occluded_actor_specs_v2_2(
            backend, renderer, rng, uav.position_world, uav.yaw,
            frame_times, contract, gap, maximum_candidates=128,
        )
        observed = [construction.gap_start, construction.gap_end]
        if (
            observed != source["constructed_gap"]
            or construction.attempts != source["constructor_attempts"]
            or construction.evidence["authority_hash"]
                != source["authority_sightline_hash"]
        ):
            raise RuntimeError(
                f"retained replay mismatch for {source['map_uuid']} gap {gap}"
            )
        actor_frames = actor_position(
            construction.actors[0], frame_times
        )[:, None, :]
        identifier = case_id(source)
        actor = construction.actors[0]
        visibility_payload = {
            key: construction.diagnostics[key]
            for key in (
                "per_actor_projected_pixel_count",
                "per_actor_visible_pixel_count",
                "per_actor_static_blocked_pixel_count",
                "per_actor_outside_fov",
                "per_actor_behind_camera",
                "per_actor_beyond_max_depth",
            )
        }
        constructor_payload = {
            "actors": construction.actors,
            "evidence": construction.evidence,
            "attempts": construction.attempts,
            "requested_gap_frames": construction.requested_gap_frames,
            "gap_start": construction.gap_start,
            "gap_end": construction.gap_end,
            "method": construction.method,
        }
        certificate_payload = {
            "uav_reference": asdict(uav.reference_certificate),
            "authority_sightline": construction.evidence,
        }
        original_occlusion = source["frozen_perception"]["occlusion"]
        case = {
            "case_id": identifier,
            "source_report": str(source_path.relative_to(ROOT)),
            "source_report_hash": sha256(source_path),
            "maze_type": source["maze_type"],
            "semantic_map_type": source["natural_type"],
            "map_uuid": source["map_uuid"],
            "map_seed": source["seed"],
            "authority_hash": map_row["authority_manifest_hash"],
            "occupancy_hash": map_row["occupancy_hash"],
            "requested_gap": gap,
            "observed_gap": observed,
            "actor_speed_mps": float(np.linalg.norm(actor["velocity"])),
            "actor_velocity_world": actor["velocity"],
            "actor_radius_m": actor["radius_m"],
            "camera_trajectory": uav.position_world,
            "camera_yaw": uav.yaw,
            "actor_trajectory": actor_frames[:, 0],
            "sequence_timestamps": frame_times,
            "cuda_visibility_hash": canonical_hash(visibility_payload),
            "constructor_result_hash": canonical_hash(
                constructor_payload
            ),
            "geometry_certificate_hash": canonical_hash(
                certificate_payload
            ),
            "constructor_attempts": construction.attempts,
            "original_failure": {
                "identity_status": source["identity_status"],
                "pre_gap_confirmed_dynamic":
                    original_occlusion.get(
                        "pre_gap_confirmed_dynamic", False
                    ),
                "track_deleted":
                    original_occlusion.get("track_deleted", False),
                "replacement_track_created":
                    original_occlusion.get(
                        "replacement_track_created", False
                    ),
                "identity_continuous":
                    original_occlusion.get(
                        "identity_continuous", False
                    ),
            },
            "annex_used": False,
            "replay_exact": True,
        }
        rows = trace_case(
            case, construction, uav, actor_frames, frame_times, sensor
        )
        cases.append(case)
        summaries.append(trace_summary(case, rows))
        pending_traces[identifier] = rows
        print(json.dumps({
            "case_id": identifier,
            "gap": gap,
            "observed": observed,
            "trace_frames": len(rows),
        }), flush=True)
    independent_maps = len({case["map_uuid"] for case in cases})
    manifest = {
        "status": "PASS",
        "manifest_version": "phase8jqv2_4i1_retained_natural_cases_v1",
        "constructor_version": CONSTRUCTOR_VERSION,
        "v2_1_hash": sha256(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py"
        ),
        "v2_2_hash": sha256(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
        ),
        "case_count": len(cases),
        "independent_map_count": independent_maps,
        "independent_seed_count": len({
            case["map_seed"] for case in cases
        }),
        "gap_counts": {
            str(gap): sum(case["requested_gap"] == gap for case in cases)
            for gap in (1, 2, 3)
        },
        "semantic_distribution": {
            name: sum(
                case["semantic_map_type"] == name for case in cases
            )
            for name in ("cave", "forest", "room")
        },
        "three_map_gate_eligible": independent_maps >= 3,
        "cases": cases,
        "new_map_sweep_executed": False,
        "annex_used": False,
        "test_accessed": False,
        "blind_accessed": False,
        "formal_generation_started": False,
        "training_executed": False,
    }
    TRACE_ROOT.mkdir(parents=True, exist_ok=False)
    for identifier, rows in pending_traces.items():
        path = TRACE_ROOT / f"{identifier}.jsonl"
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(
                    row, sort_keys=True, allow_nan=False
                ) + "\n")
    atomic_new_json(MANIFEST, manifest)
    atomic_new_json(SUMMARY, {
        "status": "PASS",
        "trace_version": "phase8jqv2_4i1_causal_trace_v1",
        "case_count": len(summaries),
        "frames_per_case": 60,
        "persistent_perception_instance_per_case": True,
        "runtime_gt_actor_input_used": False,
        "summaries": summaries,
    })
    print(json.dumps({
        "status": "PASS", "cases": len(cases),
        "independent_maps": independent_maps,
        "gaps": manifest["gap_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
