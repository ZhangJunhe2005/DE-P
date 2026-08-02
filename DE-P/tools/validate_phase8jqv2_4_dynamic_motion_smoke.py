#!/usr/bin/env python3
"""Validate isolated V3 motion smoke through frozen causal perception."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.dynamic_motion_v2 import load_motion_contract
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, Pose


DATASET = ROOT/"artifacts/phase8jqv2_4_dynamic_motion_smoke"
CONTRACT_PATH = ROOT/"configs/authoritative_dynamic_motion_contract_v2.yaml"
V1_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
V2_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
CAMERA_BODY_FROM_OPTICAL = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(name, value):
    path = ROOT/"reports"/name
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in (
            "minimum", "q05", "q10", "median", "q90", "q95",
            "maximum", "mean",
        )}
    return {
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, .05)),
        "q10": float(np.quantile(values, .10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, .90)),
        "q95": float(np.quantile(values, .95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def perception_config():
    return replace(
        DynamicPerceptionConfig.from_global_config(),
        enabled=True,
        foreground_mode="range_image_hybrid",
        camera_width=160, camera_height=96,
        camera_fx=80.0, camera_fy=80.0,
        camera_cx=80.0, camera_cy=45.0,
    )


def camera_model():
    return CameraModel(
        width=160, height=96, fx=80.0, fy=80.0, cx=80.0, cy=45.0,
        depth_scale=1.0, min_depth=0.1, max_depth=20.0,
    )


def pose(frame):
    w, x, y, z = frame["quaternion_world_from_body"]
    body = Rotation.from_quat([x, y, z, w]).as_matrix()
    timestamp = frame["timestamp_ns"]/1e9
    return Pose(
        np.asarray(frame["position_world"], dtype=np.float64),
        body @ CAMERA_BODY_FROM_OPTICAL,
        timestamp,
    )


def main():
    contract = load_motion_contract(CONTRACT_PATH)
    manifest_path = DATASET/"manifests/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sequence_manifests = [
        json.loads((DATASET/row["path"]).read_text())
        for row in manifest["sequences"]
    ]
    config = perception_config()
    model = camera_model()
    scenarios = {}
    total_actor_count = 0
    zero_motion_actor_count = 0
    dynamic_track_total = 0
    attention_frames = 0
    no_target_false_attention = 0
    actor_static_collision = 0
    actor_actor_collision = 0
    dynamic_joint_unsafe = 0
    pose_mismatch = 0

    for sequence in sequence_manifests:
        base = (
            DATASET/sequence["suite"]/sequence["split"]
            /sequence["sequence_id"]
        )
        frames = [
            json.loads(line)
            for line in (base/"frames.jsonl").read_text().splitlines()
        ]
        certificates = [
            json.loads(line)
            for line in (base/"certificates.jsonl").read_text().splitlines()
        ]
        depth = np.load(base/"depth.npy", mmap_mode="r", allow_pickle=False)
        render = json.loads((base/"render_diagnostics.json").read_text())
        scenario = sequence["scenario"]
        row = scenarios.setdefault(scenario, {
            "sequence_count": 0, "actor_count": 0,
            "configured_speeds_mps": [], "persisted_speeds_mps": [],
            "perception_estimated_speeds_mps": [],
            "zero_motion_count": 0, "actors_entered_dynamic": 0,
            "dynamic_enter_events": 0, "dynamic_exit_events": 0,
            "dynamic_attention_frames": 0, "visible_frames": 0,
            "tracked_frames": 0, "minimum_actor_uav_separation_m": float("inf"),
            "minimum_actor_actor_separation_m": float("inf"),
            "unsafe_frames": 0, "renderer": render["renderer_version"],
            "fallback_sequence_count": 0,
            "independent_any_dynamic_match_count": 0,
        })
        row["sequence_count"] += 1
        actors_by_id = {}
        for frame in frames:
            for actor in frame["actor_metadata"]:
                actors_by_id.setdefault(actor["actor_id"], []).append(actor)
        row["actor_count"] += len(actors_by_id)
        total_actor_count += len(actors_by_id)
        stored_probe = render.get("frozen_perception_probe")
        if stored_probe is not None:
            row["actors_entered_dynamic"] += int(
                stored_probe["actors_entered_dynamic"]
            )
        entered_actor_ids = set()
        previous_dynamic_ids = set()
        perception = DynamicPerception(config, (3, 5), attention_device="cpu")
        for actor_id, actor_frames in actors_by_id.items():
            positions = np.asarray([
                item["position_world"] for item in actor_frames
            ], dtype=np.float64)
            persisted = np.asarray([
                item["velocity_world"] for item in actor_frames
            ], dtype=np.float64)
            dt = np.diff([
                frame["timestamp_ns"]/1e9 for frame in frames
            ])
            finite = np.diff(positions, axis=0)/dt[:, None]
            configured = float(actor_frames[0]["configured_speed_mps"])
            row["configured_speeds_mps"].append(configured)
            row["persisted_speeds_mps"].extend(
                np.linalg.norm(finite, axis=1).tolist()
            )
            if float(np.max(np.linalg.norm(finite, axis=1))) <= 1e-9:
                row["zero_motion_count"] += 1
                zero_motion_actor_count += 1
            if actor_frames[0]["sampling_method"].startswith("deterministic"):
                row["fallback_sequence_count"] += 1
            if not np.allclose(finite, persisted[:-1], atol=1e-5):
                pose_mismatch += 1
        for index, frame in enumerate(frames):
            actors = frame["actor_metadata"]
            uav = np.asarray(frame["position_world"], dtype=np.float64)
            if actors:
                positions = np.asarray([
                    actor["position_world"] for actor in actors
                ], dtype=np.float64)
                radii = np.asarray([
                    actor["radius_m"] for actor in actors
                ], dtype=np.float64)
                row["minimum_actor_uav_separation_m"] = min(
                    row["minimum_actor_uav_separation_m"],
                    float(np.min(np.linalg.norm(positions-uav, axis=1)-radii-.3)),
                )
                if len(positions) > 1:
                    separation = (
                        np.linalg.norm(positions[0]-positions[1])
                        - radii[0]-radii[1]
                    )
                    row["minimum_actor_actor_separation_m"] = min(
                        row["minimum_actor_actor_separation_m"],
                        float(separation),
                    )
                    actor_actor_collision += int(separation <= 0)
                actor_static_collision += sum(
                    int(actor["static_collision"]
                        or actor["future_static_collision"])
                    for actor in actors
                )
            safe = bool(certificates[index]["dynamic_joint_safe"])
            dynamic_joint_unsafe += int(not safe)
            row["unsafe_frames"] += int(not safe)
            row["visible_frames"] += int(frame["actor_depth_pixel_count"] > 0)
            timestamp = frame["timestamp_ns"]/1e9
            result = perception.update_depth(
                np.asarray(depth[index], dtype=np.float32),
                pose(frame), timestamp, model,
            )
            dynamic_ids = {track.track_id for track in result.dynamic_tracks}
            dynamic_track_total += len(result.dynamic_tracks)
            row["tracked_frames"] += int(bool(result.all_tracks))
            row["dynamic_enter_events"] += len(dynamic_ids-previous_dynamic_ids)
            row["dynamic_exit_events"] += len(previous_dynamic_ids-dynamic_ids)
            previous_dynamic_ids = dynamic_ids
            nonzero = int(torch.count_nonzero(result.attention_map)) > 0
            attention_frames += int(nonzero)
            row["dynamic_attention_frames"] += int(nonzero)
            if scenario == "no_target":
                no_target_false_attention += int(nonzero)
            if actors and result.dynamic_tracks:
                gt = np.asarray([
                    actor["position_world"] for actor in actors
                ], dtype=np.float64)
                for track in result.dynamic_tracks:
                    distance = np.linalg.norm(gt-track.position_world, axis=1)
                    actor_id = actors[int(np.argmin(distance))]["actor_id"]
                    if float(np.min(distance)) <= 1.0:
                        entered_actor_ids.add(actor_id)
                        row["perception_estimated_speeds_mps"].append(
                            float(np.linalg.norm(track.velocity_world))
                        )
        row["independent_any_dynamic_match_count"] += len(entered_actor_ids)

    gate_checks = {
        "all_actor_scenarios_present":
            set(contract["scenarios"]).issubset(scenarios),
        "zero_motion_actor_count_zero": zero_motion_actor_count == 0,
        "every_actor_enters_dynamic": all(
            row["actors_entered_dynamic"] == row["actor_count"]
            for name, row in scenarios.items() if name != "no_target"
        ),
        "attention_nonempty": attention_frames > 0,
        "no_target_false_attention_zero": no_target_false_attention == 0,
        "actor_static_collision_zero": actor_static_collision == 0,
        "actor_actor_collision_zero": actor_actor_collision == 0,
        "dynamic_joint_unsafe_zero": dynamic_joint_unsafe == 0,
        "renderer_pose_matches_persisted": pose_mismatch == 0,
        "canonical_cuda_renderer": all(
            row["renderer"] == "canonical_occupancy_cuda_raycast_v1"
            for row in scenarios.values()
        ),
        "v1_unchanged": sha256(
            ROOT/"data/phase8_authoritative_v1/manifests/dataset_manifest.json"
        ) == V1_HASH,
        "v2_unchanged": sha256(
            ROOT/"data/phase8_authoritative_v2/manifests/dataset_manifest.json"
        ) == V2_HASH,
    }
    for row in scenarios.values():
        row["configured_speed_distribution_mps"] = distribution(
            row.pop("configured_speeds_mps")
        )
        row["persisted_speed_distribution_mps"] = distribution(
            row.pop("persisted_speeds_mps")
        )
        row["perception_estimated_speed_distribution_mps"] = distribution(
            row.pop("perception_estimated_speeds_mps")
        )
        for key in (
            "minimum_actor_uav_separation_m",
            "minimum_actor_actor_separation_m",
        ):
            if not np.isfinite(row[key]):
                row[key] = None
        row["dynamic_enter_fraction"] = (
            row["actors_entered_dynamic"]/row["actor_count"]
            if row["actor_count"] else 1.0
        )
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    common = {
        "status": status,
        "dataset_version": manifest["dataset_version"],
        "smoke_root_manifest_hash": sha256(manifest_path),
        "motion_contract_version": contract["contract_version"],
        "motion_contract_hash": contract["_file_hash"],
        "frozen_perception_config": asdict(config),
        "scenario_results": scenarios,
        "checks": gate_checks,
        "production_test_used": False,
        "blind_used": False,
        "formal_dataset_modified": False,
    }
    write_report(
        "phase8jqv2_4_dynamic_motion_smoke_generation.json",
        {
            **common,
            "sequence_count": len(sequence_manifests),
            "frame_count": sum(row["frame_count"] for row in sequence_manifests),
            "renderer": "canonical_occupancy_cuda_raycast_v1",
        },
    )
    write_report(
        "phase8jqv2_4_dynamic_motion_smoke_perception.json",
        {
            **common,
            "dynamic_track_total": dynamic_track_total,
            "nonzero_attention_frame_count": attention_frames,
            "no_target_false_attention_count": no_target_false_attention,
        },
    )
    write_report(
        "phase8jqv2_4_dynamic_motion_smoke_safety.json",
        {
            **common,
            "actor_static_collision_count": actor_static_collision,
            "actor_actor_collision_count": actor_actor_collision,
            "dynamic_joint_unsafe_frame_count": dynamic_joint_unsafe,
        },
    )
    write_report(
        "phase8jqv2_4_dynamic_motion_smoke_validation.json", common
    )
    print(json.dumps({
        "status": status,
        "checks": gate_checks,
        "dynamic_track_total": dynamic_track_total,
        "attention_frames": attention_frames,
        "no_target_false_attention": no_target_false_attention,
    }, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
