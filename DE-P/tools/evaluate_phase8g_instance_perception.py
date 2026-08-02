#!/usr/bin/env python3
"""Evaluate a Phase 8G split with instance IDs kept outside runtime."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.instance_evaluation import InstanceEvaluationState
from policy.dynamic.types import DynamicPerceptionConfig
from tools.evaluate_phase8c_estimated_context import camera, load_yaml, pose


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 1.0


def attention_iou(attention, instance):
    height, width = attention.shape
    truth = np.zeros((height, width), dtype=bool)
    actor_pixels = instance != 0
    for row in range(height):
        for column in range(width):
            v0, v1 = row * instance.shape[0] // height, (row + 1) * instance.shape[0] // height
            u0, u1 = column * instance.shape[1] // width, (column + 1) * instance.shape[1] // width
            truth[row, column] = bool(actor_pixels[v0:v1, u0:u1].any())
    predicted = attention > 0.05
    union = int(np.count_nonzero(predicted | truth))
    return float(np.count_nonzero(predicted & truth) / union) if union else 1.0


def evaluate(dataset, split):
    root = Path(dataset).resolve()
    sequences = [
        value.strip() for value in
        (root / "splits" / f"{split}.txt").read_text().splitlines()
        if value.strip()
    ]
    config = replace(
        DynamicPerceptionConfig.from_global_config(),
        foreground_mode="range_image_hybrid",
    )
    totals = {
        "visible_tp": 0, "visible_fp": 0, "active_tp": 0, "active_fp": 0,
        "active_gt": 0, "no_target_frames": 0, "no_target_fp_frames": 0,
        "no_target_attention_frames": 0, "prediction_only_confirmed_track_count": 0,
        "prediction_only_attention_count": 0,
        "occluded_eligible_frames": 0, "occluded_retained_frames": 0,
        "occluded_predicted_frames": 0, "occluded_correct_frames": 0,
    }
    states, errors, velocity_errors, latencies, ious = [], [], [], [], []
    observed_actor_ids, birth_actor_ids = set(), set()
    birth_features = {"grounded": [], "ungrounded": []}
    alignment_error = 0
    negative_visibility_ages = []
    all_events = []
    for sequence in sequences:
        directory = root / "sequences" / sequence
        metadata = load_yaml(directory / "metadata.yaml")
        if metadata.get("instance_runtime_input") is not False:
            raise ValueError("dataset must declare instance_runtime_input=false")
        model = camera(metadata)
        perception = DynamicPerception(
            config, (cfg["vertical_num"], cfg["horizon_num"])
        )
        state = InstanceEvaluationState(occlusion_window=config.max_missed_frames)
        frames = list(csv.DictReader(
            (directory / "frames.csv").open(newline="", encoding="utf-8")
        ))
        ever_visible_ids = set()
        for frame in frames:
            preview = np.load(
                directory / frame["instance_path"], allow_pickle=False
            )
            ever_visible_ids.update(int(x) for x in np.unique(preview) if int(x))
        never_observed_ids = set()
        if frames:
            for frame in frames:
                preview_objects = json.loads(
                    (directory / frame["dynamic_objects_path"]).read_text()
                )
                never_observed_ids.update(
                    int(obj["object_id"]) for obj in preview_objects
                    if int(obj["object_id"]) not in ever_visible_ids
                )
        for frame_number, frame in enumerate(frames):
            depth = np.load(directory / frame["depth_path"], allow_pickle=False)
            # Runtime completes before evaluator reads the instance array.
            started = time.perf_counter()
            result = perception.update_depth(
                depth, pose(frame, metadata), float(frame["timestamp"]), model
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            instance = np.load(
                directory / frame["instance_path"], allow_pickle=False
            )
            objects = json.loads(
                (directory / frame["dynamic_objects_path"]).read_text()
            )
            state.update(
                sequence, frame_number, objects, instance, result,
                never_observed_actor_ids=never_observed_ids,
            )
            object_by_id = {int(obj["object_id"]): obj for obj in objects}
            observation_by_id = {
                int(obs.observation_id): obs for obs in result.observations
            }
            visible_ids = {
                actor_id for actor_id in object_by_id
                if np.any(instance == actor_id)
            }
            observed_actor_ids.update((sequence, actor_id) for actor_id in visible_ids)
            totals["active_gt"] += sum(
                bool(obj.get("active", True))
                and int(obj["object_id"]) in state.observed_actor_last_frame
                and frame_number - state.observed_actor_last_frame[
                    int(obj["object_id"])
                ] <= config.max_missed_frames
                for obj in objects
            )
            current_tracks = {track.track_id: track for track in result.all_tracks}
            for actor_id, obj in object_by_id.items():
                occluded_eligible = (
                    bool(obj.get("active", True))
                    and actor_id not in visible_ids
                    and actor_id in state.observed_actor_last_frame
                    and frame_number - state.observed_actor_last_frame[actor_id]
                    <= config.max_missed_frames
                )
                if occluded_eligible:
                    totals["occluded_eligible_frames"] += 1
                    retained = any(
                        identity == actor_id and track_id in current_tracks
                        for track_id, identity in state.track_actor_identity.items()
                    )
                    totals["occluded_retained_frames"] += int(retained)
            for track in result.all_tracks:
                actor_id = state.track_actor_identity.get(track.track_id)
                if actor_id is not None and actor_id not in visible_ids:
                    totals["occluded_predicted_frames"] += 1
                    totals["occluded_correct_frames"] += int(
                        actor_id in object_by_id
                        and bool(object_by_id[actor_id].get("active", True))
                    )
                if track.visibility_state == "clear_missing":
                    negative_visibility_ages.append(track.prediction_only_age)
            projected_ids = {x.track_id for x in result.projected_dynamic_tracks}
            for track in result.dynamic_tracks:
                actor_id = state.track_actor_identity.get(track.track_id)
                matched = actor_id in object_by_id
                totals["active_tp" if matched else "active_fp"] += 1
                visible_match = matched and actor_id in visible_ids and track.track_id in projected_ids
                totals["visible_tp" if visible_match else "visible_fp"] += 1
                if matched:
                    obj = object_by_id[actor_id]
                    errors.append(float(np.linalg.norm(
                        track.position_world - np.asarray(obj["position_world"])
                    )))
                    velocity_errors.append(float(np.linalg.norm(
                        track.velocity_world - np.asarray(obj["velocity_world"])
                    )))
            for track in result.all_tracks:
                if track.birth_frame != frame_number:
                    continue
                obs = observation_by_id.get(int(track.birth_observation_id))
                if obs is None:
                    continue
                indices = np.asarray(obs.pixel_indices, dtype=np.int64)
                grounded = bool(
                    len(indices)
                    and np.any(instance.reshape(-1)[indices] != 0)
                )
                birth_features["grounded" if grounded else "ungrounded"].append({
                    "foreground_support": float(obs.foreground_support),
                    "point_count": int(obs.point_count),
                    "seed_pixel_count": int(obs.seed_pixel_count),
                    "range_seed_count": int(obs.range_seed_count),
                    "free_space_seed_count": int(obs.free_space_seed_count),
                    "component_pixel_count": int(obs.component_pixel_count),
                    "history_support": float(obs.history_support),
                    "depth_consistency": float(obs.depth_consistency),
                })
            for event in state.events:
                if (event["classification_reason"] == "direct_instance_overlap"
                        and event["birth_frame"] == frame_number):
                    birth_actor_ids.add((sequence, event["actor_id"]))
            no_target = not any(bool(obj.get("active", True)) for obj in objects)
            if no_target:
                totals["no_target_frames"] += 1
                totals["no_target_fp_frames"] += int(bool(result.dynamic_tracks))
                totals["no_target_attention_frames"] += int(
                    float(result.attention_map.max()) > 0.0
                )
            ious.append(attention_iou(
                result.attention_map[0, 0].detach().cpu().numpy(), instance
            ))
            for obj in objects:
                error = abs(
                    int(np.count_nonzero(instance == int(obj["object_id"])))
                    - int(obj["rendered_pixel_count"])
                )
                alignment_error = max(alignment_error, error)
        summary = state.summary()
        totals["prediction_only_confirmed_track_count"] += summary[
            "prediction_only_confirmed_track_count"
        ]
        totals["prediction_only_attention_count"] += summary[
            "prediction_only_attention_count"
        ]
        all_events.extend(summary["events"])
        states.append(summary)

    def sum_state(name):
        return int(sum(item.get(name, 0) for item in states))

    visible_actor_frames = sum_state("visible_actor_frames")
    visible_matches = sum_state("visible_actor_matches")
    payload = {
        "status": "PASS",
        "evaluator_version": "phase8g_instance_provenance_v1",
        "dataset": str(root), "split": split, "sequence_count": len(sequences),
        "causal": True, "instance_mask_runtime_input": False,
        "foreground_mode": "range_image_hybrid",
        "visible_instance_precision": safe_ratio(
            totals["visible_tp"], totals["visible_tp"] + totals["visible_fp"]
        ),
        "visible_instance_recall": safe_ratio(visible_matches, visible_actor_frames),
        "active_instance_precision": safe_ratio(
            totals["active_tp"], totals["active_tp"] + totals["active_fp"]
        ),
        "active_instance_recall": safe_ratio(totals["active_tp"], totals["active_gt"]),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0,
        "velocity_rmse_mps": (
            float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else 0.0
        ),
        "no_target_false_positive_frame_fraction": safe_ratio(
            totals["no_target_fp_frames"], totals["no_target_frames"]
        ),
        "no_target_attention_nonzero_frame_fraction": safe_ratio(
            totals["no_target_attention_frames"], totals["no_target_frames"]
        ),
        "attention_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "complete_perception_latency_ms": {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "never_observed_proximity_event_count": sum_state(
            "never_observed_proximity_event_count"
        ),
        "never_observed_illegal_observation_count": sum_state(
            "never_observed_illegal_observation_count"
        ),
        "never_observed_ungrounded_track_count": sum_state(
            "never_observed_ungrounded_track_count"
        ),
        "never_observed_attention_event_count": sum_state(
            "never_observed_attention_event_count"
        ),
        "prediction_only_confirmed_track_count": totals[
            "prediction_only_confirmed_track_count"
        ],
        "prediction_only_attention_count": totals["prediction_only_attention_count"],
        "track_birth_instance_precision": safe_ratio(
            sum_state("grounded_track_births"), sum_state("track_births")
        ),
        "track_birth_instance_recall": safe_ratio(
            len(birth_actor_ids), len(observed_actor_ids)
        ),
        "confirmed_track_instance_precision": safe_ratio(
            sum_state("grounded_confirmed_tracks"), sum_state("confirmed_tracks")
        ),
        "attention_instance_precision": safe_ratio(
            sum_state("grounded_attention_tracks"), sum_state("attention_tracks")
        ),
        "occluded_retention_precision": safe_ratio(
            totals["occluded_correct_frames"], totals["occluded_predicted_frames"]
        ),
        "occluded_retention_recall": safe_ratio(
            totals["occluded_retained_frames"], totals["occluded_eligible_frames"]
        ),
        "negative_visibility_deletion_delay": {
            "mean_frames": float(np.mean(negative_visibility_ages))
                if negative_visibility_ages else 0.0,
            "max_frames": int(max(negative_visibility_ages, default=0)),
        },
        "instance_mask_depth_alignment_error": int(alignment_error),
        "frame_record_count": sum_state("frame_record_count"),
        "unique_track_actor_event_count": len({
            (event["track_id"], event["actor_id"], event["classification_reason"])
            for event in all_events
        }),
        "events": all_events,
    }
    payload["track_birth_feature_analysis"] = {
        label: {
            "count": len(rows),
            "quantiles": {
                key: {
                    "p10": float(np.percentile([row[key] for row in rows], 10)),
                    "p50": float(np.percentile([row[key] for row in rows], 50)),
                    "p90": float(np.percentile([row[key] for row in rows], 90)),
                }
                for key in rows[0]
            } if rows else {},
        }
        for label, rows in birth_features.items()
    }
    checks = {
        "visible_instance_precision": (0.5, "min"),
        "visible_instance_recall": (0.2, "min"),
        "active_instance_precision": (0.5, "min"),
        "active_instance_recall": (0.2, "min"),
        "position_rmse_m": (1.0, "max"),
        "velocity_rmse_mps": (1.5, "max"),
        "no_target_false_positive_frame_fraction": (0.1, "max"),
        "no_target_attention_nonzero_frame_fraction": (0.05, "max"),
        "attention_iou_mean": (0.05, "min"),
        "track_birth_instance_precision": (0.5, "min"),
        "confirmed_track_instance_precision": (0.5, "min"),
        "attention_instance_precision": (0.5, "min"),
    }
    failures = []
    for name, (threshold, direction) in checks.items():
        value = payload[name]
        if value is None or (value < threshold if direction == "min" else value > threshold):
            failures.append(f"{name}={value} violates {direction} {threshold}")
    for name in (
        "never_observed_illegal_observation_count",
        "never_observed_ungrounded_track_count",
        "never_observed_attention_event_count",
        "prediction_only_confirmed_track_count",
        "prediction_only_attention_count",
        "instance_mask_depth_alignment_error",
    ):
        if payload[name] != 0:
            failures.append(f"{name}={payload[name]} must be 0")
    payload["hard_failures"] = failures
    payload["status"] = "PASS" if not failures else "FAIL"
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    display = dict(result)
    display["events"] = f"{len(result['events'])} records saved in report"
    print(json.dumps(display, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
