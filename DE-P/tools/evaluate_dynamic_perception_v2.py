#!/usr/bin/env python3
"""Phase 8E causal perception metrics with explicit occlusion semantics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import DynamicPerceptionConfig
from policy.dynamic_sequence_dataset import validate_dataset_splits
from tools.evaluate_phase8c_estimated_context import (
    attention_metrics, camera, gt_attention, load_yaml, pose,
)
from tools.analyze_causal_range_residuals import actor_proxy_mask


def visible(obj):
    return (bool(obj.get("active", True)) and bool(obj.get("inside_image", True))
            and not bool(obj.get("occluded", False)) and float(obj["visibility"]) > 0)


def active(obj):
    return bool(obj.get("active", True))


def gated_matches(objects, tracks, threshold):
    if not objects or not tracks:
        return []
    distances = np.asarray([[
        np.linalg.norm(np.asarray(obj["position_world"], dtype=float) - track.position_world)
        for track in tracks
    ] for obj in objects])
    rows, columns = linear_sum_assignment(distances)
    return [(int(row), int(column), float(distances[row, column]))
            for row, column in zip(rows, columns)
            if distances[row, column] <= threshold]


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def unmatched_by_index(values, matched_indices):
    return [value for index, value in enumerate(values) if index not in matched_indices]


def bin_value(value, edges, labels):
    return labels[int(np.searchsorted(edges, value, side="right"))]


def evaluate_split(root, split, sequences, config):
    counts = defaultdict(float)
    position_errors, velocity_errors = [], []
    attention_ious, center_errors, confirmation_times = [], [], []
    attention_values, false_attention_cells = [], []
    per_scenario = defaultdict(lambda: defaultdict(float))
    strata = defaultdict(lambda: defaultdict(lambda: np.zeros(2, dtype=np.int64)))
    pixel_counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    foreground_times, perception_times = [], []
    seed_delays, component_delays = [], []
    never_observed_false_match_details = []
    for sequence in sequences:
        directory = root / "sequences" / sequence
        metadata = load_yaml(directory / "metadata.yaml")
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        model = camera(metadata)
        perception = DynamicPerception(config, (cfg["vertical_num"], cfg["horizon_num"]))
        scenario = str(metadata.get("scenario_type", "unknown"))
        no_target = scenario == "no_target"
        last_visible_frame, first_visible_time, first_match_time = {}, {}, {}
        first_visible_frame = {}
        first_seed_time, first_component_time = {}, {}
        previous_assignment, ever_matched, previous_matched = {}, set(), {}
        previous_frame_pose = None
        for frame_position, frame in enumerate(frames):
            depth = np.load(directory / frame["depth_path"], allow_pickle=False)
            frame_pose = pose(frame, metadata)
            perception_started = time.perf_counter()
            result = perception.update_depth(
                depth, frame_pose, float(frame["timestamp"]), model,
            )
            perception_times.append((time.perf_counter() - perception_started) * 1000.0)
            foreground_times.append(float(
                result.diagnostics.get("foreground", {}).get("foreground_ms", 0.0)
            ))
            objects = json.loads((directory / frame["dynamic_objects_path"]).read_text())
            truth_pixels = actor_proxy_mask(objects, np.asarray(depth, np.float32), model)
            if config.foreground_mode == "range_image_hybrid":
                for stage, mask in {
                    "free_space_seed": perception.range_foreground.last_free_seed,
                    "range_residual_seed": perception.range_foreground.last_range_seed,
                    "completed_component": perception.range_foreground.last_component_mask,
                }.items():
                    mask = np.asarray(mask, dtype=bool)
                    pixel_counts[stage] += np.asarray((
                        np.sum(mask & truth_pixels), np.sum(mask & ~truth_pixels),
                        np.sum(~mask & truth_pixels),
                    ), dtype=np.int64)
            visible_objects = [obj for obj in objects if visible(obj)]
            for obj in visible_objects:
                actor_id = int(obj["object_id"])
                last_visible_frame[actor_id] = frame_position
                first_visible_time.setdefault(actor_id, float(frame["timestamp"]))
                first_visible_frame.setdefault(actor_id, frame_position)
                if config.foreground_mode == "range_image_hybrid":
                    actor_pixels = actor_proxy_mask([obj], np.asarray(depth, np.float32), model)
                    if np.any(perception.range_foreground.last_range_seed & actor_pixels):
                        first_seed_time.setdefault(actor_id, float(frame["timestamp"]))
                    if np.any(perception.range_foreground.last_component_mask & actor_pixels):
                        first_component_time.setdefault(actor_id, float(frame["timestamp"]))
            eligible = [obj for obj in objects if active(obj) and (
                visible(obj) or (
                    int(obj["object_id"]) in last_visible_frame
                    and frame_position - last_visible_frame[int(obj["object_id"])]
                    <= config.max_missed_frames
                )
            )]
            never_observed = [obj for obj in objects if active(obj)
                              and int(obj["object_id"]) not in last_visible_frame]
            tracks = list(result.dynamic_tracks)
            matches = gated_matches(eligible, tracks, config.association_distance_threshold)
            matched_object_indices = {row for row, _column, _distance in matches}
            matched_track_indices = {column for _row, column, _distance in matches}
            projected_ids = {
                projected.track_id for projected in result.projected_dynamic_tracks
            }
            projected_tracks = [track for track in tracks if track.track_id in projected_ids]
            visible_matches = gated_matches(
                visible_objects, projected_tracks, config.association_distance_threshold
            )
            visible_matched_track_ids = {
                projected_tracks[column].track_id
                for _row, column, _distance in visible_matches
            }
            retained_occluded_track_ids = {
                tracks[column].track_id
                for row, column, _distance in matches if not visible(eligible[row])
            }
            visible_tp = len(visible_matches)
            visible_fn = len(visible_objects) - visible_tp
            # Out-of-image tracks are not current visible detections. Correctly
            # retained occluded tracks are also excluded, but remain in active metrics.
            visible_fp = sum(
                track.track_id not in visible_matched_track_ids
                and track.track_id not in retained_occluded_track_ids
                for track in projected_tracks
            )
            counts["visible_tp"] += visible_tp
            counts["visible_fp"] += visible_fp
            counts["visible_fn"] += visible_fn
            counts["active_tp"] += len(matches)
            counts["active_fp"] += len(tracks) - len(matches)
            counts["active_fn"] += len(eligible) - len(matches)
            counts["frames"] += 1
            counts["visible_gt"] += len(visible_objects)
            counts["active_eligible_gt"] += len(eligible)
            counts["estimated_dynamic"] += len(tracks)
            camera_translation = (0.0 if previous_frame_pose is None else float(np.linalg.norm(
                frame_pose.position_world - previous_frame_pose.position_world
            )))
            camera_rotation = (0.0 if previous_frame_pose is None else float(np.arccos(np.clip(
                (np.trace(previous_frame_pose.rotation_world_from_camera.T
                          @ frame_pose.rotation_world_from_camera) - 1) / 2, -1, 1
            ))))
            for object_index, obj in enumerate(eligible):
                position = np.asarray(obj["position_world"], dtype=float)
                velocity = np.asarray(obj["velocity_world"], dtype=float)
                line = position - frame_pose.position_world
                distance = float(np.linalg.norm(line)); speed = float(np.linalg.norm(velocity))
                radial = abs(float(np.dot(velocity, line / max(distance, 1e-9))))
                lateral = float(np.sqrt(max(speed * speed - radial * radial, 0.0)))
                values = {
                    "scenario": scenario, "map_id": str(frame["map_id"]),
                    "actor_shape": str(obj.get("type", "unknown")),
                    "distance_bin": bin_value(distance, (4, 8, 12), ("0-4", "4-8", "8-12", "12+")),
                    "pixel_area_bin": bin_value(float(obj.get("rendered_pixel_count", 0)),
                                                (25, 100, 400), ("0-25", "25-100", "100-400", "400+")),
                    "velocity_bin": bin_value(speed, (.5, 1.5, 3.0), ("0-.5", ".5-1.5", "1.5-3", "3+")),
                    "motion_type": "lateral" if lateral >= radial else "radial",
                    "first_visible_frame": str(first_visible_frame.get(
                        int(obj["object_id"]), "not_visible"
                    )),
                    "visibility": "visible" if visible(obj) else "occluded",
                    "camera_translation_bin": bin_value(camera_translation, (.02, .10), ("low", "medium", "high")),
                    "camera_rotation_bin": bin_value(camera_rotation, (.01, .05), ("low", "medium", "high")),
                }
                for dimension, value in values.items():
                    strata[dimension][value] += (int(object_index in matched_object_indices), 1)
            scenario_counts = per_scenario[scenario]
            scenario_counts["frames"] += 1
            scenario_counts["dynamic_track_frames"] += int(bool(tracks))
            if no_target:
                counts["no_target_frames"] += 1
                counts["no_target_false_positive_tracks"] += len(tracks)
                counts["no_target_false_positive_frames"] += int(bool(tracks))
            # A correctly associated visible/retained track cannot also be a
            # false detection of a never-observed actor.  Association is
            # mutually exclusive here just as it is in the primary Hungarian
            # assignment; the prior implementation accidentally reused every
            # track and could double-assign one track to two GT actors.
            unmatched_tracks = unmatched_by_index(tracks, matched_track_indices)
            guessed = gated_matches(
                never_observed, unmatched_tracks, config.association_distance_threshold
            )
            counts["never_observed_false_positive_matches"] += len(guessed)
            never_observed_false_match_details.extend({
                "sequence_id": sequence, "frame_index": int(frame["frame_index"]),
                "actor_id": int(never_observed[row]["object_id"]),
                "track_id": int(unmatched_tracks[column].track_id), "distance_m": distance,
            } for row, column, distance in guessed)
            counts["never_observed_actor_frames"] += len(never_observed)
            prior_assignments = dict(previous_assignment)
            current_matched = {}
            for object_index, track_index, _distance in matches:
                obj, track = eligible[object_index], tracks[track_index]
                actor_id = int(obj["object_id"])
                current_matched[actor_id] = track.track_id
                position_errors.append(float(np.linalg.norm(
                    np.asarray(obj["position_world"], dtype=float) - track.position_world
                )))
                velocity_errors.append(float(np.linalg.norm(
                    np.asarray(obj["velocity_world"], dtype=float) - track.velocity_world
                )))
                if actor_id in previous_assignment and previous_assignment[actor_id] != track.track_id:
                    counts["id_switches"] += 1
                if actor_id in ever_matched and not previous_matched.get(actor_id, False):
                    counts["track_fragmentations"] += 1
                previous_assignment[actor_id] = track.track_id
                ever_matched.add(actor_id)
                first_match_time.setdefault(actor_id, float(frame["timestamp"]))
            for obj in eligible:
                actor_id = int(obj["object_id"])
                is_occluded = not visible(obj)
                if is_occluded:
                    counts["occluded_eligible_frames"] += 1
                    retained = (actor_id in current_matched and actor_id in prior_assignments
                                and current_matched[actor_id] == prior_assignments[actor_id])
                    counts["occluded_retained_same_id_frames"] += int(retained)
                previous_matched[actor_id] = actor_id in current_matched
            truth_attention = gt_attention(eligible, frame_pose, model, config)
            iou, center_error = attention_metrics(result.attention_map[0], truth_attention)
            if eligible:
                attention_ious.append(iou)
                if center_error is not None:
                    center_errors.append(center_error)
            attention = result.attention_map.detach().cpu().numpy()
            if no_target:
                maximum = float(attention.max())
                attention_values.append(attention.reshape(-1))
                false_attention_cells.append(int((attention > 0.05).sum()))
                counts["no_target_attention_nonzero_frames"] += int(maximum > 0)
            previous_frame_pose = frame_pose
        for actor_id, timestamp in first_visible_time.items():
            if actor_id in first_seed_time:
                seed_delays.append(first_seed_time[actor_id] - timestamp)
            if actor_id in first_component_time:
                component_delays.append(first_component_time[actor_id] - timestamp)
            if actor_id in first_match_time:
                confirmation_times.append(first_match_time[actor_id] - timestamp)
    no_target_attention = (np.concatenate(attention_values) if attention_values
                           else np.asarray([0.0]))
    def pixel_metric(values):
        tp, fp, fn = (int(value) for value in values)
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": safe_fraction(tp, tp + fp),
                "recall": safe_fraction(tp, tp + fn)}

    result = {
        "sequence_count": len(sequences), "frame_count": int(counts["frames"]),
        "visible_detection_precision": safe_fraction(
            counts["visible_tp"], counts["visible_tp"] + counts["visible_fp"]
        ),
        "visible_detection_recall": safe_fraction(
            counts["visible_tp"], counts["visible_tp"] + counts["visible_fn"]
        ),
        "active_track_precision": safe_fraction(
            counts["active_tp"], counts["active_tp"] + counts["active_fp"]
        ),
        "active_track_recall": safe_fraction(
            counts["active_tp"], counts["active_tp"] + counts["active_fn"]
        ),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors))))
        if position_errors else None,
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors))))
        if velocity_errors else None,
        "id_switches": int(counts["id_switches"]),
        "track_fragmentations": int(counts["track_fragmentations"]),
        "time_to_confirm_mean_s": float(np.mean(confirmation_times))
        if confirmation_times else None,
        "first_visible_to_first_seed_mean_s": float(np.mean(seed_delays))
        if seed_delays else None,
        "first_visible_to_first_component_mean_s": float(np.mean(component_delays))
        if component_delays else None,
        "first_visible_to_dynamic_attention_mean_s": float(np.mean(confirmation_times))
        if confirmation_times else None,
        "pixel_stage_metrics": {
            name: pixel_metric(values) for name, values in sorted(pixel_counts.items())
        },
        "foreground_latency_ms": {
            "mean": float(np.mean(foreground_times)) if foreground_times else None,
            "p95": float(np.percentile(foreground_times, 95)) if foreground_times else None,
        },
        "complete_perception_latency_ms": {
            "mean": float(np.mean(perception_times)) if perception_times else None,
            "p95": float(np.percentile(perception_times, 95)) if perception_times else None,
        },
        "bounded_capacity": {
            "range_history_frames": config.range_history_frames,
            "background_max_voxels": config.background_max_voxels,
        },
        "no_target_false_positive_track_count": int(counts["no_target_false_positive_tracks"]),
        "no_target_false_positive_frame_fraction": safe_fraction(
            counts["no_target_false_positive_frames"], counts["no_target_frames"]
        ),
        "never_observed_false_positive_match_count": int(
            counts["never_observed_false_positive_matches"]
        ),
        "never_observed_false_positive_match_details": never_observed_false_match_details,
        "never_observed_actor_frame_count": int(counts["never_observed_actor_frames"]),
        "occluded_track_retention_same_id": safe_fraction(
            counts["occluded_retained_same_id_frames"], counts["occluded_eligible_frames"]
        ),
        "occluded_eligible_frame_count": int(counts["occluded_eligible_frames"]),
        "attention_iou_mean": float(np.mean(attention_ious)) if attention_ious else None,
        "attention_center_error_cells": float(np.mean(center_errors)) if center_errors else None,
        "no_target_attention_nonzero_frame_fraction": safe_fraction(
            counts["no_target_attention_nonzero_frames"], counts["no_target_frames"]
        ),
        "no_target_attention_max": float(no_target_attention.max()),
        "no_target_attention_mean": float(no_target_attention.mean()),
        "no_target_false_attention_cell_count_mean": (
            float(np.mean(false_attention_cells)) if false_attention_cells else 0.0
        ),
        "scenario_metrics": {
            scenario: {
                "frames": int(values["frames"]),
                "dynamic_track_frame_fraction": safe_fraction(
                    values["dynamic_track_frames"], values["frames"]
                ),
            } for scenario, values in sorted(per_scenario.items())
        },
        "stratified_active_recall": {
            dimension: {
                key: {"matched": int(value[0]), "eligible": int(value[1]),
                      "recall": safe_fraction(value[0], value[1])}
                for key, value in sorted(groups.items())
            } for dimension, groups in sorted(strata.items())
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", default="train,valid,test",
        help="comma-separated dataset splits; test must only be used for the final frozen evaluation",
    )
    parser.add_argument(
        "--max-sequences-per-scenario", type=int, default=0,
        help="deterministic development subset (0 means all sequences)",
    )
    parser.add_argument("--foreground-mode",
                        choices=("temporal_voxel", "range_image_hybrid"),
                        default=None)
    args = parser.parse_args()
    root = args.dataset.resolve()
    _manifest, splits = validate_dataset_splits(root)
    config = DynamicPerceptionConfig.from_global_config()
    if args.foreground_mode is not None:
        config = replace(config, foreground_mode=args.foreground_mode)
    requested_splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    unknown_splits = set(requested_splits) - set(splits)
    if unknown_splits:
        raise ValueError(f"unknown splits: {sorted(unknown_splits)}")
    selected = {}
    for split in requested_splits:
        sequences = list(splits[split])
        if args.max_sequences_per_scenario > 0:
            counts = defaultdict(int)
            limited = []
            for sequence in sequences:
                metadata = load_yaml(root / "sequences" / sequence / "metadata.yaml")
                scenario = str(metadata.get("scenario_type", "unknown"))
                if counts[scenario] >= args.max_sequences_per_scenario:
                    continue
                counts[scenario] += 1
                limited.append(sequence)
            sequences = limited
        selected[split] = sequences
    metrics = {split: evaluate_split(root, split, sequences, config)
               for split, sequences in selected.items()}
    hard_failures, warnings = [], []
    for split, values in metrics.items():
        checks = (
            ("visible_detection_precision", 0.5, "min"),
            ("active_track_precision", 0.5, "min"),
            ("visible_detection_recall", 0.2, "min"),
            ("active_track_recall", 0.2, "min"),
            ("position_rmse_m", 1.0, "max"),
            ("velocity_rmse_mps", 1.5, "max"),
            ("no_target_false_positive_frame_fraction", 0.1, "max"),
            ("no_target_attention_nonzero_frame_fraction", 0.05, "max"),
            ("attention_iou_mean", 0.05, "min"),
        )
        for name, threshold, direction in checks:
            value = values[name]
            failed = value is None or (value < threshold if direction == "min" else value > threshold)
            if failed:
                hard_failures.append(f"{split}: {name}={value} violates {direction} {threshold}")
        if values["never_observed_false_positive_match_count"]:
            hard_failures.append(
                f"{split}: never-observed false matches={values['never_observed_false_positive_match_count']}"
            )
        if values["visible_detection_precision"] is not None and values["visible_detection_precision"] < 0.7:
            warnings.append(f"{split}: visible precision below target 0.7")
        if values["velocity_rmse_mps"] is not None and values["velocity_rmse_mps"] > 1.0:
            warnings.append(f"{split}: velocity RMSE above target 1.0 m/s")
        if (values["no_target_false_positive_frame_fraction"] is not None
                and values["no_target_false_positive_frame_fraction"] > 0.05):
            warnings.append(f"{split}: no-target FP above target 0.05")
    manifest_hashes = {
        split: hashlib.sha256(("\n".join(sequences) + "\n").encode()).hexdigest()
        for split, sequences in selected.items()
    }
    implementation_files = [
        ROOT / "policy/dynamic/range_image_foreground.py",
        ROOT / "policy/dynamic/image_foreground_components.py",
        ROOT / "policy/dynamic/dynamic_perception.py",
        ROOT / "policy/dynamic/temporal_foreground.py",
        ROOT / "config/traj_opt.yaml", Path(__file__),
    ]
    implementation_hash = hashlib.sha256(b"".join(
        path.read_bytes() for path in implementation_files
    )).hexdigest()
    payload = {
        "status": "PASS" if not hard_failures else "FAIL",
        "evaluator_version": "phase8f_range_image_stages_v1",
        "dataset": str(root), "causal": True, "future_gt_used_by_tracker": False,
        "evaluated_splits": list(requested_splits),
        "max_sequences_per_scenario": args.max_sequences_per_scenario,
        "foreground_mode": config.foreground_mode,
        "sequence_manifests": selected,
        "sequence_manifest_sha256": manifest_hashes,
        "implementation_sha256": implementation_hash,
        "metric_semantics": {
            "visible": "active, inside image, non-occluded, visibility>0",
            "active_track": "active and visible, or previously observed within max_missed_frames",
            "visible_false_positive": "unmatched estimated track; correctly retained occluded matches excluded",
            "no_target": "every dynamic track is a hard false positive",
        },
        "split_metrics": metrics,
        "hard_thresholds": {
            "visible_precision_min": 0.5, "active_precision_min": 0.5,
            "recall_min": 0.2, "position_rmse_max_m": 1.0,
            "velocity_rmse_max_mps": 1.5,
            "no_target_false_positive_frame_fraction_max": 0.1,
            "no_target_attention_nonzero_frame_fraction_max": 0.05,
            "attention_iou_min": 0.05, "never_observed_false_matches": 0,
        },
        "hard_failures": hard_failures, "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
