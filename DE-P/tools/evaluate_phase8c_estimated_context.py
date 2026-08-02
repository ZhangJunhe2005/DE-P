#!/usr/bin/env python3
"""Evaluate causal depth perception against formal GT on every sequence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from ruamel.yaml import YAML
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.attention import build_dynamic_attention
from policy.dynamic.types import CameraModel, DynamicPerceptionConfig, DynamicTrack, Pose
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits
from policy.dynamic_training_config import DynamicTrainingConfig
from tools.phase8b_common import evaluate_batch, load_config, make_trainer


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def camera(metadata):
    values = metadata["camera_intrinsics"]
    return CameraModel(
        width=int(metadata["raw_image_width"]), height=int(metadata["raw_image_height"]),
        fx=float(values["fx"]), fy=float(values["fy"]), cx=float(values["cx"]),
        cy=float(values["cy"]), depth_scale=float(metadata["depth_scale"]),
        min_depth=float(values["min_depth"]), max_depth=float(values["max_depth"]),
    )


def pose(frame, metadata):
    rotation_body = Rotation.from_quat([float(frame[name]) for name in
        ("camera_qx", "camera_qy", "camera_qz", "camera_qw")]).as_matrix()
    position_body = np.asarray([frame[name] for name in
        ("camera_x", "camera_y", "camera_z")], dtype=float)
    offset = np.asarray(metadata["camera_position_body"], dtype=float)
    body_from_camera = np.asarray(metadata["camera_rotation_body_from_camera"], dtype=float)
    return Pose(position_body + rotation_body @ offset,
                rotation_body @ body_from_camera, float(frame["timestamp"]))


def gt_attention(objects, camera_pose, camera_model, perception_config):
    tracks = []
    for obj in objects:
        covariance = np.eye(6, dtype=float) * 0.01
        covariance[:3, :3] = np.asarray(obj["position_covariance"], dtype=float)
        tracks.append(DynamicTrack(
            track_id=int(obj["object_id"]), position_world=obj["position_world"],
            velocity_world=obj["velocity_world"], state_covariance=covariance,
            age=10, hit_count=10, missed_count=0, is_confirmed=True,
            is_dynamic=bool(obj["dynamic"]), timestamp=camera_pose.timestamp,
            dynamic_reason="formal_ground_truth",
        ))
    return build_dynamic_attention(
        tracks, camera_pose, camera_model,
        (cfg["vertical_num"], cfg["horizon_num"]), perception_config,
    )[0]


def attention_metrics(estimated, truth):
    est = estimated.detach().cpu().numpy().reshape(3, 5)
    gt = truth.detach().cpu().numpy().reshape(3, 5)
    union = np.logical_or(est > 0.05, gt > 0.05).sum()
    intersection = np.logical_and(est > 0.05, gt > 0.05).sum()
    iou = float(intersection / union) if union else 1.0
    yy, xx = np.mgrid[:3, :5]
    def center(value):
        total = value.sum()
        return None if total <= 1e-12 else np.asarray([(xx * value).sum(), (yy * value).sum()]) / total
    est_center, gt_center = center(est), center(gt)
    error = None if est_center is None or gt_center is None else float(np.linalg.norm(est_center - gt_center))
    return iou, error


def evaluate_tracks(root, split, sequences, config):
    counts = defaultdict(float)
    position_errors, velocity_errors, attention_ious, center_errors, confirmation_times = [], [], [], [], []
    previous_assignment, match_state, ever_matched = {}, {}, set()
    for sequence in sequences:
        directory = root / "sequences" / sequence
        metadata = load_yaml(directory / "metadata.yaml")
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        model = camera(metadata)
        perception = DynamicPerception(config, (cfg["vertical_num"], cfg["horizon_num"]))
        first_visible, first_matched = {}, {}
        no_target = metadata["scenario_type"] == "no_target"
        for frame in frames:
            depth = np.load(directory / frame["depth_path"], allow_pickle=False)
            frame_pose = pose(frame, metadata)
            result = perception.update_depth(
                depth, frame_pose, float(frame["timestamp"]), model,
            )
            objects = json.loads((directory / frame["dynamic_objects_path"]).read_text())
            visible = [obj for obj in objects if bool(obj["active"]) and
                       bool(obj["inside_image"]) and float(obj["visibility"]) > 0]
            for obj in visible:
                first_visible.setdefault(int(obj["object_id"]), float(frame["timestamp"]))
            estimated = list(result.dynamic_tracks)
            matches = []
            if visible and estimated:
                distance = np.asarray([[np.linalg.norm(
                    np.asarray(obj["position_world"]) - track.position_world
                ) for track in estimated] for obj in visible])
                rows, columns = linear_sum_assignment(distance)
                matches = [(int(row), int(column)) for row, column in zip(rows, columns)
                           if distance[row, column] <= config.association_distance_threshold]
            counts["true_positive"] += len(matches)
            counts["false_positive"] += len(estimated) - len(matches)
            counts["false_negative"] += len(visible) - len(matches)
            counts["visible_gt"] += len(visible)
            counts["estimated_dynamic"] += len(estimated)
            counts["frames"] += 1
            counts["no_target_frames"] += int(no_target)
            if no_target and estimated:
                counts["no_target_false_positive_frames"] += 1
            matched_gt = set()
            for gt_index, estimate_index in matches:
                obj, track = visible[gt_index], estimated[estimate_index]
                actor_id = int(obj["object_id"])
                matched_gt.add(actor_id)
                position_errors.append(float(np.linalg.norm(
                    np.asarray(obj["position_world"]) - track.position_world
                )))
                velocity_errors.append(float(np.linalg.norm(
                    np.asarray(obj["velocity_world"]) - track.velocity_world
                )))
                if actor_id in previous_assignment and previous_assignment[actor_id] != track.track_id:
                    counts["id_switches"] += 1
                previous_assignment[actor_id] = track.track_id
                if actor_id in ever_matched and match_state.get(actor_id) is False:
                    counts["track_fragmentations"] += 1
                ever_matched.add(actor_id); match_state[actor_id] = True
                first_matched.setdefault(actor_id, float(frame["timestamp"]))
            for obj in visible:
                actor_id = int(obj["object_id"])
                if actor_id not in matched_gt:
                    match_state[actor_id] = False
            truth_attention = gt_attention(visible, frame_pose, model, config)
            iou, center_error = attention_metrics(result.attention_map[0], truth_attention)
            if visible:
                attention_ious.append(iou)
                if center_error is not None:
                    center_errors.append(center_error)
        for actor_id, timestamp in first_visible.items():
            if actor_id in first_matched:
                confirmation_times.append(first_matched[actor_id] - timestamp)
    precision = counts["true_positive"] / max(counts["true_positive"] + counts["false_positive"], 1)
    recall = counts["true_positive"] / max(counts["true_positive"] + counts["false_negative"], 1)
    return {
        "sequence_count": len(sequences), "frame_count": int(counts["frames"]),
        "detection_precision": precision, "detection_recall": recall,
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None,
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else None,
        "id_switches": int(counts["id_switches"]),
        "track_fragmentations": int(counts["track_fragmentations"]),
        "time_to_confirm_mean_s": float(np.mean(confirmation_times)) if confirmation_times else None,
        "no_target_false_positive_frames": int(counts["no_target_false_positive_frames"]),
        "no_target_false_positive_frame_fraction": (
            counts["no_target_false_positive_frames"] / max(counts["no_target_frames"], 1)
        ),
        "attention_iou_mean": float(np.mean(attention_ious)) if attention_ious else None,
        "attention_center_error_cells": float(np.mean(center_errors)) if center_errors else None,
        "matched_detection_count": int(counts["true_positive"]),
    }


@torch.inference_mode()
def compare_network(root, splits, checkpoint, cache_dir, config, training_config_path):
    training = DynamicTrainingConfig.from_global_config()
    ground_truth_training = replace(training, context_source="ground_truth")
    estimated_training = replace(training, context_source="estimated", estimated_cache_dir=str(cache_dir))
    dynamic_config = replace(config, enabled=True, use_attention=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = DepNetwork(backbone_variant="corrected", dynamic_config=dynamic_config).to(device).eval()
    load_dep_checkpoint(network, checkpoint, "corrected")
    experiment_config = load_config(training_config_path)
    trainer = make_trainer(
        experiment_config, data_root=str(root),
        map_catalog=experiment_config["dynamic_map_catalog"],
    )
    trainer.policy.load_state_dict(network.state_dict(), strict=True)
    result = {}
    for split, sequences in splits.items():
        gt_dataset = DynamicSequenceDataset(root, split, ground_truth_training, config)
        estimated_dataset = DynamicSequenceDataset(root, split, estimated_training, config)
        by_sequence = defaultdict(list)
        for index, (sequence, _current, _category) in enumerate(gt_dataset._windows):
            by_sequence[sequence].append(index)
        differences, score_differences, changed, ious, centers = [], [], 0, [], []
        gt_selected_risks, estimated_selected_risks = [], []
        gt_collisions, estimated_collisions = [], []
        for sequence in sequences:
            indices = by_sequence[sequence]
            index = indices[len(indices) // 2]
            gt_sample, estimated_sample = gt_dataset[index], estimated_dataset[index]
            iou, center = attention_metrics(estimated_sample["attention"], gt_sample["attention"])
            ious.append(iou)
            if center is not None:
                centers.append(center)
            gt_batch = dynamic_sequence_collate([gt_sample])
            est_batch = dynamic_sequence_collate([estimated_sample])
            depth = gt_batch["current_depth"].to(device)
            observation = gt_batch["observation_9d"].to(device)
            gt_end, gt_score = network.inference(depth, observation, gt_batch["dynamic_context"].to(device))
            est_end, est_score = network.inference(depth, observation, est_batch["dynamic_context"].to(device))
            differences.append(float((gt_end - est_end).abs().mean()))
            score_differences.append(float((gt_score - est_score).abs().mean()))
            changed += int(int(gt_score.argmin()) != int(est_score.argmin()))
            # Both passes are evaluated against the same recorded future GT.
            # Only the causal context supplied to the policy differs.
            gt_risk = evaluate_batch(trainer, gt_batch)
            estimated_risk = evaluate_batch(trainer, est_batch)
            gt_selected_risks.append(gt_risk["top1_raw_risk"])
            estimated_selected_risks.append(estimated_risk["top1_raw_risk"])
            gt_collisions.append(gt_risk["top1_collision_fraction"])
            estimated_collisions.append(estimated_risk["top1_collision_fraction"])
        gt_selected_mean = float(np.mean(gt_selected_risks))
        estimated_selected_mean = float(np.mean(estimated_selected_risks))
        result[split] = {
            "compared_windows": len(sequences),
            "endstate_mae": float(np.mean(differences)), "score_mae": float(np.mean(score_differences)),
            "best_primitive_changed_fraction": changed / len(sequences),
            "window_attention_iou_mean": float(np.mean(ious)),
            "window_attention_center_error_cells": float(np.mean(centers)) if centers else None,
            "gt_selected_raw_risk_mean": gt_selected_mean,
            "estimated_selected_raw_risk_mean": estimated_selected_mean,
            "estimated_minus_gt_selected_raw_risk": estimated_selected_mean - gt_selected_mean,
            "gt_selected_collision_fraction": float(np.mean(gt_collisions)),
            "estimated_selected_collision_fraction": float(np.mean(estimated_collisions)),
            "estimated_minus_gt_selected_collision_fraction": (
                float(np.mean(estimated_collisions)) - float(np.mean(gt_collisions))
            ),
        }
    trainer.tensorboard_log.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    _manifest, splits = validate_dataset_splits(root)
    config = DynamicPerceptionConfig.from_global_config()
    split_metrics = {split: evaluate_tracks(root, split, sequences, config)
                     for split, sequences in splits.items()}
    network = compare_network(
        root, splits, args.checkpoint, args.cache_dir, config, args.config.resolve()
    )
    hard_failures, warnings = [], []
    for split, values in split_metrics.items():
        if values["detection_precision"] < 0.5:
            hard_failures.append(f"{split}: detection precision below 0.5")
        elif values["detection_precision"] < 0.7:
            warnings.append(f"{split}: detection precision below warning level 0.7")
        if values["detection_recall"] < 0.2:
            hard_failures.append(f"{split}: detection recall below 0.2")
        elif values["detection_recall"] < 0.4:
            warnings.append(f"{split}: detection recall below warning level 0.4")
        if values["position_rmse_m"] is None or values["position_rmse_m"] > 1.0:
            hard_failures.append(f"{split}: position RMSE exceeds 1.0 m")
        elif values["position_rmse_m"] > 0.5:
            warnings.append(f"{split}: position RMSE exceeds warning level 0.5 m")
        if values["velocity_rmse_mps"] is None or values["velocity_rmse_mps"] > 1.5:
            hard_failures.append(f"{split}: velocity RMSE exceeds 1.5 m/s")
        elif values["velocity_rmse_mps"] > 1.0:
            warnings.append(f"{split}: velocity RMSE exceeds warning level 1.0 m/s")
        if values["attention_iou_mean"] is None or values["attention_iou_mean"] < 0.05:
            hard_failures.append(f"{split}: attention IoU below 0.05")
        elif values["attention_iou_mean"] < 0.15:
            warnings.append(f"{split}: attention IoU below warning level 0.15")
        if values["no_target_false_positive_frame_fraction"] > 0.1:
            hard_failures.append(f"{split}: no-target false-positive frames exceed 0.1")
        elif values["no_target_false_positive_frame_fraction"] > 0.05:
            warnings.append(f"{split}: no-target false-positive frames exceed warning level 0.05")
    for split, values in network.items():
        if values["estimated_minus_gt_selected_raw_risk"] > 0.25:
            hard_failures.append(f"{split}: estimated-context selected-risk delta exceeds 0.25")
        elif values["estimated_minus_gt_selected_raw_risk"] > 0.10:
            warnings.append(f"{split}: estimated-context selected-risk delta exceeds warning level 0.10")
        if values["estimated_minus_gt_selected_collision_fraction"] > 0.10:
            hard_failures.append(f"{split}: estimated-context selected-collision delta exceeds 0.10")
        elif values["estimated_minus_gt_selected_collision_fraction"] > 0.05:
            warnings.append(f"{split}: estimated-context selected-collision delta exceeds warning level 0.05")
    if hard_failures:
        recommendation = {
            "estimated_context_training_sufficient": False,
            "recommended_start_epoch": None,
            "recommended_ratio": 0.0,
            "action": "improve perception/cache quality before estimated-context training",
        }
    elif warnings:
        recommendation = {
            "estimated_context_training_sufficient": True,
            "recommended_start_epoch": 12,
            "recommended_ratio": 0.25,
            "action": "use delayed low-ratio estimated context and keep fixed estimated validation",
        }
    else:
        recommendation = {
            "estimated_context_training_sufficient": True,
            "recommended_start_epoch": 8,
            "recommended_ratio": 0.5,
            "action": "estimated context may be introduced gradually, capped at 0.5",
        }
    payload = {
        "status": "PASS" if not hard_failures else "FAIL", "dataset": str(root),
        "causal_history_and_current_only": True, "future_gt_used_by_tracker": False,
        "split_metrics": split_metrics, "network_gt_vs_estimated": network,
        "estimated_cache_dir": str(args.cache_dir.resolve()),
        "estimated_cache_files": len(tuple(args.cache_dir.glob("*.pt"))),
        "cache_key_includes_config_version_sequence_hash_and_timestamps": True,
        "hard_thresholds": {"precision_min": 0.5, "recall_min": 0.2,
                       "position_rmse_max_m": 1.0, "velocity_rmse_max_mps": 1.5,
                       "attention_iou_min": 0.05,
                       "no_target_false_positive_frame_fraction_max": 0.1,
                       "selected_raw_risk_delta_max": 0.25,
                       "selected_collision_fraction_delta_max": 0.10},
        "warning_thresholds": {"precision_min": 0.7, "recall_min": 0.4,
                       "position_rmse_max_m": 0.5, "velocity_rmse_max_mps": 1.0,
                       "attention_iou_min": 0.15,
                       "no_target_false_positive_frame_fraction_max": 0.05,
                       "selected_raw_risk_delta_max": 0.10,
                       "selected_collision_fraction_delta_max": 0.05},
        "recommendation": recommendation,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "failures": hard_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
