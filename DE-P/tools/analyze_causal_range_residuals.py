#!/usr/bin/env python3
"""Causal ego-motion compensated range-residual separability analysis.

Ground truth is used only after a mask has been produced, for offline metrics.
The detector itself consumes current/history raw depth, poses and intrinsics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.dynamic.camera_model import make_depth_frame
from policy.dynamic_sequence_dataset import validate_dataset_splits
from tools.evaluate_phase8c_estimated_context import camera, load_yaml, pose


def reproject_depth(history, current_pose, model):
    """Reproject one historical DepthFrame with a nearest-pixel z-buffer."""
    points_world = (history.camera_pose_world.rotation_world_from_camera
                    @ history.points_camera.T).T + history.camera_pose_world.position_world
    rotation_camera_from_world = current_pose.rotation_world_from_camera.T
    points = (rotation_camera_from_world
              @ (points_world - current_pose.position_world).T).T
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z >= model.min_depth) & (z < model.max_depth)
    u = np.rint(model.fx * points[:, 0] / np.maximum(z, 1e-9) + model.cx).astype(np.int64)
    v = np.rint(model.fy * points[:, 1] / np.maximum(z, 1e-9) + model.cy).astype(np.int64)
    valid &= (u >= 0) & (u < model.width) & (v >= 0) & (v < model.height)
    flat = np.full(model.width * model.height, np.inf, dtype=np.float32)
    index = v[valid] * model.width + u[valid]
    np.minimum.at(flat, index, z[valid].astype(np.float32))
    return flat.reshape(model.height, model.width)


def depth_edge(depth, valid):
    edge = np.zeros_like(depth, dtype=np.float32)
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    delta = np.abs(depth[:, 1:] - depth[:, :-1])
    edge[:, 1:] = np.maximum(edge[:, 1:], np.where(horizontal, delta, 0))
    edge[:, :-1] = np.maximum(edge[:, :-1], np.where(horizontal, delta, 0))
    delta = np.abs(depth[1:, :] - depth[:-1, :])
    edge[1:, :] = np.maximum(edge[1:, :], np.where(vertical, delta, 0))
    edge[:-1, :] = np.maximum(edge[:-1, :], np.where(vertical, delta, 0))
    return edge


def actor_proxy_mask(objects, depth, model):
    """Conservative rendered-pixel proxy; never used by residual generation."""
    yy, xx = np.indices(depth.shape)
    mask = np.zeros(depth.shape, dtype=bool)
    for obj in objects:
        if (not obj.get("active", True) or obj.get("occluded", False)
                or not obj.get("inside_image", False) or obj.get("visibility", 0) <= 0):
            continue
        z_surface = float(obj.get("expected_surface_depth", np.nan))
        z_center = z_surface + float(obj.get("radius", 0.4))
        if not np.isfinite(z_center) or z_center <= 0:
            continue
        radius_px = max(1.0, model.fx * float(obj.get("radius", 0.4)) / z_center)
        if obj.get("type") in {"cylinder", "capsule"}:
            half_h = model.fy * float(obj.get("height", 0.8)) / (2.0 * z_center)
        else:
            half_h = radius_px
        spatial = (((xx - float(obj["projected_u"])) / radius_px) ** 2
                   + ((yy - float(obj["projected_v"])) / max(half_h, 1.0)) ** 2 <= 1.0)
        # Reject foreground occluders in the projected box.  The generous far
        # band accommodates non-spherical shapes while remaining GT-only.
        depth_match = (depth >= z_surface - 0.20) & (
            depth <= z_surface + 2.25 * float(obj.get("radius", 0.4))
        )
        mask |= spatial & depth_match
    return mask


def ratios(mask, truth, valid):
    selected = mask & valid
    tp = int(np.sum(selected & truth)); fp = int(np.sum(selected & ~truth))
    fn = int(np.sum(~selected & truth)); static = int(np.sum(valid & ~truth))
    return tp, fp, fn, static


def select_sequences(root, splits, split, per_scenario):
    selected, counts = [], defaultdict(int)
    for sequence in splits[split]:
        metadata = load_yaml(root / "sequences" / sequence / "metadata.yaml")
        scenario = str(metadata.get("scenario_type", "unknown"))
        if counts[scenario] < per_scenario:
            selected.append(sequence); counts[scenario] += 1
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--sequences-per-scenario", type=int, default=2)
    parser.add_argument("--history-frames", type=int, default=4)
    args = parser.parse_args()
    if args.sequences_per_scenario <= 0 or args.history_frames <= 0:
        raise ValueError("subset and history bounds must be positive")
    root = args.dataset.resolve()
    _manifest, splits = validate_dataset_splits(root)
    sequences = select_sequences(root, splits, args.split, args.sequences_per_scenario)
    sequence_hash = hashlib.sha256(("\n".join(sequences) + "\n").encode()).hexdigest()
    candidates = [
        (absolute, relative, support, edge)
        for absolute in (0.12, 0.20, 0.30)
        for relative in (0.01, 0.02, 0.04)
        for support in (1, 2, 3)
        for edge in (0.30, 0.60, 1.00)
    ]
    totals = {candidate: np.zeros(4, dtype=np.int64) for candidate in candidates}
    strata = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.int64)))
    evidence_delays = defaultdict(list)
    support_histogram = np.zeros(args.history_frames + 1, dtype=np.int64)
    residual_dynamic, residual_static = [], []
    for sequence in sequences:
        directory = root / "sequences" / sequence
        metadata = load_yaml(directory / "metadata.yaml")
        model = camera(metadata)
        rows = list(csv.DictReader((directory / "frames.csv").open()))
        history = []
        first_visible, first_evidence = {}, {}
        scenario = str(metadata.get("scenario_type", "unknown"))
        previous_pose = None
        for frame_index, row in enumerate(rows):
            timestamp = float(row["timestamp"]); current_pose = pose(row, metadata)
            raw = np.load(directory / row["depth_path"], allow_pickle=False)
            frame = make_depth_frame(raw, model, current_pose, timestamp, stride=1)
            objects = json.loads((directory / row["dynamic_objects_path"]).read_text())
            truth = actor_proxy_mask(objects, frame.depth_m, model)
            edge = depth_edge(frame.depth_m, frame.valid_mask)
            predictions = [reproject_depth(old, current_pose, model) for old in history]
            if predictions:
                predicted = np.stack(predictions)
                valid_history = np.isfinite(predicted) & frame.valid_mask[None]
                residual = predicted - frame.depth_m[None]
                support_histogram += np.bincount(
                    np.minimum(valid_history.sum(axis=0), args.history_frames).ravel(),
                    minlength=args.history_frames + 1,
                )[:args.history_frames + 1]
                best = np.max(np.where(valid_history, residual, -np.inf), axis=0)
                residual_dynamic.extend(best[truth & np.isfinite(best)].tolist())
                static_sample = best[~truth & np.isfinite(best)][::20]
                residual_static.extend(static_sample.tolist())
                for candidate in candidates:
                    absolute, relative, minimum_support, edge_threshold = candidate
                    threshold = absolute + relative * frame.depth_m
                    closer = valid_history & (residual > threshold[None])
                    static_consistent = np.sum(
                        valid_history & (np.abs(residual) <= 0.12), axis=0
                    ) >= minimum_support
                    seed = ((closer.sum(axis=0) >= minimum_support)
                            & ~static_consistent & (edge <= edge_threshold)
                            & frame.valid_mask)
                    values = np.asarray(ratios(seed, truth, frame.valid_mask))
                    totals[candidate] += values
                    strata[scenario][candidate] += values
                    if candidate == (0.20, 0.02, 2, 0.60):
                        for obj in objects:
                            actor_id = int(obj["object_id"])
                            if (obj.get("active", True) and not obj.get("occluded", False)
                                    and obj.get("visibility", 0) > 0):
                                first_visible.setdefault(actor_id, frame_index)
                                if np.any(seed & actor_proxy_mask([obj], frame.depth_m, model)):
                                    first_evidence.setdefault(actor_id, frame_index)
            history.append(frame)
            history = history[-args.history_frames:]
            previous_pose = current_pose
        for actor_id, start in first_visible.items():
            if actor_id in first_evidence:
                evidence_delays["all"].append(first_evidence[actor_id] - start)
                evidence_delays["visible_at_frame0" if start == 0 else "enters_later"].append(
                    first_evidence[actor_id] - start
                )

    def metric(values):
        tp, fp, fn, static = (int(x) for x in values)
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "static_trigger_fraction": fp / static if static else None}
    ranked = []
    for candidate, values in totals.items():
        result = metric(values)
        # Train-only deterministic selection: maximize recall among precision
        # >=0.5, otherwise maximize F0.5.  Validation never changes thresholds.
        p, r = result["precision"] or 0.0, result["recall"] or 0.0
        score = (2.0 if p >= 0.5 else 0.0) + (1.25 * p * r / (0.25 * p + r)
                                                    if p + r else 0.0)
        ranked.append((score, candidate, result))
    ranked.sort(reverse=True)
    chosen = ranked[0]
    payload = {
        "status": "PASS", "analysis_version": "phase8f_causal_range_signal_v1",
        "dataset": str(root), "split": args.split,
        "test_split_read": False, "future_frames_used": 0,
        "static_map_oracle_used": False, "gt_used_by_mask_generator": False,
        "gt_metric_semantics": "conservative actor projection/depth proxy; no recorded instance mask exists",
        "sequence_manifest": sequences, "sequence_manifest_sha256": sequence_hash,
        "history_frames": args.history_frames,
        "selected_train_thresholds": {
            "absolute_m": chosen[1][0], "relative": chosen[1][1],
            "minimum_history_support": chosen[1][2], "edge_guard_m": chosen[1][3],
        },
        "selected_metrics": chosen[2],
        "top_candidates": [dict(metric=item[2], absolute_m=item[1][0],
                                relative=item[1][1], minimum_history_support=item[1][2],
                                edge_guard_m=item[1][3]) for item in ranked[:12]],
        "scenario_metrics_at_selected": {
            name: metric(values[chosen[1]]) for name, values in sorted(strata.items())
        },
        "history_support_histogram": support_histogram.tolist(),
        "first_visible_to_first_seed_frames": {
            name: {"count": len(values), "mean": float(np.mean(values)) if values else None,
                   "p95": float(np.percentile(values, 95)) if values else None}
            for name, values in evidence_delays.items()
        },
        "residual_quantiles_m": {
            "actor_proxy": np.quantile(residual_dynamic, [0.1, .5, .9, .99]).tolist()
            if residual_dynamic else None,
            "static_sampled": np.quantile(residual_static, [0.1, .5, .9, .99]).tolist()
            if residual_static else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
