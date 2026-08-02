#!/usr/bin/env python3
"""Conformance audit for the legacy V2 static continuous lower bound."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from policy.safety_evaluator_v2 import (
    SafetyEvaluatorV2Config, evaluate_quintic, quintic_coefficients,
)


SOURCE = ROOT / "artifacts/phase8jv2s0/fixed_050_seed8403-paired-static.npz"
OUTPUT = ROOT / "artifacts/phase8jv2s0/static_continuous_conformance.npz"
REPORT = ROOT / "reports/phase8jv2s0_static_continuous_conformance.json"
LEVELS = (16, 32, 64, 128)
MAX_LEVEL = 128
RADIUS = 0.3
TOLERANCE = 1e-6
# Predeclared before the full audit.  Either threshold is sufficient to
# require a versioned numerical rebaseline.
FALSE_UNSAFE_WINDOW_TOLERANCE = 0.005
FALSE_UNSAFE_TRAJECTORY_TOLERANCE = 0.001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dense_positions(start, end, config):
    batch = len(start)
    latency = config.latency_s
    prefix_times = np.linspace(0.0, latency, 2 * MAX_LEVEL + 1)
    prefix = (
        start[:, None, :, 0]
        + prefix_times[None, :, None] * start[:, None, :, 1]
        + .5 * prefix_times[None, :, None] ** 2 * start[:, None, :, 2]
    )
    first = start.copy()
    first[:, :, 0] = prefix[:, -1]
    first[:, :, 1] = start[:, :, 1] + latency * start[:, :, 2]
    duration = config.wall_clock_horizon_s - latency
    controlled_times = np.linspace(
        0.0, duration,
        config.controlled_samples * config.subdivisions * MAX_LEVEL + 1,
    )
    count = (
        2 * MAX_LEVEL
        + config.controlled_samples * config.subdivisions * MAX_LEVEL + 1
    )
    output = np.empty((batch, 15, count, 3), np.float32)
    output[:, :, :2*MAX_LEVEL] = prefix[:, None, :-1]
    for row in range(batch):
        for candidate in range(15):
            coefficients = quintic_coefficients(
                first[row], end[row, candidate], duration
            )
            positions, _, _ = evaluate_quintic(
                coefficients, controlled_times
            )
            output[row, candidate, 2*MAX_LEVEL:] = positions
    return output


def certificate(raw, positions):
    trajectories = raw.shape[0]
    base_segments = (raw.shape[1] - 1) // MAX_LEVEL
    raw_segments = np.stack([
        raw[:, segment*MAX_LEVEL:segment*MAX_LEVEL+MAX_LEVEL+1]
        for segment in range(base_segments)
    ], axis=1)
    position_segments = np.stack([
        positions[:, segment*MAX_LEVEL:segment*MAX_LEVEL+MAX_LEVEL+1]
        for segment in range(base_segments)
    ], axis=1)
    collision = raw_segments.min(axis=(1, 2)) < RADIUS - TOLERANCE
    # Bottom-up dyadic certificate. A parent is certified if its endpoint
    # Lipschitz bound is safe, or both children are certified.
    safe_nodes = None
    for depth in range(7, -1, -1):
        step = 2 ** (7 - depth)
        left = np.arange(0, MAX_LEVEL, step)
        right = left + step
        bound = np.minimum(
            raw_segments[:, :, left], raw_segments[:, :, right]
        ) - np.linalg.norm(
            position_segments[:, :, right] - position_segments[:, :, left],
            axis=-1,
        )
        direct = bound >= RADIUS - TOLERANCE
        if safe_nodes is None:
            safe_nodes = direct
        else:
            children = safe_nodes.reshape(
                trajectories, base_segments, -1, 2
            ).all(axis=-1)
            safe_nodes = direct | children
    certified_safe = safe_nodes[:, :, 0].all(axis=1)
    unknown = ~collision & ~certified_safe
    return collision, certified_safe, unknown


def bin_name(value, edges, labels):
    return labels[int(np.searchsorted(edges, value, side="right"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()
    paired = np.load(SOURCE, allow_pickle=False)
    failure = ~(
        np.asarray(paired["legacy_clearance"]) >= -TOLERANCE
    ).any(axis=1)
    failure_indices = np.flatnonzero(failure)
    full_failure_count = len(failure_indices)
    if args.max_failures:
        failure_indices = failure_indices[:args.max_failures]
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        ROOT / "reports/phase8jq_controller_authoritative_envelope.json",
    )
    loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=ROOT / "configs/static_map_catalog.yaml",
    )
    device = loss.device
    all_current, all_dense, all_dense_clearance, all_collision = [], [], [], []
    all_certified, all_unknown, all_map, all_window = [], [], [], []
    all_segment_length, all_speed = [], []
    representatives = defaultdict(list)
    started = time.perf_counter()
    for begin in range(0, len(failure_indices), args.batch_size):
        indices = failure_indices[begin:begin+args.batch_size]
        start = np.asarray(paired["start"][indices], float)
        end = np.asarray(paired["end"][indices], float)
        maps = np.asarray(paired["map_id"][indices], int)
        positions = dense_positions(start, end, config)
        size = len(indices)
        query = torch.from_numpy(
            positions.reshape(size, -1, 3)
        ).to(device)
        map_tensor = torch.as_tensor(maps, device=device)
        with torch.inference_mode():
            _, raw_tensor = loss.safety_loss.get_distance_cost(
                query, map_tensor
            )
        raw = raw_tensor.reshape(
            size, 15, positions.shape[2]
        ).detach().cpu().numpy()
        trajectory_positions = positions.reshape(-1, positions.shape[2], 3)
        trajectory_raw = raw.reshape(-1, raw.shape[2])
        current_positions = trajectory_positions[:, ::MAX_LEVEL]
        current_raw = trajectory_raw[:, ::MAX_LEVEL]
        current_bound = np.minimum(
            current_raw[:, :-1], current_raw[:, 1:]
        ) - np.linalg.norm(
            np.diff(current_positions, axis=1), axis=2
        )
        current_clearance = np.minimum(
            current_raw.min(axis=1), current_bound.min(axis=1)
        ) - RADIUS
        dense_safe = []
        dense_clearance = []
        for level in LEVELS:
            stride = MAX_LEVEL // level
            level_clearance = trajectory_raw[:, ::stride].min(axis=1) - RADIUS
            dense_clearance.append(level_clearance)
            dense_safe.append(level_clearance >= -TOLERANCE)
        dense_safe = np.stack(dense_safe, axis=1)
        dense_clearance = np.stack(dense_clearance, axis=1)
        collision, certified_safe, unknown = certificate(
            trajectory_raw, trajectory_positions
        )
        base_length = np.linalg.norm(
            np.diff(current_positions, axis=1), axis=2
        ).max(axis=1)
        speeds = np.repeat(
            np.linalg.norm(start[:, :, 1], axis=1), 15
        )
        expanded_maps = np.repeat(maps, 15)
        expanded_windows = np.repeat(indices, 15)
        false_unsafe = (
            (current_clearance < -TOLERANCE) & dense_safe[:, -1]
        )
        for local in np.flatnonzero(false_unsafe):
            map_id = int(expanded_maps[local])
            if len(representatives[map_id]) < 5:
                representatives[map_id].append({
                    "window_index": int(expanded_windows[local]),
                    "candidate_index": int(local % 15),
                    "positions": trajectory_positions[local].copy(),
                    "dense_esdf_clearance_m": float(
                        trajectory_raw[local].min() - RADIUS
                    ),
                })
        all_current.append(current_clearance)
        all_dense.append(dense_safe)
        all_dense_clearance.append(dense_clearance)
        all_collision.append(collision)
        all_certified.append(certified_safe)
        all_unknown.append(unknown)
        all_map.append(expanded_maps)
        all_window.append(expanded_windows)
        all_segment_length.append(base_length)
        all_speed.append(speeds)
        done = begin + size
        if done % 100 == 0 or done == len(failure_indices):
            print(
                f"continuous conformance {done}/{len(failure_indices)}",
                flush=True,
            )
    current = np.concatenate(all_current)
    dense = np.concatenate(all_dense)
    dense_clearance = np.concatenate(all_dense_clearance)
    collision = np.concatenate(all_collision)
    certified = np.concatenate(all_certified)
    unknown = np.concatenate(all_unknown)
    maps = np.concatenate(all_map)
    windows = np.concatenate(all_window)
    segment_length = np.concatenate(all_segment_length)
    speeds = np.concatenate(all_speed)
    current_safe = current >= -TOLERANCE
    false_unsafe = ~current_safe & dense[:, -1]
    false_safe = current_safe & ~dense[:, -1]
    false_unsafe_windows = np.unique(windows[false_unsafe])

    independent = []
    catalog = {
        int(row["map_id"]): Path(row["static_ply"])
        for row in __import__("ruamel.yaml", fromlist=["YAML"]).YAML(
            typ="safe"
        ).load(ROOT / "configs/static_map_catalog.yaml")["maps"]
    }
    for map_id, rows in sorted(representatives.items()):
        points = np.asarray(
            o3d.io.read_point_cloud(str(catalog[map_id])).points
        )
        tree = cKDTree(points)
        for row in rows:
            nearest, _ = tree.query(row.pop("positions"), k=1, workers=-1)
            independent.append({
                **row,
                "map_id": map_id,
                "pointcloud_nearest_clearance_m":
                    float(nearest.min() - RADIUS),
                "pointcloud_collision": bool(nearest.min() < RADIUS),
            })

    map_groups = {}
    for map_id in sorted(set(maps)):
        selected = maps == map_id
        map_groups[str(map_id)] = {
            "trajectory_count": int(selected.sum()),
            "current_unsafe_dense_128_safe": int(
                false_unsafe[selected].sum()
            ),
            "certificate_unknown": int(unknown[selected].sum()),
            "dense_collision": int(collision[selected].sum()),
        }
    speed_groups = Counter()
    length_groups = Counter()
    for flag, speed, length in zip(false_unsafe, speeds, segment_length):
        if not flag:
            continue
        speed_groups[bin_name(
            speed, [3.0, 6.0, 7.2], ["<=3", "3-6", "6-7.2", ">7.2"]
        )] += 1
        length_groups[bin_name(
            length, [.05, .10, .20, .50],
            ["<=.05", ".05-.10", ".10-.20", ".20-.50", ">.50"],
        )] += 1
    trajectory_fraction = float(false_unsafe.mean())
    window_fraction = float(
        len(false_unsafe_windows) / max(len(failure_indices), 1)
    )
    numerical_rebaseline = (
        trajectory_fraction > FALSE_UNSAFE_TRAJECTORY_TOLERANCE
        or window_fraction > FALSE_UNSAFE_WINDOW_TOLERANCE
    )
    arrays = {
        "failure_window_index": failure_indices,
        "trajectory_window_index": windows,
        "map_id": maps,
        "current_clearance": current,
        "dense_safe_by_level": dense,
        "dense_clearance_by_level": dense_clearance,
        "dense_collision": collision,
        "recursive_certified_safe": certified,
        "recursive_unknown": unknown,
        "max_base_segment_length": segment_length,
        "initial_speed": speeds,
        "levels": np.asarray(LEVELS),
        "source_artifact_sha256": np.asarray(sha256(SOURCE)),
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, OUTPUT)
    report = {
        "status": "FAIL_NUMERICAL_REBASELINE_REQUIRED"
        if numerical_rebaseline else "PASS",
        "scope": "FULL_C0_FAILURE_SET" if not args.max_failures else "SMOKE",
        "paired_c0_failure_window_count": full_failure_count,
        "audited_failure_window_count": len(failure_indices),
        "audited_trajectory_count": len(current),
        "predeclared_tolerance": {
            "false_unsafe_window_fraction_max":
                FALSE_UNSAFE_WINDOW_TOLERANCE,
            "false_unsafe_trajectory_fraction_max":
                FALSE_UNSAFE_TRAJECTORY_TOLERANCE,
        },
        "comparison": {
            "current_unsafe_dense_128_safe_trajectories":
                int(false_unsafe.sum()),
            "current_unsafe_dense_128_safe_trajectory_fraction":
                trajectory_fraction,
            "current_unsafe_dense_128_safe_windows":
                int(len(false_unsafe_windows)),
            "current_unsafe_dense_128_safe_window_fraction":
                window_fraction,
            "current_safe_dense_128_unsafe_trajectories":
                int(false_safe.sum()),
            "recursive_certified_safe_trajectories": int(certified.sum()),
            "recursive_confirmed_collision_trajectories":
                int(collision.sum()),
            "recursive_unknown_trajectories": int(unknown.sum()),
            "maximum_current_vs_dense_clearance_difference_m": float(
                np.max(np.abs(current - dense_clearance[:, -1]))
            ),
        },
        "dense_levels": {
            str(level): {
                "sample_safe_trajectories": int(dense[:, index].sum()),
                "current_unsafe_sample_safe_trajectories": int(
                    ((~current_safe) & dense[:, index]).sum()
                ),
            }
            for index, level in enumerate(LEVELS)
        },
        "interpretation": {
            "current_lower_bound_negative": (
                "unable to certify safety unless an evaluated endpoint is "
                "inside the 0.3 m physical radius"
            ),
            "confirmed_collision_rule": (
                "queried ESDF distance - 0.3 m < -1e-6"
            ),
            "certificate_unknown_is_not_collision": True,
        },
        "map_level": map_groups,
        "false_unsafe_by_initial_speed_mps": dict(speed_groups),
        "false_unsafe_by_max_base_segment_length_m": dict(length_groups),
        "independent_pointcloud_fixtures": independent,
        "artifact": str(OUTPUT.resolve()),
        "artifact_sha256": sha256(OUTPUT),
        "performance": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "geometry_semantics_changed": False,
        "uav_radius_m": RADIUS,
        "next_allowed_phase": (
            "phase8jqv2_1_static_numerical_rebaseline"
            if numerical_rebaseline else None
        ),
    }
    atomic_json(
        REPORT if not args.max_failures
        else ROOT / "reports/phase8jv2s0_static_continuous_conformance_smoke.json",
        report,
    )
    print(json.dumps({
        "status": report["status"],
        "scope": report["scope"],
        "false_unsafe_trajectories": int(false_unsafe.sum()),
        "false_unsafe_windows": int(len(false_unsafe_windows)),
        "trajectory_fraction": trajectory_fraction,
        "window_fraction": window_fraction,
        "next_allowed_phase": report["next_allowed_phase"],
    }, indent=2))


if __name__ == "__main__":
    main()
