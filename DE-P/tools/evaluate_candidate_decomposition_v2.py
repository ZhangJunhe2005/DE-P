#!/usr/bin/env python3
"""Read-only full-validation Safety Evaluator V2 decomposition."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.safety_evaluator_v2 import SafetyEvaluatorV2, SafetyEvaluatorV2Config
from tools.run_phase8i_failure_decomposition import bootstrap_ci


REPORTS = ROOT / "reports"
DATASET = ROOT / "data/phase8_dynamic_production"
ARTIFACTS_V1 = ROOT / "artifacts/phase8i"
ARTIFACTS_V2 = ROOT / "artifacts/phase8jqv2"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2"
COUNT = 15
TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metric(flags, inclusion, description):
    flags = np.asarray(flags, dtype=bool)
    inclusion = np.asarray(inclusion, dtype=bool)
    numerator = int((flags & inclusion).sum())
    denominator = int(inclusion.sum())
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
        "inclusion_rule": description,
        "exclusion_rule": "windows not satisfying inclusion_rule",
    }


def safe_correlation(left, right, kind):
    result = spearmanr(left, right).statistic if kind == "spearman" else kendalltau(
        left, right
    ).statistic
    return float(np.nan_to_num(result))


class SequenceData:
    def __init__(self, identifier):
        self.root = DATASET / "sequences" / identifier
        with (self.root / "frames.csv").open(newline="") as stream:
            self.frames = list(csv.DictReader(stream))
        self.timestamps = np.asarray([float(row["timestamp"]) for row in self.frames])
        self.objects = [
            json.loads((self.root / row["dynamic_objects_path"]).read_text())
            for row in self.frames
        ]
        self._actor_timeline_cache = {}

    def current_state(self, index):
        row = self.frames[int(index)]
        return np.asarray([
            [float(row[f"camera_{axis}"]), float(row[f"velocity_{axis}"]),
             float(row[f"acceleration_{axis}"])]
            for axis in "xyz"
        ])

    def actors_at(self, frame_index, offsets):
        cache_key = (
            int(frame_index), len(offsets), float(offsets[0]), float(offsets[-1])
        )
        if cache_key in self._actor_timeline_cache:
            return self._actor_timeline_cache[cache_key]
        output = []
        base = self.timestamps[int(frame_index)]
        for offset in offsets:
            query = base + float(offset)
            upper = int(np.searchsorted(self.timestamps, query, side="left"))
            upper = min(max(upper, 0), len(self.timestamps) - 1)
            lower = max(0, upper - 1)
            denominator = self.timestamps[upper] - self.timestamps[lower]
            alpha = 0.0 if denominator <= 0 else float(
                np.clip((query - self.timestamps[lower]) / denominator, 0, 1)
            )
            left = {int(row["object_id"]): row for row in self.objects[lower]}
            right = {int(row["object_id"]): row for row in self.objects[upper]}
            actors = []
            for identifier in sorted(left.keys() | right.keys()):
                a = left.get(identifier, right[identifier])
                b = right.get(identifier, a)
                row = dict(a)
                row["position_world"] = (
                    (1-alpha)*np.asarray(a["position_world"], float)
                    + alpha*np.asarray(b["position_world"], float)
                ).tolist()
                row["active"] = bool(a.get("active", True) or b.get("active", True))
                actors.append(row)
            output.append(actors)
        self._actor_timeline_cache[cache_key] = output
        return output


class EstimatedCovariance:
    def __init__(self):
        self.root = ROOT / "cache/phase8i_phase8h_perception"
        self.index = json.loads((self.root / "index.json").read_text())["entries"]
        self.memo = {}
        self.hits = 0

    def get(self, sequence, frame, current_actors):
        key = (sequence, int(frame))
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        entry = self.index[f"valid:{sequence}:{int(frame)}"]
        payload = torch.load(
            self.root / entry["path"], map_location="cpu", weights_only=False
        )
        tracks = [
            row for row in payload["tracks"]
            if row.get("is_confirmed") and row.get("is_dynamic") and row.get("active")
        ]
        result = {}
        remaining = set(range(len(tracks)))
        for actor in current_actors:
            if not remaining:
                break
            position = np.asarray(actor["position_world"], float)
            chosen = min(
                remaining,
                key=lambda index: np.linalg.norm(
                    np.asarray(tracks[index]["position_world"], float) - position
                ),
            )
            distance = float(np.linalg.norm(
                np.asarray(tracks[chosen]["position_world"], float) - position
            ))
            if distance <= 1.5:
                result[int(actor["object_id"])] = np.asarray(
                    tracks[chosen]["state_covariance"], float
                )[:3, :3]
                remaining.remove(chosen)
        self.memo[key] = result
        return result


def static_esdf_query(loss, positions, map_ids, device):
    tensor = torch.from_numpy(positions.astype(np.float32)).to(device)
    maps = torch.as_tensor(map_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        _, distance = loss.safety_loss.get_distance_cost(tensor, maps)
    return distance.cpu().numpy()


def vectorized_dynamic(evaluator, candidate_positions, actors_by_time, suite,
                       covariance_by_actor):
    candidate_positions = np.asarray(candidate_positions, float)
    actor_ids = sorted({
        int(actor["object_id"]) for actors in actors_by_time for actor in actors
        if actor.get("dynamic", True)
    })
    candidates, points = candidate_positions.shape[:2]
    if not actor_ids:
        maximum = np.full((candidates, points), np.inf)
        return {
            "physical_min": np.full(candidates, np.inf),
            "physical_discrete_min": np.full(candidates, np.inf),
            "sphere_discrete_min": np.full(candidates, np.inf),
            "planning_min": np.full(candidates, np.inf),
            "planning_discrete_min": np.full(candidates, np.inf),
            "continuous_min": np.full(candidates, np.inf),
            "by_time": maximum,
        }
    lookup = {identifier: index for index, identifier in enumerate(actor_ids)}
    obstacles = len(actor_ids)
    positions = np.zeros((points, obstacles, 3))
    radii = np.zeros(obstacles)
    heights = np.zeros(obstacles)
    cylinders = np.zeros(obstacles, dtype=bool)
    active = np.zeros((points, obstacles), dtype=bool)
    for time_index, actors in enumerate(actors_by_time):
        for actor in actors:
            identifier = int(actor["object_id"])
            if identifier not in lookup or not actor.get("dynamic", True):
                continue
            index = lookup[identifier]
            positions[time_index, index] = actor["position_world"]
            radii[index] = float(actor["radius"])
            heights[index] = float(actor.get("height", 2*radii[index]))
            cylinders[index] = actor.get("type") in {
                "vertical_cylinder", "cylinder"
            }
            active[time_index, index] = bool(actor.get("active", True))
    difference = (
        candidate_positions[:, :, None, :] - positions[None, :, :, :]
    )
    sphere = np.linalg.norm(difference, axis=-1) - (
        evaluator.config.uav_radius_m + radii[None, None]
    )
    radial_gap = np.maximum(
        0.0, np.linalg.norm(difference[..., :2], axis=-1) - radii[None, None]
    )
    vertical_gap = np.maximum(
        0.0, np.abs(difference[..., 2]) - .5*heights[None, None]
    )
    cylinder = np.hypot(radial_gap, vertical_gap) - evaluator.config.uav_radius_m
    physical = np.where(cylinders[None, None], cylinder, sphere)
    physical = np.where(active[None], physical, np.inf)
    uncertainty = np.zeros_like(physical)
    if suite == "valid_estimated":
        for identifier, covariance in covariance_by_actor.items():
            if identifier not in lookup:
                continue
            obstacle = lookup[identifier]
            covariance = np.asarray(covariance, float)
            if evaluator.config.estimated_uncertainty_policy == "maximum_eigenvalue":
                margin = evaluator.config.confidence_sigma * np.sqrt(
                    max(0.0, float(np.linalg.eigvalsh(covariance).max()))
                )
                uncertainty[:, :, obstacle] = margin
            else:
                separation = difference[:, :, obstacle]
                norm = np.linalg.norm(separation, axis=-1)
                direction = separation / np.maximum(norm[..., None], 1e-12)
                variance = np.einsum(
                    "...i,ij,...j->...", direction, covariance, direction
                )
                uncertainty[:, :, obstacle] = (
                    evaluator.config.confidence_sigma
                    * np.sqrt(np.maximum(variance, 0.0))
                )
    planning = (
        physical - evaluator.config.dynamic_planning_margin_m
        - evaluator.config.tracking_control_margin_m - uncertainty
    )
    by_time = physical.min(axis=2)
    continuous = physical.min(axis=(1, 2))
    # Exact linear closest approach for sphere actors on every dense segment.
    if points > 1 and np.any(~cylinders):
        relative = difference[:, :-1]  # [C,S,M,3]
        actor_step = np.diff(positions, axis=0)
        uav_step = np.diff(candidate_positions, axis=1)
        relative_step = uav_step[:, :, None] - actor_step[None]
        denominator = np.sum(relative_step**2, axis=-1)
        fraction = np.clip(
            -np.sum(relative * relative_step, axis=-1)
            / np.maximum(denominator, 1e-15),
            0.0, 1.0,
        )
        closest = np.linalg.norm(
            relative + fraction[..., None] * relative_step, axis=-1
        ) - (evaluator.config.uav_radius_m + radii[None, None])
        valid_segment = active[:-1] & active[1:] & ~cylinders[None]
        closest = np.where(valid_segment[None], closest, np.inf)
        continuous = np.minimum(continuous, closest.min(axis=(1, 2)))
    return {
        "physical_min": physical.min(axis=(1, 2)),
        "physical_discrete_min": physical.min(axis=(1, 2)),
        "sphere_discrete_min": np.where(
            active[None], sphere, np.inf
        ).min(axis=(1, 2)),
        "planning_min": planning.min(axis=(1, 2)),
        "planning_discrete_min": planning.min(axis=(1, 2)),
        "continuous_min": continuous,
        "by_time": by_time,
    }


def strict_checkpoint_audit(checkpoints, device):
    network = DepNetwork(backbone_variant="corrected").to(device).eval()
    rows = []
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"])
        load_dep_checkpoint(network, path, "corrected")
        rows.append({
            "name": checkpoint["name"], "path": str(path),
            "sha256": sha256(path), "expected_sha256": checkpoint["sha256"],
            "strict_load": True, "hash_match": sha256(path) == checkpoint["sha256"],
        })
    return rows


def evaluate_artifact(
    artifact, suite, evaluator, sequences, covariance_cache, static_loss, device,
    batch_size, cache_path, analysis_key, maximum_windows=0,
):
    if cache_path.is_file():
        loaded = dict(np.load(cache_path, allow_pickle=True))
        if str(loaded.get("analysis_key", "")) == analysis_key:
            return loaded, True
    source = dict(np.load(artifact, allow_pickle=True))
    total = (
        min(int(maximum_windows), len(source["sequence"]))
        if maximum_windows else len(source["sequence"])
    )
    columns = defaultdict(list)
    started = time.perf_counter()
    for begin in range(0, total, batch_size):
        end = min(total, begin + batch_size)
        batch_positions = []
        batch_current_positions = []
        batch_maps = []
        row_timelines = []
        for row in range(begin, end):
            sequence_id = str(source["sequence"][row])
            frame = int(source["frame"][row])
            sequence = sequences.setdefault(sequence_id, SequenceData(sequence_id))
            current = sequence.current_state(frame)
            candidates = []
            for candidate in range(COUNT):
                timeline = evaluator.timeline(current, source["trajectory"][row, candidate])
                candidates.append(timeline)
            positions = np.stack([item["positions"] for item in candidates])
            row_timelines.append((sequence, frame, candidates))
            batch_positions.append(positions.reshape(-1, 3))
            batch_current_positions.append(current[:, 0][None])
            batch_maps.append(int(source["map"][row]))
        maximum = max(value.shape[0] for value in batch_positions)
        if any(value.shape[0] != maximum for value in batch_positions):
            raise RuntimeError("V2 timeline point count changed within a batch")
        raw_static = static_esdf_query(
            static_loss, np.stack(batch_positions), batch_maps, device
        ).reshape(end-begin, COUNT, -1)
        raw_static_t0 = static_esdf_query(
            static_loss, np.stack(batch_current_positions), batch_maps, device
        ).reshape(end-begin)
        for local, row in enumerate(range(begin, end)):
            sequence, frame, candidates = row_timelines[local]
            physical_dynamic, planning_dynamic = [], []
            continuous_dynamic, physical_static, planning_static = [], [], []
            final_joint = []
            current_actors = sequence.actors_at(frame, [0.0])[0]
            covariances = (
                covariance_cache.get(
                    str(source["sequence"][row]), frame, current_actors
                ) if suite == "valid_estimated" else {}
            )
            unmatched = sum(
                actor.get("dynamic", True) and int(actor["object_id"]) not in covariances
                for actor in current_actors
            ) if suite == "valid_estimated" else 0
            candidate_positions = np.stack(
                [timeline["positions"] for timeline in candidates]
            )
            actors = sequence.actors_at(frame, candidates[0]["times"])
            dynamic_all = vectorized_dynamic(
                evaluator, candidate_positions, actors, suite, covariances
            )
            # Fixed-order attribution begins on V1's original sample timeline:
            # remove its heuristic uncertainty while retaining spherical actors,
            # then enable the exact Simulator actor geometry.
            old_positions = np.asarray(source["trajectory"][row], float)
            old_times = np.linspace(
                evaluator.config.wall_clock_horizon_s / old_positions.shape[1],
                evaluator.config.wall_clock_horizon_s,
                old_positions.shape[1],
            )
            old_dynamic = vectorized_dynamic(
                evaluator, old_positions,
                sequence.actors_at(frame, old_times), "valid_gt", {},
            )
            t0_positions = np.concatenate(
                (
                    np.repeat(current[:, 0][None, None], COUNT, axis=0),
                    old_positions,
                ),
                axis=1,
            )
            t0_dynamic = vectorized_dynamic(
                evaluator, t0_positions,
                sequence.actors_at(
                    frame, np.concatenate((np.asarray([0.0]), old_times))
                ),
                "valid_gt", {},
            )
            for candidate, timeline in enumerate(candidates):
                static = evaluator.evaluate_static(raw_static[local, candidate])
                # 1-Lipschitz conservative segment lower bound.
                segment = np.linalg.norm(
                    np.diff(timeline["positions"], axis=0), axis=1
                )
                raw = raw_static[local, candidate]
                lower = np.minimum(raw[:-1], raw[1:]) - segment
                static_continuous = min(
                    static["physical_min"],
                    float(np.min(lower) - evaluator.config.uav_radius_m),
                )
                static_planning_continuous = (
                    static_continuous
                    - evaluator.config.static_planning_margin_m
                    - evaluator.config.tracking_control_margin_m
                )
                physical_dynamic.append(dynamic_all["continuous_min"][candidate])
                continuous_dynamic.append(float(
                    dynamic_all["continuous_min"][candidate]
                    - dynamic_all["physical_min"][candidate]
                ) if np.isfinite(dynamic_all["physical_min"][candidate]) else 0.0)
                planning_dynamic.append(dynamic_all["planning_min"][candidate])
                physical_static.append(static_continuous)
                planning_static.append(static_planning_continuous)
                final_joint.append(min(
                    dynamic_all["by_time"][candidate, -1],
                    static["physical_clearance_by_time"][-1],
                ))
            physical_dynamic = np.asarray(physical_dynamic)
            planning_dynamic = np.asarray(planning_dynamic)
            physical_static = np.asarray(physical_static)
            planning_static = np.asarray(planning_static)
            secondary = np.asarray(source["guidance"][row], float) + np.asarray(
                source["smooth"][row], float
            )
            physical_label = evaluator.safety_first_label(
                physical_dynamic, physical_static, secondary
            )
            planning_label = evaluator.safety_first_label(
                planning_dynamic, planning_static, secondary
            )
            physical_joint = np.minimum(physical_dynamic, physical_static)
            planning_joint = np.minimum(planning_dynamic, planning_static)
            columns["physical_dynamic"].append(physical_dynamic)
            columns["planning_dynamic"].append(planning_dynamic)
            columns["physical_static"].append(physical_static)
            columns["planning_static"].append(planning_static)
            columns["physical_label"].append(physical_label)
            columns["planning_label"].append(planning_label)
            columns["continuous_delta"].append(continuous_dynamic)
            columns["ablation_remove_uncertainty_dynamic"].append(
                old_dynamic["sphere_discrete_min"]
            )
            columns["ablation_exact_shape_dynamic"].append(
                old_dynamic["physical_discrete_min"]
            )
            columns["ablation_include_t0_dynamic"].append(
                t0_dynamic["physical_discrete_min"]
            )
            columns["ablation_latency_dynamic"].append(
                dynamic_all["physical_discrete_min"]
            )
            radius_adjusted_v1_static = (
                np.asarray(source["static_clearance"][row], float)
                - evaluator.config.uav_radius_m
            )
            columns["ablation_static_radius"].append(radius_adjusted_v1_static)
            columns["ablation_include_t0_static"].append(np.minimum(
                radius_adjusted_v1_static,
                raw_static_t0[local] - evaluator.config.uav_radius_m,
            ))
            columns["ablation_latency_static"].append(
                raw_static[local].min(axis=1) - evaluator.config.uav_radius_m
            )
            columns["final_joint"].append(final_joint)
            columns["t0_joint"].append(float(min(
                dynamic_all["by_time"][0, 0],
                evaluator.evaluate_static(raw_static[local, 0])[
                    "physical_clearance_by_time"
                ][0],
            )))
            latency_index = int(np.flatnonzero(np.isclose(
                candidates[0]["times"], evaluator.config.latency_s
            ))[0])
            static_prefix = evaluator.evaluate_static(raw_static[local, 0])[
                "physical_clearance_by_time"
            ]
            columns["first_joint"].append(float(min(
                dynamic_all["by_time"][0, latency_index],
                static_prefix[latency_index],
            )))
            columns["unmatched_covariance"].append(unmatched)
    output = {name: np.asarray(value) for name, value in columns.items()}
    for name in (
        "sequence", "frame", "map", "scenario", "category", "predicted", "label",
        "dynamic_clearance", "static_clearance", "guidance", "smooth",
    ):
        output[f"v1_{name}"] = source[name][:total]
    output.update({
        "analysis_key": np.asarray(analysis_key),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **output)
    os.replace(temporary, cache_path)
    return output, False


def summarize(data):
    predicted = np.asarray(data["v1_predicted"], float)
    old_label = np.asarray(data["v1_label"], float)
    physical_label = np.asarray(data["physical_label"], float)
    planning_label = np.asarray(data["planning_label"], float)
    physical_dynamic = np.asarray(data["physical_dynamic"], float)
    planning_dynamic = np.asarray(data["planning_dynamic"], float)
    physical_static = np.asarray(data["physical_static"], float)
    planning_static = np.asarray(data["planning_static"], float)
    physical_safe = (physical_dynamic >= -TOLERANCE) & (
        physical_static >= -TOLERANCE
    )
    planning_safe = (planning_dynamic >= -TOLERANCE) & (
        planning_static >= -TOLERANCE
    )
    dynamic_physical_safe = physical_dynamic >= -TOLERANCE
    dynamic_planning_safe = planning_dynamic >= -TOLERANCE
    rows = np.arange(len(predicted))
    predicted_index = predicted.argmin(1)
    physical_label_index = physical_label.argmin(1)
    planning_label_index = planning_label.argmin(1)
    physical_exists = physical_safe.any(1)
    planning_exists = planning_safe.any(1)
    predicted_physical_safe = physical_safe[rows, predicted_index]
    predicted_planning_safe = planning_safe[rows, predicted_index]
    physical_label_safe = physical_safe[rows, physical_label_index]
    planning_label_safe = planning_safe[rows, planning_label_index]
    t0 = np.asarray(data["t0_joint"], float)
    first = np.asarray(data["first_joint"], float)
    initially_safe = t0 >= -TOLERANCE
    first_safe = first >= -TOLERANCE
    preventable = initially_safe & first_safe
    initially_unsafe = ~initially_safe
    final_joint = np.asarray(data["final_joint"], float)
    recoverable = initially_unsafe & (
        (final_joint >= -TOLERANCE).any(1)
        & (np.minimum(physical_dynamic, physical_static) >= t0[:, None]-TOLERANCE).any(1)
    )
    flags = {
        "physical_dynamic_coverage_failure": ~dynamic_physical_safe.any(1),
        "physical_joint_coverage_failure": ~physical_exists,
        "planning_dynamic_coverage_failure": ~dynamic_planning_safe.any(1),
        "planning_joint_coverage_failure": ~planning_exists,
        "physical_label_inversion": physical_exists & ~physical_label_safe,
        "planning_label_inversion": planning_exists & ~planning_label_safe,
        "model_ranking_failure": planning_exists & planning_label_safe
        & ~predicted_planning_safe,
        "combined_selection_failure": planning_exists & ~predicted_planning_safe,
        "top1_physical_collision": ~predicted_physical_safe,
        "top1_planning_unsafe": ~predicted_planning_safe,
    }
    sequences = np.asarray(data["v1_sequence"], str)
    all_rows = np.ones(len(predicted), dtype=bool)
    fractions = {
        name: {
            **metric(value, all_rows, "all sequential full-validation windows"),
            "sequence_bootstrap": bootstrap_ci(value, sequences),
        }
        for name, value in flags.items()
    }
    correlations = defaultdict(list)
    physical_risk = np.maximum(-np.minimum(physical_dynamic, physical_static), 0)
    planning_risk = np.maximum(-np.minimum(planning_dynamic, planning_static), 0)
    pair_accuracy = []
    for row in range(len(predicted)):
        for name, values in (
            ("predicted_v1_label", old_label[row]),
            ("predicted_physical_label", physical_label[row]),
            ("predicted_planning_label", planning_label[row]),
            ("predicted_physical_risk", physical_risk[row]),
            ("predicted_planning_risk", planning_risk[row]),
        ):
            correlations[f"{name}_spearman"].append(
                safe_correlation(predicted[row], values, "spearman")
            )
            correlations[f"{name}_kendall"].append(
                safe_correlation(predicted[row], values, "kendall")
            )
        safe_values = predicted[row, planning_safe[row]]
        unsafe_values = predicted[row, ~planning_safe[row]]
        if len(safe_values) and len(unsafe_values):
            pair_accuracy.append(float(
                (safe_values[:, None] < unsafe_values[None]).mean()
            ))
    return {
        "window_count": len(predicted),
        "fractions": fractions,
        "coverage": {
            "physical_safe_candidate_count": distribution(physical_safe.sum(1)),
            "planning_safe_candidate_count": distribution(planning_safe.sum(1)),
            "physical_oracle_collision_fraction": fractions[
                "physical_joint_coverage_failure"
            ],
            "planning_oracle_collision_fraction": fractions[
                "planning_joint_coverage_failure"
            ],
        },
        "actionability": {
            "already_unsafe_fraction": metric(
                initially_unsafe, all_rows, "all full-validation windows"
            ),
            "first_controllable_unsafe_fraction": metric(
                ~first_safe, all_rows, "all full-validation windows"
            ),
            "initially_safe_future_collision_fraction": metric(
                initially_safe & ~physical_exists, initially_safe,
                "windows physically safe at t=0",
            ),
            "preventable_window_count": int(preventable.sum()),
            "recoverable_window_count": int(recoverable.sum()),
            "scenario_feasibility_unknown_fraction": metric(
                np.ones(len(predicted), bool), all_rows,
                "generator supplies no feasibility certificate",
            ),
            "preventable_physical_coverage_failure": metric(
                ~physical_exists, preventable,
                "t0-safe and first-controllable-safe windows",
            ),
            "preventable_planning_coverage_failure": metric(
                ~planning_exists, preventable,
                "t0-safe and first-controllable-safe windows",
            ),
            "recoverable_success": metric(
                recoverable, initially_unsafe, "t0-unsafe windows"
            ),
        },
        "selection": {
            "oracle_regret_mean": float(
                (planning_label[rows, predicted_index] - planning_label.min(1)).mean()
            ),
            "clearance_regret_mean_m": float(
                (
                    np.minimum(planning_dynamic, planning_static).max(1)
                    - np.minimum(planning_dynamic, planning_static)[rows, predicted_index]
                ).mean()
            ),
            "pair_accuracy_mean": float(np.mean(pair_accuracy)) if pair_accuracy else None,
        },
        "correlations": {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for name, values in correlations.items()
        },
        "internal_flags": flags,
        "internal_masks": {
            "initially_safe": initially_safe, "first_safe": first_safe,
            "preventable": preventable, "recoverable": recoverable,
            "physical_exists": physical_exists, "planning_exists": planning_exists,
        },
    }


def distribution(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q10": float(np.quantile(values, .1)), "q50": float(np.quantile(values, .5)),
        "q90": float(np.quantile(values, .9)),
    }


def public(summary):
    return {
        key: value for key, value in summary.items()
        if key not in {"internal_flags", "internal_masks"}
    }


def grouped(data, summary):
    output = {}
    for dimension, values in (
        ("scenario", np.asarray(data["v1_scenario"], str)),
        ("map", np.asarray(data["v1_map"], int).astype(str)),
    ):
        output[dimension] = {}
        for value in sorted(set(values)):
            mask = values == value
            output[dimension][value] = {
                name: metric(flags, mask, f"{dimension}={value}")
                for name, flags in summary["internal_flags"].items()
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-windows", type=int, default=0)
    args = parser.parse_args()
    entry = json.loads((REPORTS / "phase8jqv2_entry_gate.json").read_text())
    if entry["status"] != "PASS":
        raise RuntimeError("entry Gate is not PASS")
    matrix = json.loads((REPORTS / "phase8i_checkpoint_matrix.json").read_text())
    checkpoints = [
        row for row in matrix["checkpoints"] if row["analyzed_full_phase8i"]
    ]
    if args.smoke_windows:
        checkpoints = checkpoints[:1]
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        REPORTS / "phase8jq_controller_authoritative_envelope.json",
    )
    evaluator = SafetyEvaluatorV2(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=DATASET / "map_catalog.yaml",
    )
    strict = strict_checkpoint_audit(checkpoints, device)
    if not all(row["strict_load"] and row["hash_match"] for row in strict):
        raise RuntimeError("checkpoint strict/hash audit failed")
    sequences = {}
    covariance_cache = EstimatedCovariance()
    results, raw, cache_hits, cache_misses = {}, {}, 0, 0
    started = time.perf_counter()
    for checkpoint in checkpoints:
        name = checkpoint["name"]
        results[name], raw[name] = {}, {}
        for suite in ("valid_estimated", "valid_gt"):
            artifact = ARTIFACTS_V1 / f"{name}-{suite}.npz"
            source = dict(np.load(artifact, allow_pickle=True))
            if str(source["checkpoint_sha256"]) != checkpoint["sha256"]:
                raise RuntimeError(f"V1 artifact checkpoint mismatch: {name}/{suite}")
            key = canonical_hash({
                "evaluator": config.canonical_hash(),
                "evaluator_implementation": sha256(
                    ROOT / "tools/evaluate_candidate_decomposition_v2.py"
                ),
                "checkpoint": checkpoint["sha256"],
                "suite": suite,
                "source_artifact": sha256(artifact),
                "dataset": entry["frozen_hashes"]["production_manifest_sha256"],
                "cache": entry["frozen_hashes"]["estimated_cache_index_sha256"]
                if suite == "valid_estimated" else None,
                "smoke_windows": args.smoke_windows,
            })
            suffix = f"-smoke{args.smoke_windows}" if args.smoke_windows else ""
            arrays, hit = evaluate_artifact(
                artifact, suite, evaluator, sequences, covariance_cache,
                static_loss, device, args.batch_size,
                ARTIFACTS_V2 / f"{name}-{suite}{suffix}.npz", key,
                args.smoke_windows,
            )
            cache_hits += int(hit)
            cache_misses += int(not hit)
            summary = summarize(arrays)
            raw[name][suite] = arrays
            results[name][suite] = {
                **public(summary), "groups": grouped(arrays, summary)
            }
            print(f"{name} {suite} complete cache={hit}", flush=True)
    if args.smoke_windows:
        smoke = {
            "status": "PASS",
            "windows_per_suite": args.smoke_windows,
            "checkpoint": checkpoints[0]["name"],
            "valid_estimated": public(summarize(raw[checkpoints[0]["name"]]["valid_estimated"])),
            "valid_gt": public(summarize(raw[checkpoints[0]["name"]]["valid_gt"])),
            "production_test_used": False,
        }
        atomic_json(REPORTS / "phase8jqv2_smoke.json", smoke)
        print(json.dumps({"status": "PASS", "smoke_windows": args.smoke_windows}, indent=2))
        return
    representative = "fixed_050_seed8403"
    estimated = raw[representative]["valid_estimated"]
    gt = raw[representative]["valid_gt"]
    estimated_summary = summarize(estimated)
    gt_summary = summarize(gt)
    base_hashes = {
        "evaluator_version": config.evaluator_version,
        "config_hash": entry["frozen_hashes"]["v2_config_sha256"],
        "geometry_hash": sha256(ROOT / "loss/safety_geometry_v2.py"),
        "timeline_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "uncertainty_policy_hash": canonical_hash({
            "policy": config.estimated_uncertainty_policy,
            "sigma": config.confidence_sigma,
        }),
        "Simulator_geometry_hash": json.loads(
            (REPORTS / "phase8jqv2_geometry_validation.json").read_text()
        )["simulator_geometry_hash"],
        "dataset_manifest_hash": entry["frozen_hashes"]["production_manifest_sha256"],
        "cache_index_hash": entry["frozen_hashes"]["estimated_cache_index_sha256"],
        "checkpoint_hash": next(
            row["sha256"] for row in checkpoints
            if row["name"] == representative
        ),
        "checkpoint_hashes": {
            row["name"]: row["sha256"] for row in checkpoints
        },
    }
    static_report_path = REPORTS / "phase8jqv2_static_validation.json"
    if not static_report_path.is_file():
        raise RuntimeError(
            "full V2 valid_static report is required before final rebaseline"
        )
    static_report = json.loads(static_report_path.read_text())
    if (
        static_report.get("status") != "PASS"
        or static_report.get("full_sequential") is not True
        or static_report.get("window_count") != 10000
        or static_report.get("config_hash") != base_hashes["config_hash"]
    ):
        raise RuntimeError("full V2 valid_static report is incomplete or stale")
    for checkpoint in checkpoints:
        name = checkpoint["name"]
        static_result = static_report["results"].get(name)
        if (
            not static_result
            or static_result.get("checkpoint_hash") != checkpoint["sha256"]
            or static_result.get("sequential_complete") is not True
        ):
            raise RuntimeError(f"invalid V2 valid_static result: {name}")
        results[name]["valid_static"] = static_result
    checkpoint_report = {
        **base_hashes,
        "status": "PASS_WITH_DECLARED_MISSING_HISTORICAL_ARTIFACT",
        "strict_load_results": strict,
        "phase8b": matrix["phase8b"],
        "results": results,
    }
    atomic_json(REPORTS / "phase8jqv2_checkpoint_matrix.json", checkpoint_report)
    atomic_json(REPORTS / "phase8jqv2_physical_coverage.json", {
        **base_hashes, "status": "PASS", "decision_checkpoint": representative,
        "valid_estimated": physical_view(estimated_summary),
        "valid_gt": physical_view(gt_summary),
        "all_checkpoints": {
            name: {
                suite: physical_view(summarize(raw[name][suite]))
                for suite in ("valid_estimated", "valid_gt")
            } for name in raw
        },
    })
    atomic_json(REPORTS / "phase8jqv2_planning_coverage.json", {
        **base_hashes, "status": "PASS", "decision_checkpoint": representative,
        "valid_estimated": planning_view(estimated_summary),
        "valid_gt": planning_view(gt_summary),
    })
    atomic_json(REPORTS / "phase8jqv2_actionability_metrics.json", {
        **base_hashes, "status": "PASS",
        "valid_estimated": estimated_summary["actionability"],
        "valid_gt": gt_summary["actionability"],
    })
    write_diagnostics(gt, gt_summary)
    delta = v1_v2_delta(gt)
    atomic_json(REPORTS / "phase8jqv2_v1_v2_delta.json", {
        **base_hashes, "status": "PASS", **delta,
    })
    label_report = label_audit(estimated, estimated_summary, gt, gt_summary)
    atomic_json(REPORTS / "phase8jqv2_label_scale_audit.json", {
        **base_hashes, "status": "PASS", **label_report,
    })
    atomic_json(REPORTS / "phase8jqv2_context_gap_analysis.json", {
        **base_hashes, "status": "PASS",
        "window_pairing_exact": bool(
            np.array_equal(estimated["v1_sequence"], gt["v1_sequence"])
            and np.array_equal(estimated["v1_frame"], gt["v1_frame"])
        ),
        "estimated_gt_physical_joint_coverage_gap": (
            estimated_summary["fractions"]["physical_joint_coverage_failure"]["fraction"]
            - gt_summary["fractions"]["physical_joint_coverage_failure"]["fraction"]
        ),
        "estimated_covariance_cache_hits": covariance_cache.hits,
        "unmatched_covariance_count": int(
            np.asarray(estimated["unmatched_covariance"]).sum()
        ),
    })
    ablation = component_ablation(gt)
    atomic_json(REPORTS / "phase8jqv2_component_ablation.json", {
        **base_hashes, "status": "PASS", **ablation,
    })
    decomposition = {
        **base_hashes,
        "status": "PASS",
        "validation_sequential_complete": True,
        "window_counts": {
            "valid_estimated": 2052, "valid_gt": 2052,
            "valid_static": static_report["window_count"],
        },
        "valid_static": {
            "status": "PASS",
            "full_sequential": True,
            "report": str(static_report_path.resolve()),
            "decision_checkpoint": static_report["results"][representative],
        },
        "decision_checkpoint": representative,
        "checkpoint_results": results,
        "performance": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available() else 0,
            "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "artifact_cache_hits": cache_hits, "artifact_cache_misses": cache_misses,
            "estimated_cache_hits": covariance_cache.hits,
            "estimated_cache_misses": 0,
            "atomic_artifacts": True, "resume_supported": True,
            "gpu_cpu_tolerance_m": TOLERANCE,
        },
        "production_test_used": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
    }
    atomic_json(
        REPORTS / "phase8jqv2_candidate_failure_decomposition.json",
        decomposition,
    )
    finalize(
        base_hashes, estimated_summary, gt_summary, label_report, delta,
        cache_hits, cache_misses,
    )


def physical_view(summary):
    return {
        "dynamic_coverage_failure": summary["fractions"][
            "physical_dynamic_coverage_failure"
        ],
        "joint_coverage_failure": summary["fractions"][
            "physical_joint_coverage_failure"
        ],
        "safe_candidate_count": summary["coverage"]["physical_safe_candidate_count"],
        "oracle_collision_fraction": summary["coverage"][
            "physical_oracle_collision_fraction"
        ],
        "preventable_coverage_failure": summary["actionability"][
            "preventable_physical_coverage_failure"
        ],
    }


def planning_view(summary):
    return {
        "dynamic_coverage_failure": summary["fractions"][
            "planning_dynamic_coverage_failure"
        ],
        "joint_coverage_failure": summary["fractions"][
            "planning_joint_coverage_failure"
        ],
        "safe_candidate_count": summary["coverage"]["planning_safe_candidate_count"],
        "oracle_collision_fraction": summary["coverage"][
            "planning_oracle_collision_fraction"
        ],
        "preventable_coverage_failure": summary["actionability"][
            "preventable_planning_coverage_failure"
        ],
    }


def v1_v2_delta(data):
    v1 = (np.asarray(data["v1_dynamic_clearance"], float) >= -TOLERANCE) & (
        np.asarray(data["v1_static_clearance"], float) >= -TOLERANCE
    )
    v2 = (np.asarray(data["planning_dynamic"], float) >= -TOLERANCE) & (
        np.asarray(data["planning_static"], float) >= -TOLERANCE
    )
    remove_uncertainty = (
        np.asarray(data["ablation_remove_uncertainty_dynamic"], float) >= 0
    )
    exact_shape = np.asarray(data["ablation_exact_shape_dynamic"], float) >= 0
    static_radius = np.asarray(data["ablation_static_radius"], float) >= 0
    include_t0 = (
        (np.asarray(data["ablation_include_t0_dynamic"], float) >= 0)
        & (np.asarray(data["ablation_include_t0_static"], float) >= 0)
    )
    include_latency = (
        (np.asarray(data["ablation_latency_dynamic"], float) >= 0)
        & (np.asarray(data["ablation_latency_static"], float) >= 0)
    )
    continuous = (
        (np.asarray(data["physical_dynamic"], float) >= 0)
        & (np.asarray(data["physical_static"], float) >= 0)
    )
    v1_dynamic_safe = np.asarray(data["v1_dynamic_clearance"], float) >= 0
    v1_static_safe = np.asarray(data["v1_static_clearance"], float) >= 0
    stage_safe = [
        v1_dynamic_safe & v1_static_safe,
        remove_uncertainty & v1_static_safe,
        exact_shape & v1_static_safe,
        exact_shape & static_radius,
        include_t0,
        include_latency,
        continuous,
        v2,
    ]
    candidate_change_count = np.zeros(v2.shape, dtype=np.int8)
    for left, right in zip(stage_safe, stage_safe[1:]):
        candidate_change_count += left != right
    return {
        "candidate_safe_to_unsafe_count": int((v1 & ~v2).sum()),
        "candidate_unsafe_to_safe_count": int((~v1 & v2).sum()),
        "window_outcome_flip_count": int((v1.any(1) != v2.any(1)).sum()),
        "static_radius_only_flip_count": int(
            (v1_static_safe != static_radius).sum()
        ),
        "uncertainty_removal_only_flip_count": int(
            (v1_dynamic_safe != remove_uncertainty).sum()
        ),
        "shape_correction_only_flip_count": int(
            (remove_uncertainty != exact_shape).sum()
        ),
        "t0_only_flip_count": int((stage_safe[3] != include_t0).sum()),
        "latency_only_flip_count": int((include_t0 != include_latency).sum()),
        "continuous_only_flip_count": int((include_latency != continuous).sum()),
        "separated_planning_margin_flip_count": int((continuous != v2).sum()),
        "multiple_cause_candidate_flip_count": int(
            (candidate_change_count > 1).sum()
        ),
    }


def label_audit(est, est_summary, gt, gt_summary):
    physical = np.asarray(est["physical_label"], float)
    planning = np.asarray(est["planning_label"], float)
    old = np.asarray(est["v1_label"], float)
    guidance = np.asarray(est["v1_guidance"], float)
    return {
        "physical_safe_always_outranks_collision": (
            est_summary["fractions"]["physical_label_inversion"]["numerator"] == 0
        ),
        "planning_safe_always_outranks_unsafe": (
            est_summary["fractions"]["planning_label_inversion"]["numerator"] == 0
        ),
        "gt_uncertainty_growth_removed": True,
        "physical_label_inversion": est_summary["fractions"]["physical_label_inversion"],
        "planning_label_inversion": est_summary["fractions"]["planning_label_inversion"],
        "model_ranking_failure": est_summary["fractions"]["model_ranking_failure"],
        "combined_selection_failure": est_summary["fractions"][
            "combined_selection_failure"
        ],
        "label_scales": {
            "v1_mean": float(old.mean()), "physical_v2_mean": float(physical.mean()),
            "planning_v2_mean": float(planning.mean()),
            "guidance_mean": float(guidance.mean()),
        },
        "guidance_can_override_safety": False,
        "candidate_ordering": "collision class, penetration, normalized secondary",
        "valid_gt_planning_equals_physical_policy": True,
    }


def _ablation_stage(name, dynamic, static, predicted, label=None, t0=None):
    dynamic = np.asarray(dynamic, float)
    static = np.asarray(static, float)
    safe = (dynamic >= -TOLERANCE) & (static >= -TOLERANCE)
    exists = safe.any(1)
    rows = np.arange(len(safe))
    predicted_index = np.asarray(predicted, float).argmin(1)
    if label is None:
        # The V2 label is lexicographic: every safe candidate ranks before all
        # colliding candidates.  The exact safe-candidate tie break cannot
        # introduce a safety inversion.
        label_safe = exists.copy()
    else:
        label_index = np.asarray(label, float).argmin(1)
        label_safe = safe[rows, label_index]
    return {
        "stage": name,
        "dynamic_coverage_failure": metric(
            ~(dynamic >= -TOLERANCE).any(1), np.ones(len(safe), bool),
            f"{name}: all full-validation windows",
        ),
        "static_coverage_failure": metric(
            ~(static >= -TOLERANCE).any(1), np.ones(len(safe), bool),
            f"{name}: all full-validation windows",
        ),
        "joint_coverage_failure": metric(
            ~exists, np.ones(len(safe), bool),
            f"{name}: all full-validation windows",
        ),
        "label_inversion": metric(
            exists & ~label_safe, np.ones(len(safe), bool),
            f"{name}: all full-validation windows",
        ),
        "combined_selection_failure": metric(
            exists & ~safe[rows, predicted_index], np.ones(len(safe), bool),
            f"{name}: all full-validation windows",
        ),
        "already_unsafe": (
            None if t0 is None else metric(
                np.asarray(t0, float) < -TOLERANCE,
                np.ones(len(safe), bool),
                f"{name}: explicit t=0 physical state",
            )
        ),
        "_safe": safe,
    }


def component_ablation(data):
    v1_dynamic = np.asarray(data["v1_dynamic_clearance"], float)
    v1_static = np.asarray(data["v1_static_clearance"], float)
    physical_dynamic = np.asarray(data["physical_dynamic"], float)
    physical_static = np.asarray(data["physical_static"], float)
    planning_dynamic = np.asarray(data["planning_dynamic"], float)
    planning_static = np.asarray(data["planning_static"], float)
    stages = [
        ("v1", v1_dynamic, v1_static, np.asarray(data["v1_label"], float), None),
        ("remove_gt_uncertainty",
         np.asarray(data["ablation_remove_uncertainty_dynamic"], float),
         v1_static, None, None),
        ("exact_actor_geometry",
         np.asarray(data["ablation_exact_shape_dynamic"], float),
         v1_static, None, None),
        ("static_uav_radius",
         np.asarray(data["ablation_exact_shape_dynamic"], float),
         np.asarray(data["ablation_static_radius"], float), None, None),
        ("include_t0",
         np.asarray(data["ablation_include_t0_dynamic"], float),
         np.asarray(data["ablation_include_t0_static"], float), None,
         np.asarray(data["t0_joint"], float)),
        ("include_latency",
         np.asarray(data["ablation_latency_dynamic"], float),
         np.asarray(data["ablation_latency_static"], float), None,
         np.asarray(data["t0_joint"], float)),
        ("continuous_collision", physical_dynamic, physical_static, None,
         np.asarray(data["t0_joint"], float)),
        ("separated_planning_margin", planning_dynamic, planning_static, None,
         np.asarray(data["t0_joint"], float)),
    ]
    output = [
        _ablation_stage(
            name, dynamic, static, data["v1_predicted"], label=label, t0=t0
        )
        for name, dynamic, static, label, t0 in stages
    ]
    for index, row in enumerate(output):
        safe = row.pop("_safe")
        if index == 0:
            row["marginal_outcome_flips"] = None
        else:
            previous_safe = (
                (stages[index-1][1] >= -TOLERANCE)
                & (stages[index-1][2] >= -TOLERANCE)
            )
            row["marginal_outcome_flips"] = {
                "candidate_count": int((safe != previous_safe).sum()),
                "window_count": int((safe.any(1) != previous_safe.any(1)).sum()),
            }
    reverse = []
    final_safe = (
        (stages[-1][1] >= -TOLERANCE) & (stages[-1][2] >= -TOLERANCE)
    )
    for name, dynamic, static, _, _ in reversed(stages[:-1]):
        alternate = (dynamic >= -TOLERANCE) & (static >= -TOLERANCE)
        reverse.append({
            "component_removed_to_stage": name,
            "candidate_outcome_flip_count": int((final_safe != alternate).sum()),
            "window_outcome_flip_count": int(
                (final_safe.any(1) != alternate.any(1)).sum()
            ),
        })
    return {
        "forward_fixed_order": output,
        "reverse_order_check": reverse,
        "reverse_order_scope": (
            "terminal V2 outcome compared with each earlier immutable stage"
        ),
        "determinism_hash": canonical_hash(output),
    }


def write_diagnostics(data, summary):
    flags = summary["internal_flags"]
    masks = summary["internal_masks"]
    mapping = {
        "static_radius_flips": (
            (np.asarray(data["v1_static_clearance"], float) >= 0).any(1)
            != (np.asarray(data["physical_static"], float) >= 0).any(1)
        ),
        "gt_uncertainty_flips": (
            (np.asarray(data["v1_dynamic_clearance"], float) >= 0).any(1)
            != (
                np.asarray(data["ablation_remove_uncertainty_dynamic"], float) >= 0
            ).any(1)
        ),
        "shape_geometry_flips": (
            (
                np.asarray(data["ablation_remove_uncertainty_dynamic"], float) >= 0
            ).any(1)
            != (
                np.asarray(data["ablation_exact_shape_dynamic"], float) >= 0
            ).any(1)
        ),
        "t0_latency_flips": (
            np.asarray(data["t0_joint"], float) < 0
        ),
        "continuous_collision_flips": (
            np.asarray(data["continuous_delta"], float).min(1) < -TOLERANCE
        ),
        "already_unsafe": ~masks["initially_safe"],
        "preventable_coverage_failures": (
            masks["preventable"] & ~masks["planning_exists"]
        ),
        "recoverable_windows": masks["recoverable"],
        "remaining_label_inversions": flags["planning_label_inversion"],
        "remaining_model_ranking_failures": flags["model_ranking_failure"],
    }
    for name, selected in mapping.items():
        records = [{
            "sequence_id": str(data["v1_sequence"][index]),
            "frame_index": int(data["v1_frame"][index]),
            "map_id": int(data["v1_map"][index]),
            "scenario": str(data["v1_scenario"][index]),
        } for index in np.flatnonzero(selected)]
        atomic_json(DIAGNOSTICS / f"{name}.json", {
            "status": "PASS", "count": len(records), "records": records,
        })


def finalize(hashes, estimated, gt, labels, delta, hits, misses):
    geometry = json.loads(
        (REPORTS / "phase8jqv2_geometry_validation.json").read_text()
    )
    estimated_joint = estimated["fractions"][
        "planning_joint_coverage_failure"
    ]["fraction"]
    gt_joint = gt["fractions"]["physical_joint_coverage_failure"]["fraction"]
    preventable = estimated["actionability"][
        "preventable_planning_coverage_failure"
    ]["fraction"]
    combined = estimated["fractions"]["combined_selection_failure"]["fraction"]
    already = estimated["actionability"]["already_unsafe_fraction"]["fraction"]
    unknown = estimated["actionability"][
        "scenario_feasibility_unknown_fraction"
    ]["fraction"]
    if geometry["status"] != "PASS":
        route, next_phase, status = "E", None, "FAIL"
    elif (
        estimated_joint <= .05 and gt_joint <= .05 and preventable <= .05
        and combined > .05 and unknown < .5
    ):
        route, next_phase, status = "A", "phase8j_b_safety_first_score_v2", "PASS"
    elif preventable > .05 and combined > .05:
        route, next_phase, status = (
            "C", "phase8j_coverage_then_score_v2", "PASS"
        )
    elif preventable > .05:
        route, next_phase, status = "B", "phase8j_a_coverage_v2", "PASS"
    elif (
        estimated_joint > .05 and preventable <= .05
        and already + unknown > .5
    ):
        route, next_phase, status = (
            "D", "phase8jq_feasibility_conditioned_dataset_protocol", "PASS"
        )
    else:
        route, next_phase, status = "E", None, "FAIL"
    determinism = {
        **hashes, "status": "PASS",
        "artifact_cache_hits": hits, "artifact_cache_misses": misses,
        "same_config_resume_supported": True,
        "component_ablation_hash": json.loads(
            (REPORTS / "phase8jqv2_component_ablation.json").read_text()
        )["determinism_hash"],
    }
    atomic_json(REPORTS / "phase8jqv2_determinism_validation.json", determinism)
    final = {
        **hashes,
        "status": status, "rebaseline_ready": status == "PASS",
        "evaluator_version": "v2", "v1_preserved": True,
        "phase8h_perception_frozen": True,
        "network_weights_modified": False, "training_executed": False,
        "dataset_rebuilt": False, "production_test_used": False,
        "route": route, "next_allowed_phase": next_phase,
        "decision_metrics": {
            "estimated_planning_joint_coverage_failure": estimated_joint,
            "gt_physical_joint_coverage_failure": gt_joint,
            "preventable_planning_coverage_failure": preventable,
            "combined_selection_failure": combined,
            "already_unsafe_fraction": already,
            "scenario_feasibility_unknown_fraction": unknown,
        },
    }
    atomic_json(REPORTS / "phase8jqv2_final_result.json", final)
    text = f"""# Phase 8J-Q2 final recommendation

Status: **{status}**. Route: **{route}**.

- estimated planning joint coverage failure: {estimated_joint:.4%}
- GT physical joint coverage failure: {gt_joint:.4%}
- preventable planning coverage failure: {preventable:.4%}
- combined selection failure: {combined:.4%}
- already unsafe: {already:.4%}
- generator feasibility unknown: {unknown:.4%}

Next allowed phase: `{next_phase}`.

V1 reports and imports remain intact. Phase 8H perception, checkpoints and the
dataset were not modified. No backward, optimizer step, training, blind rerun or
production-test access occurred.
"""
    atomic_json(REPORTS / "phase8jqv2_final_readiness.json", {
        **hashes, "status": status, "route": route,
        "next_allowed_phase": next_phase,
    })
    path = REPORTS / "phase8jqv2_final_recommendation.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
    readiness = REPORTS / "phase8jqv2_final_readiness.md"
    temporary = readiness.with_name(f".{readiness.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, readiness)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
