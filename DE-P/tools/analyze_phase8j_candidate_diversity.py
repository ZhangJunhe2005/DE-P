#!/usr/bin/env python3
"""Phase 8J-A pre-loss audit of the fixed 15-candidate generator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pointbiserialr, spearmanr
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_dataset import seed_dataset_worker
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset
from policy.dynamic_training_config import DynamicTrainingConfig
from tools.run_phase8i_failure_decomposition import make_runtime


TRAJECTORY_COUNT = 15
SAFETY_TOLERANCE = 1e-6
# Fixed before training and independent of valid outcomes.
ENDPOINT_NEAR_DUPLICATE_M = 0.25
TRAJECTORY_NEAR_DUPLICATE_M = 0.15
COVARIANCE_RELATIVE_RANK_TOLERANCE = 1e-3
UNUSED_PRIMITIVE_FRACTION = 0.005


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def describe(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "quantiles": {
            str(q): float(np.quantile(finite, q)) if finite.size else None
            for q in (0, .1, .25, .5, .75, .9, .95, .99, 1)
        },
    }


def connected_component_count(adjacency):
    remaining = set(range(len(adjacency)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[current])) & remaining
            remaining -= neighbors
            stack.extend(neighbors)
    return count


def correlation(flag, value):
    flag = np.asarray(flag, dtype=bool)
    value = np.asarray(value, dtype=float)
    if flag.all() or (~flag).all() or np.allclose(value, value[0]):
        return {"point_biserial": 0.0, "spearman": 0.0}
    return {
        "point_biserial": float(np.nan_to_num(pointbiserialr(flag, value).statistic)),
        "spearman": float(np.nan_to_num(spearmanr(flag.astype(float), value).statistic)),
    }


def collate(samples):
    return dynamic_sequence_collate(
        samples, max_obstacles=DynamicTrainingConfig.from_global_config().max_obstacles
    )


def datasets_and_loaders(data_root, cache_root, batch_size, workers):
    base = DynamicTrainingConfig.from_global_config()
    configs = {
        "ground_truth": replace(base, context_source="ground_truth", estimated_cache_dir=None),
        "estimated": replace(
            base, context_source="estimated", estimated_cache_dir=str(cache_root)
        ),
    }
    suites = {}
    for context, config in configs.items():
        valid = DynamicSequenceDataset(data_root, "valid", training_config=config)
        train = DynamicSequenceDataset(data_root, "train", training_config=config)
        suites[f"valid_{context}"] = (valid, list(range(len(valid))))
        decomposition = json.loads(
            (ROOT / "reports/phase8i_candidate_failure_decomposition.json").read_text()
        )
        rows = decomposition["manifests"]["production_train_frozen_stratified"]["rows"]
        lookup = {
            (sequence, int(frame)): index
            for index, (sequence, frame, _) in enumerate(train._windows)
        }
        indices = [lookup[(row["sequence_id"], int(row["frame_index"]))] for row in rows]
        if len(indices) != 432 or len(set(indices)) != 432:
            raise RuntimeError("frozen production train stratification is incomplete")
        suites[f"train_stratified_{context}"] = (train, indices)
    result = {}
    for name, (dataset, indices) in suites.items():
        subset = Subset(dataset, indices)
        result[name] = (
            dataset,
            DataLoader(
                subset, batch_size=batch_size, shuffle=False, num_workers=workers,
                collate_fn=collate, worker_init_fn=seed_dataset_worker,
                pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
            ),
        )
    return result


@torch.inference_mode()
def collect(trainer, dataset, loader):
    result = {name: [] for name in (
        "sequence", "frame", "map", "scenario", "category", "trajectory",
        "endstate", "predicted_score", "dynamic_clearance", "static_clearance",
        "goal_world", "start_world", "attention_sum",
    )}
    scenario = {
        sequence: str(metadata.get("scenario_type", metadata.get("scenario_id", "unknown")))
        for sequence, (_, metadata, _) in dataset._sequences.items()
    }
    trainer.policy.eval()
    for batch in loader:
        context = batch["dynamic_context"].to(trainer.device)
        obstacles = batch["dynamic_obstacles"].to(trainer.device)
        details = trainer._forward_details(
            batch["current_depth"], batch["position_world"],
            batch["rotation_world_from_body"], batch["observation_9d"], batch["map_id"],
            context, obstacles, trainer.dynamic_dep_loss,
        )
        size = len(batch["sequence_id"])
        fixed = details["start_state_world_expanded"].permute(0, 2, 1)
        end = details["end_state_world_expanded"].permute(0, 2, 1)
        trajectories, _ = trainer.dynamic_dep_loss.safety_loss.trajectory_sampler.grouped(
            fixed, end, size
        )
        _, static_distance = trainer.dynamic_dep_loss.safety_loss.get_distance_cost(
            trajectories.reshape(size, -1, 3), details["batch_map_id"]
        )
        arrays = {
            "trajectory": trajectories.detach().cpu().numpy(),
            "endstate": details["end_state_world_expanded"].reshape(
                size, TRAJECTORY_COUNT, 3, 3
            ).detach().cpu().numpy(),
            "predicted_score": details["predicted_score"].reshape(
                size, TRAJECTORY_COUNT
            ).detach().cpu().numpy(),
            "dynamic_clearance": details[
                "dynamic_diagnostics"
            ].candidate_min_clearance.detach().cpu().numpy(),
            "static_clearance": static_distance.reshape(
                size, TRAJECTORY_COUNT, -1
            ).amin(2).detach().cpu().numpy(),
            "goal_world": batch["goal_world"].numpy(),
            "start_world": batch["position_world"].numpy(),
            "attention_sum": context.attention().sum(dim=(1, 2, 3)).detach().cpu().numpy(),
        }
        for row in range(size):
            for name in arrays:
                result[name].append(arrays[name][row])
            sequence_id = batch["sequence_id"][row]
            result["sequence"].append(sequence_id)
            result["frame"].append(int(batch["frame_index"][row]))
            result["map"].append(int(batch["map_id"][row]))
            result["scenario"].append(scenario[sequence_id])
            result["category"].append(batch["sample_category"][row])
    return {
        name: np.asarray(values, dtype=object if name in {
            "sequence", "scenario", "category"
        } else None)
        for name, values in result.items()
    }


def analyze(data):
    trajectories = np.asarray(data["trajectory"], dtype=float)
    endstate = np.asarray(data["endstate"], dtype=float)
    endpoints = endstate[:, :, 0]
    velocities = endstate[:, :, 1]
    accelerations = endstate[:, :, 2]
    scores = np.asarray(data["predicted_score"], dtype=float)
    dynamic_clearance = np.asarray(data["dynamic_clearance"], dtype=float)
    static_clearance = np.asarray(data["static_clearance"], dtype=float)
    dynamic_safe = dynamic_clearance >= -SAFETY_TOLERANCE
    static_safe = static_clearance >= -SAFETY_TOLERANCE
    joint_safe = dynamic_safe & static_safe
    windows = len(trajectories)
    upper = np.triu_indices(TRAJECTORY_COUNT, 1)
    endpoint_mean, endpoint_min = [], []
    trajectory_mean, trajectory_min = [], []
    velocity_spread, acceleration_spread = [], []
    near_duplicate_fraction, effective_unique, covariance_rank = [], [], []
    lateral_spread, forward_spread, vertical_spread, midpoint_spread = [], [], [], []
    radial_min, radial_max, velocity_max, acceleration_max = [], [], [], []
    oracle_index = []
    for row in range(windows):
        endpoint_pair = np.linalg.norm(
            endpoints[row, :, None] - endpoints[row, None, :], axis=-1
        )
        trajectory_pair = np.linalg.norm(
            trajectories[row, :, None] - trajectories[row, None, :], axis=-1
        ).mean(-1)
        velocity_pair = np.linalg.norm(
            velocities[row, :, None] - velocities[row, None, :], axis=-1
        )
        acceleration_pair = np.linalg.norm(
            accelerations[row, :, None] - accelerations[row, None, :], axis=-1
        )
        endpoint_values = endpoint_pair[upper]
        trajectory_values = trajectory_pair[upper]
        endpoint_mean.append(endpoint_values.mean())
        endpoint_min.append(endpoint_values.min())
        trajectory_mean.append(trajectory_values.mean())
        trajectory_min.append(trajectory_values.min())
        velocity_spread.append(velocity_pair[upper].mean())
        acceleration_spread.append(acceleration_pair[upper].mean())
        duplicates = (
            (endpoint_pair < ENDPOINT_NEAR_DUPLICATE_M)
            & (trajectory_pair < TRAJECTORY_NEAR_DUPLICATE_M)
            & ~np.eye(TRAJECTORY_COUNT, dtype=bool)
        )
        near_duplicate_fraction.append(duplicates[upper].mean())
        effective_unique.append(connected_component_count(duplicates | np.eye(
            TRAJECTORY_COUNT, dtype=bool
        )))
        centered = endpoints[row] - endpoints[row].mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        covariance_rank.append(int(np.sum(
            singular > max(singular[0] * COVARIANCE_RELATIVE_RANK_TOLERANCE, 1e-8)
        )))
        direction = np.asarray(data["goal_world"][row], float) - np.asarray(
            data["start_world"][row], float
        )
        horizontal = direction.copy()
        horizontal[2] = 0
        if np.linalg.norm(horizontal) < 1e-8:
            horizontal = np.array([1.0, 0.0, 0.0])
        forward = horizontal / np.linalg.norm(horizontal)
        lateral = np.array([-forward[1], forward[0], 0.0])
        delta = endpoints[row] - np.asarray(data["start_world"][row], float)
        lateral_spread.append(np.ptp(delta @ lateral))
        forward_spread.append(np.ptp(delta @ forward))
        vertical_spread.append(np.ptp(delta[:, 2]))
        midpoint = trajectories[row, :, trajectories.shape[2] // 2]
        midpoint_pair = np.linalg.norm(
            midpoint[:, None] - midpoint[None, :], axis=-1
        )
        midpoint_spread.append(midpoint_pair[upper].mean())
        radius = np.linalg.norm(delta, axis=1)
        radial_min.append(radius.min())
        radial_max.append(radius.max())
        velocity_max.append(np.linalg.norm(velocities[row], axis=1).max())
        acceleration_max.append(np.linalg.norm(accelerations[row], axis=1).max())
        dynamic_violation = np.maximum(-dynamic_clearance[row], 0)
        static_violation = np.maximum(-static_clearance[row], 0)
        secondary = np.linalg.norm(endpoints[row] - data["goal_world"][row], axis=1)
        order = np.lexsort((secondary, static_violation, dynamic_violation, ~joint_safe[row]))
        oracle_index.append(int(order[0]))
    selected = scores.argmin(1)
    joint_failure = ~joint_safe.any(1)
    dynamic_failure = ~dynamic_safe.any(1)
    metrics = {
        "window_count": windows,
        "endpoint_pairwise_distance_m": describe(endpoint_mean),
        "endpoint_min_pairwise_distance_m": describe(endpoint_min),
        "trajectory_pairwise_distance_m": describe(trajectory_mean),
        "trajectory_min_pairwise_distance_m": describe(trajectory_min),
        "endpoint_velocity_pairwise_spread_mps": describe(velocity_spread),
        "endpoint_acceleration_pairwise_spread_mps2": describe(acceleration_spread),
        "near_duplicate_candidate_pair_fraction": describe(near_duplicate_fraction),
        "effective_unique_candidate_count": describe(effective_unique),
        "candidate_endpoint_covariance_rank": describe(covariance_rank),
        "lateral_endpoint_spread_m": describe(lateral_spread),
        "forward_endpoint_spread_m": describe(forward_spread),
        "vertical_endpoint_spread_m": describe(vertical_spread),
        "mid_trajectory_pairwise_spread_m": describe(midpoint_spread),
        "endpoint_radius_min_m": describe(radial_min),
        "endpoint_radius_max_m": describe(radial_max),
        "endpoint_velocity_max_mps": describe(velocity_max),
        "endpoint_acceleration_max_mps2": describe(acceleration_max),
        "dynamic_safe_candidate_count": describe(dynamic_safe.sum(1)),
        "joint_safe_candidate_count": describe(joint_safe.sum(1)),
        "dynamic_coverage_failure_fraction": float(dynamic_failure.mean()),
        "joint_coverage_failure_fraction": float(joint_failure.mean()),
        "selected_primitive_frequency": {
            str(index): int((selected == index).sum()) for index in range(TRAJECTORY_COUNT)
        },
        "oracle_primitive_frequency": {
            str(index): int((np.asarray(oracle_index) == index).sum())
            for index in range(TRAJECTORY_COUNT)
        },
        "unused_selected_primitives": [
            index for index in range(TRAJECTORY_COUNT)
            if float((selected == index).mean()) < UNUSED_PRIMITIVE_FRACTION
        ],
        "coverage_diversity_correlation": {
            "endpoint_pairwise": correlation(joint_failure, endpoint_mean),
            "endpoint_min_pairwise": correlation(joint_failure, endpoint_min),
            "trajectory_pairwise": correlation(joint_failure, trajectory_mean),
            "trajectory_min_pairwise": correlation(joint_failure, trajectory_min),
            "effective_unique_count": correlation(joint_failure, effective_unique),
            "lateral_spread": correlation(joint_failure, lateral_spread),
            "mid_trajectory_spread": correlation(joint_failure, midpoint_spread),
        },
    }
    groups = {}
    for group_name, group_values in (
        ("scenario", data["scenario"]), ("map", data["map"]), ("category", data["category"])
    ):
        groups[group_name] = {}
        for value in np.unique(group_values):
            mask = group_values == value
            groups[group_name][str(value)] = {
                "window_count": int(mask.sum()),
                "joint_coverage_failure_fraction": float(joint_failure[mask].mean()),
                "endpoint_pairwise_distance_mean_m": float(np.mean(np.asarray(endpoint_mean)[mask])),
                "trajectory_pairwise_distance_mean_m": float(
                    np.mean(np.asarray(trajectory_mean)[mask])
                ),
                "lateral_endpoint_spread_mean_m": float(
                    np.mean(np.asarray(lateral_spread)[mask])
                ),
                "mid_trajectory_spread_mean_m": float(
                    np.mean(np.asarray(midpoint_spread)[mask])
                ),
                "near_duplicate_pair_fraction_mean": float(
                    np.mean(np.asarray(near_duplicate_fraction)[mask])
                ),
                "effective_unique_candidate_count_mean": float(
                    np.mean(np.asarray(effective_unique)[mask])
                ),
                "joint_safe_candidate_count_mean": float(joint_safe[mask].sum(1).mean()),
            }
        metrics[f"by_{group_name}"] = groups[group_name]
    # Small deterministic examples for offline audit.
    example_indices = np.argsort(np.asarray(effective_unique))[:12]
    examples = [{
        "sequence_id": str(data["sequence"][row]),
        "frame_index": int(data["frame"][row]),
        "map_id": int(data["map"][row]),
        "scenario": str(data["scenario"][row]),
        "effective_unique_candidate_count": int(effective_unique[row]),
        "near_duplicate_pair_fraction": float(near_duplicate_fraction[row]),
        "joint_coverage_failure": bool(joint_failure[row]),
        "selected_primitive": int(selected[row]),
        "oracle_primitive": int(oracle_index[row]),
        "endpoints_world_m": endpoints[row].tolist(),
    } for row in example_indices]
    return metrics, examples


def build_markdown(payload):
    estimated = payload["suites"]["valid_estimated"]
    gt = payload["suites"]["valid_ground_truth"]
    corr = estimated["coverage_diversity_correlation"]
    unused = estimated["unused_selected_primitives"]
    lines = [
        "# Phase 8J Candidate Collapse Audit", "",
        "本报告在加入任何 coverage/diversity loss 前生成。安全阈值保持 Phase 8I；"
        "near-duplicate 阈值预先固定为 endpoint < 0.25 m 且完整轨迹平均距离 < 0.15 m。",
        "",
        "## 结论", "",
    ]
    endpoint_corr = corr["endpoint_pairwise"]["point_biserial"]
    trajectory_corr = corr["trajectory_pairwise"]["point_biserial"]
    unique_mean = estimated["effective_unique_candidate_count"]["mean"]
    duplicate_mean = estimated["near_duplicate_candidate_pair_fraction"]["mean"]
    if duplicate_mean > .1 or unique_mean < 12:
        collapse = "存在明显 candidate collapse"
    else:
        collapse = "不存在全局严重 duplicate collapse"
    lines += [
        f"- {collapse}：effective unique mean={unique_mean:.3f}/15，"
        f"near-duplicate pair mean={duplicate_mean:.4f}。",
        f"- coverage failure 与 endpoint diversity 的 point-biserial={endpoint_corr:.3f}，"
        f"与 full-trajectory diversity={trajectory_corr:.3f}。",
        f"- GT/estimated joint coverage 分别为 "
        f"{gt['joint_coverage_failure_fraction']:.4f}/"
        f"{estimated['joint_coverage_failure_fraction']:.4f}。",
        f"- score 长期未选择的 primitive：{unused if unused else '无'}。", "",
        "## 六个问题", "",
        "1. coverage failure 是否由候选集中到相似轨迹：见上述 duplicate、unique 和"
        "相关系数；只有相关性与失败场景同时支持时才判为主要原因。",
        "2. endpoint 与中间段：报告同时给出 endpoint、mid-trajectory 和完整轨迹"
        "距离，避免只依据终点。",
        "3. crossing/multi-target/occluded：按场景记录 lateral、mid-trajectory、"
        "safe count 与 coverage，可据此选择只对过近 pair 的 repulsion。",
        f"4. primitive 使用：selected frequency 已完整记录；未使用列表为 {unused}。",
        "5. 多 primitive 是否相同：由 near-duplicate pair fraction 和固定 examples"
        "直接证明。",
        "6. 输出压缩：candidate position 由 tanh 后径向映射到 [0,10] m，速度/加速度"
        "由 tanh 压到 ±6；报告记录 radius、velocity、acceleration 的边界分布。", "",
        "在此审计前未增加 diversity loss，也未执行 optimizer.step。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/phase8_dynamic_production"))
    parser.add_argument("--cache", default=str(ROOT / "cache/phase8i_phase8h_perception"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs/phase8d_shakedown/fixed_050_seed8403/checkpoints/best_dynamic.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    entry = json.loads((ROOT / "reports/phase8j_entry_gate.json").read_text())
    if entry["status"] != "PASS":
        raise RuntimeError("Phase 8J entry Gate is not PASS")
    runtime = make_runtime(
        Path(args.dataset) / "map_catalog.yaml", ROOT / "configs/static_map_catalog.yaml"
    )
    load_dep_checkpoint(runtime.policy, args.checkpoint, "corrected")
    suites = datasets_and_loaders(
        Path(args.dataset), Path(args.cache), args.batch_size, args.workers
    )
    output = {
        "status": "PASS",
        "analysis_version": "phase8j_candidate_diversity_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "production_test_used": False,
        "optimizer_step_executed": False,
        "thresholds": {
            "safety_tolerance_m": SAFETY_TOLERANCE,
            "endpoint_near_duplicate_m": ENDPOINT_NEAR_DUPLICATE_M,
            "trajectory_near_duplicate_m": TRAJECTORY_NEAR_DUPLICATE_M,
            "covariance_relative_rank_tolerance": COVARIANCE_RELATIVE_RANK_TOLERANCE,
            "unused_primitive_fraction": UNUSED_PRIMITIVE_FRACTION,
        },
        "suites": {},
    }
    all_examples = {}
    for name, (dataset, loader) in suites.items():
        metrics, examples = analyze(collect(runtime, dataset, loader))
        output["suites"][name] = metrics
        all_examples[name] = examples
    atomic_json(ROOT / "reports/phase8j_candidate_diversity_baseline.json", output)
    report = ROOT / "reports/phase8j_candidate_collapse_audit.md"
    temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temporary.write_text(build_markdown(output))
    os.replace(temporary, report)
    atomic_json(ROOT / "diagnostics/phase8j/candidate_collapse_examples.json", {
        "status": "PASS", "examples_by_suite": all_examples,
    })
    print(json.dumps({
        "status": "PASS",
        "report": str(report),
        "valid_estimated": output["suites"]["valid_estimated"],
    }, indent=2))


if __name__ == "__main__":
    main()
