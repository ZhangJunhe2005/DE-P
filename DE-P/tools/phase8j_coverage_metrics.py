"""Full-validation metrics for Phase 8J-A candidate coverage."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from tools.run_phase8i_failure_decomposition import scenario_lookup


TOLERANCE = 1e-6


def _fraction(values):
    values = np.asarray(values, dtype=bool)
    return float(values.mean()) if len(values) else 0.0


def _sequence_bootstrap(flags, sequences, seed=8810, draws=1000):
    flags = np.asarray(flags, dtype=float)
    sequences = np.asarray(sequences, dtype=str)
    names = np.unique(sequences)
    per_sequence = {name: flags[sequences == name] for name in names}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        selected = rng.choice(names, len(names), replace=True)
        values = np.concatenate([per_sequence[name] for name in selected])
        estimates.append(values.mean())
    return {
        "point_estimate": float(flags.mean()),
        "sequence_bootstrap_95_ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "sequence_count": int(len(names)),
    }


def _groups(keys, metrics):
    result = {}
    keys = np.asarray(keys)
    for key in np.unique(keys):
        mask = keys == key
        result[str(key)] = {
            name: float(np.asarray(values)[mask].mean())
            for name, values in metrics.items()
        }
        result[str(key)]["window_count"] = int(mask.sum())
    return result


@torch.inference_mode()
def evaluate_dynamic(trainer, loader):
    trainer.policy.eval()
    scenarios = scenario_lookup(loader.dataset)
    values = defaultdict(list)
    for batch in loader:
        details = trainer.compute_batch("dynamic", batch)
        count = len(batch["sequence_id"])
        diagnostics = details["dynamic_diagnostics"]
        dynamic_clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()
        static_clearance = details["candidate_static_clearance"].detach().cpu().numpy()
        dynamic_violation = np.maximum(-dynamic_clearance, 0.0)
        static_violation = np.maximum(-static_clearance, 0.0)
        dynamic_safe = (
            (dynamic_violation <= TOLERANCE)
            & (dynamic_clearance >= -TOLERANCE)
        )
        static_safe = (
            (static_violation <= TOLERANCE)
            & (static_clearance >= -TOLERANCE)
        )
        joint_safe = dynamic_safe & static_safe
        trajectories = details["candidate_trajectories_world"].detach().cpu().numpy()
        endpoints = trajectories[:, :, -1]
        start = details["start_state_world_expanded"].reshape(
            count, 15, 3, 3
        )[:, 0, 0].detach().cpu().numpy()
        target_count = diagnostics.dynamic_obstacle_count.detach().cpu().numpy()
        dynamic_cost = diagnostics.candidate_cost.detach().cpu().numpy()
        for row in range(count):
            endpoint_pair = np.linalg.norm(
                endpoints[row, :, None] - endpoints[row, None, :], axis=-1
            )
            trajectory_pair = np.linalg.norm(
                trajectories[row, :, None] - trajectories[row, None, :], axis=-1
            ).mean(-1)
            upper = np.triu_indices(15, 1)
            near = (
                (endpoint_pair[upper] < 0.25)
                & (trajectory_pair[upper] < 0.15)
            )
            values["sequence"].append(batch["sequence_id"][row])
            values["map"].append(int(batch["map_id"][row]))
            values["category"].append(batch["sample_category"][row])
            values["scenario"].append(scenarios[batch["sequence_id"][row]])
            values["dynamic_failure"].append(not dynamic_safe[row].any())
            values["joint_failure"].append(not joint_safe[row].any())
            values["joint_count"].append(int(joint_safe[row].sum()))
            values["dynamic_count"].append(int(dynamic_safe[row].sum()))
            values["endpoint_spread"].append(float(endpoint_pair[upper].mean()))
            values["trajectory_spread"].append(float(trajectory_pair[upper].mean()))
            values["near_duplicate"].append(float(near.mean()))
            values["no_target_dynamic_max"].append(
                float(np.abs(dynamic_cost[row]).max()) if target_count[row] == 0 else 0.0
            )
            values["max_endpoint_radius"].append(float(
                np.linalg.norm(endpoints[row] - start[row], axis=1).max()
            ))
        for name in ("guidance_loss", "smooth_loss", "static_safety_loss"):
            values[name].append(float(details[name].detach().cpu()))

    joint_count = np.asarray(values["joint_count"])
    metrics = {
        "window_count": len(values["sequence"]),
        "dynamic_coverage_failure_fraction": _sequence_bootstrap(
            values["dynamic_failure"], values["sequence"]
        ),
        "joint_coverage_failure_fraction": _sequence_bootstrap(
            values["joint_failure"], values["sequence"]
        ),
        "oracle_dynamic_collision_fraction": _fraction(values["dynamic_failure"]),
        "oracle_joint_collision_fraction": _fraction(values["joint_failure"]),
        "joint_safe_candidate_count": {
            "mean": float(joint_count.mean()),
            "median": float(np.median(joint_count)),
            "q10": float(np.quantile(joint_count, 0.10)),
        },
        "dynamic_safe_candidate_count_mean": float(np.mean(values["dynamic_count"])),
        "endpoint_pairwise_distance_mean_m": float(np.mean(values["endpoint_spread"])),
        "trajectory_pairwise_distance_mean_m": float(np.mean(values["trajectory_spread"])),
        "near_duplicate_candidate_pair_fraction": float(np.mean(values["near_duplicate"])),
        "no_target_dynamic_cost_max_abs": float(max(values["no_target_dynamic_max"], default=0)),
        "maximum_endpoint_radius_m": float(max(values["max_endpoint_radius"], default=0)),
        "guidance_mean": float(np.mean(values["guidance_loss"])),
        "smoothness_mean": float(np.mean(values["smooth_loss"])),
        "static_cost_mean": float(np.mean(values["static_safety_loss"])),
        "finite": bool(all(np.isfinite(np.asarray(item, dtype=float)).all() for item in (
            values["joint_count"], values["endpoint_spread"], values["trajectory_spread"],
            values["guidance_loss"], values["smooth_loss"], values["static_safety_loss"],
        ))),
    }
    group_metrics = {
        "dynamic_coverage_failure_fraction": values["dynamic_failure"],
        "joint_coverage_failure_fraction": values["joint_failure"],
        "joint_safe_candidate_count_mean": values["joint_count"],
    }
    metrics["by_map"] = _groups(values["map"], group_metrics)
    metrics["by_scenario"] = _groups(values["scenario"], group_metrics)
    metrics["by_category"] = _groups(values["category"], group_metrics)
    return metrics


@torch.inference_mode()
def evaluate_static(trainer, loader):
    trainer.policy.eval()
    failure, counts, guidance, smoothness, safety = [], [], [], [], []
    finite = True
    maximum_radius = 0.0
    for batch in loader:
        details = trainer.compute_batch("static", batch)
        clearance = details["candidate_static_clearance"].detach().cpu().numpy()
        safe = (np.maximum(-clearance, 0) <= TOLERANCE) & (clearance >= -TOLERANCE)
        failure.extend(~safe.any(1))
        counts.extend(safe.sum(1))
        guidance.append(float(details["guidance_loss"].detach().cpu()))
        smoothness.append(float(details["smooth_loss"].detach().cpu()))
        safety.append(float(details["static_safety_loss"].detach().cpu()))
        trajectories = details["candidate_trajectories_world"]
        batch_size = trajectories.shape[0]
        start = details["start_state_world_expanded"].reshape(
            batch_size, 15, 3, 3
        )[:, 0, 0]
        maximum_radius = max(maximum_radius, float(
            torch.linalg.vector_norm(trajectories[:, :, -1] - start[:, None], dim=-1).max()
        ))
        finite = finite and all(bool(torch.isfinite(details[name]).all()) for name in (
            "guidance_loss", "smooth_loss", "static_safety_loss",
            "candidate_trajectories_world",
        ))
    counts = np.asarray(counts)
    return {
        "window_count": len(failure),
        "static_joint_coverage_failure_fraction": _fraction(failure),
        "static_safe_candidate_count_median": float(np.median(counts)),
        "static_safe_candidate_count_q10": float(np.quantile(counts, .1)),
        "guidance_mean": float(np.mean(guidance)),
        "smoothness_mean": float(np.mean(smoothness)),
        "static_cost_mean": float(np.mean(safety)),
        "maximum_endpoint_radius_m": maximum_radius,
        "finite": finite,
        "static_smoke_pass": bool(finite and maximum_radius <= 10.0 + 1e-4),
    }


def evaluate_all(trainer):
    return {
        "valid_gt": evaluate_dynamic(
            trainer, trainer.validation_suites["valid_gt"]
        ),
        "valid_estimated": evaluate_dynamic(
            trainer, trainer.validation_suites["valid_estimated"]
        ),
        "valid_static": evaluate_static(
            trainer, trainer.validation_suites["valid_static"]
        ),
    }
