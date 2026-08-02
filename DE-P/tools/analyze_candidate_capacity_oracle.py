#!/usr/bin/env python3
"""Bounded O0--O4 candidate-capacity oracle for Phase 8J-R.

The tool never trains network weights.  Future actor ground truth is used only
inside this offline oracle, while network inputs remain the frozen causal
estimated/GT contexts supplied by the validation datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from loss.coverage_loss import CoverageObjectiveConfig
from loss.dynamic_types import (
    DynamicLossConfig,
    DynamicObjectiveConfig,
    DynamicObstacleBatch,
    RiskMetricsConfig,
)
from loss.trajectory_sampler import QuinticTrajectorySampler
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dynamic_training_config import DynamicTrainingConfig
from policy.phase8j_coverage_trainer import Phase8JCoverageTrainer
from policy.state_transform import rotate_body2world, state_body2world
from policy.training_schedule import TrainingScheduleConfig
from tools.generate_phase8i_estimated_cache import phase8h_perception_config
from tools.run_phase8i_failure_decomposition import scenario_lookup


TOLERANCE = 1e-6
DENSE_LEVELS = (64, 128, 256, 512)
TEMPORAL_PROFILES = (
    {"name": "pass_before", "duration_scale": 0.65, "velocity_scale": 1.0,
     "acceleration_scale": 1.0, "radius_scale": 1.0},
    {"name": "faster", "duration_scale": 0.80, "velocity_scale": 1.0,
     "acceleration_scale": 1.0, "radius_scale": 1.0},
    {"name": "nominal", "duration_scale": 1.00, "velocity_scale": 1.0,
     "acceleration_scale": 1.0, "radius_scale": 1.0},
    {"name": "slower_yield", "duration_scale": 1.25, "velocity_scale": 0.5,
     "acceleration_scale": 0.5, "radius_scale": 0.8},
    {"name": "brake", "duration_scale": 1.25, "velocity_scale": 0.0,
     "acceleration_scale": 0.0, "radius_scale": 0.55},
    {"name": "delayed_pass_after", "duration_scale": 1.50,
     "velocity_scale": 0.0, "acceleration_scale": 0.0, "radius_scale": 0.45},
)
EXTENDED_PROFILES = (
    {"name": "extended_yield", "duration_scale": 1.75,
     "velocity_scale": 0.0, "acceleration_scale": 0.0, "radius_scale": 0.65},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def slice_obstacles(value: DynamicObstacleBatch, indices) -> DynamicObstacleBatch:
    return DynamicObstacleBatch(**{
        name: (
            getattr(value, name)[indices]
            if getattr(value, name) is not None else None
        )
        for name in value.__dataclass_fields__
    })


def concatenate_obstacles(values: list[DynamicObstacleBatch]) -> DynamicObstacleBatch:
    obstacle_fields = {
        "positions_world", "velocities_world", "position_covariances",
        "radii", "track_timestamps", "confidence", "valid_mask",
        "dynamic_mask", "observable_mask", "ever_observed_in_history",
        "future_positions_world", "future_valid_mask",
        "future_visibility_mask",
    }
    maximum = max(value.max_obstacles for value in values)
    combined = {}
    for name in values[0].__dataclass_fields__:
        tensors = [getattr(value, name) for value in values]
        if all(tensor is None for tensor in tensors):
            combined[name] = None
            continue
        template = next(tensor for tensor in tensors if tensor is not None)
        normalized = []
        for value, tensor in zip(values, tensors):
            if tensor is not None:
                normalized.append(tensor)
                continue
            shape = list(template.shape)
            shape[0] = value.batch_size
            if name in obstacle_fields:
                shape[1] = value.max_obstacles
            normalized.append(torch.zeros(
                shape, device=template.device, dtype=template.dtype
            ))
        tensors = normalized
        if name in obstacle_fields:
            padded = []
            for tensor in tensors:
                shape = list(tensor.shape)
                shape[1] = maximum
                destination = torch.zeros(
                    shape, device=tensor.device, dtype=tensor.dtype
                )
                destination[:, :tensor.shape[1]] = tensor
                padded.append(destination)
            tensors = padded
        combined[name] = torch.cat(tensors, dim=0)
    return DynamicObstacleBatch(**combined)


def make_trainer() -> tuple[Phase8JCoverageTrainer, dict]:
    matrix = YAML(typ="safe").load(
        ROOT / "configs/phase8j_coverage_matrix.yaml"
    )
    base = YAML(typ="safe").load(Path(matrix["base_config"]))
    cache = json.loads(Path(matrix["estimated_cache_validation"]).read_text())
    training = dict(base["dynamic_training"])
    training.update({
        "estimated_cache_dir": cache["cache_root"],
        "context_source": "estimated",
        "curriculum": [{
            "start_epoch": 0, "context_source": "estimated", "ratio": 1.0,
        }],
    })
    coverage = dict(matrix["common"])
    coverage.update(matrix["ablations"]["C_k3"])
    trainer = Phase8JCoverageTrainer(
        learning_rate=float(matrix["learning_rate"]),
        batch_size=16,
        loss_weight=[1.0, 0.0],
        tensorboard_path=str(ROOT / "runs/phase8jr_capacity_oracle"),
        checkpoint_path=matrix["baseline_checkpoint"],
        backbone_variant="corrected",
        dataset_mode="mixed",
        dynamic_data_root=base["dataset_root"],
        freeze_policy=base["freeze_policy"],
        random_seed=8513,
        num_workers=4,
        training_config_override=DynamicTrainingConfig.from_mapping(training),
        dynamic_loss_config_override=DynamicLossConfig.from_mapping(
            base["dynamic_loss"]
        ),
        dynamic_objective_config_override=DynamicObjectiveConfig.from_mapping(
            base["dynamic_objective"]
        ),
        risk_metrics_config_override=RiskMetricsConfig.from_mapping(
            base["risk_metrics"]
        ),
        static_map_catalog_override=base["static_map_catalog"],
        dynamic_map_catalog_override=base["dynamic_map_catalog"],
        training_schedule_override=TrainingScheduleConfig.from_mapping(
            base["training_schedule"]
        ),
        static_cache_size=int(base["static_dataset"]["cache_size"]),
        coverage_config=CoverageObjectiveConfig.from_mapping(coverage),
        dynamic_perception_config_override=phase8h_perception_config(),
        estimated_cache_perception_config_override=phase8h_perception_config(),
    )
    checkpoint = Path(
        json.loads(
            (ROOT / "reports/phase8j_coverage_gate.json").read_text()
        )["representative"]["checkpoint"]
    )
    load_dep_checkpoint(trainer.policy, checkpoint, "corrected")
    trainer.policy.eval()
    return trainer, {
        "matrix": str((ROOT / "configs/phase8j_coverage_matrix.yaml").resolve()),
        "matrix_sha256": sha256(ROOT / "configs/phase8j_coverage_matrix.yaml"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_manifest_sha256": sha256(
            ROOT / "data/phase8_dynamic_production/dataset_manifest.yaml"
        ),
        "estimated_cache_index_sha256": cache["index_sha256"],
    }


def current_raw_predictions(trainer, batch):
    depth = batch["current_depth"].to(trainer.device)
    observation = batch["observation_9d"].to(trainer.device).clone()
    context = batch["dynamic_context"].to(trainer.device)
    normalized = trainer.policy.state_transform.normalize_obs(observation)
    primitive_observation = trainer.policy.state_transform.prepare_input(normalized)
    raw, _ = trainer.policy.forward(
        depth, primitive_observation, dynamic_context=context
    )
    return raw.permute(0, 2, 3, 1).reshape(depth.shape[0], 15, 9)


def raw_to_body(trainer, raw, primitive_ids):
    primitive = trainer.policy.state_transform.lattice_primitive
    ids = primitive_ids.to(raw.device)
    yaw, pitch = primitive.getAngleLattice()
    yaw = yaw.to(raw.device, raw.dtype).flip(0)[ids]
    pitch = pitch.to(raw.device, raw.dtype).flip(0)[ids]
    rotation = primitive.getRotation().to(raw.device, raw.dtype).flip(0)[ids]
    yaw = yaw[None].expand(raw.shape[0], -1)
    pitch = pitch[None].expand(raw.shape[0], -1)
    rotation = rotation[None].expand(raw.shape[0], -1, -1, -1)
    delta_yaw = raw[:, :, 0] * primitive.yaw_diff
    delta_pitch = raw[:, :, 1] * primitive.pitch_diff
    radius = (raw[:, :, 2] + 1.0) * primitive.radio_range
    cosine = torch.cos(pitch + delta_pitch)
    position = torch.stack((
        cosine * torch.cos(yaw + delta_yaw) * radius,
        cosine * torch.sin(yaw + delta_yaw) * radius,
        torch.sin(pitch + delta_pitch) * radius,
    ), dim=-1)
    velocity = torch.matmul(
        rotation, (raw[:, :, 3:6] * primitive.vel_max).unsqueeze(-1)
    ).squeeze(-1)
    acceleration = torch.matmul(
        rotation, (raw[:, :, 6:9] * primitive.acc_max).unsqueeze(-1)
    ).squeeze(-1)
    return position, velocity, acceleration


def derivatives_world(trainer, raw, primitive_ids, position, rotation, observation):
    body_position, body_velocity, body_acceleration = raw_to_body(
        trainer, raw, primitive_ids
    )
    batch, candidates = raw.shape[:2]
    start_position = position.to(trainer.device)
    rotation = rotation.to(trainer.device)
    observation = observation.to(trainer.device)
    _, start_velocity, start_acceleration = state_body2world(
        start_position, rotation, observation[:, 6:9],
        observation[:, 0:3], observation[:, 3:6],
    )
    fixed = torch.stack(
        (start_position, start_velocity, start_acceleration), dim=2
    )
    end_position = (
        rotate_body2world(
            rotation[:, None].expand(-1, candidates, -1, -1),
            body_position,
        ) + start_position[:, None]
    )
    end_velocity = rotate_body2world(
        rotation[:, None].expand(-1, candidates, -1, -1), body_velocity
    )
    end_acceleration = rotate_body2world(
        rotation[:, None].expand(-1, candidates, -1, -1), body_acceleration
    )
    predicted = torch.stack(
        (end_position, end_velocity, end_acceleration), dim=3
    )
    fixed = fixed[:, None].expand(-1, candidates, -1, -1)
    # sampler convention is [K, xyz, pva]
    return (
        fixed.reshape(batch * candidates, 3, 3),
        predicted.reshape(batch * candidates, 3, 3),
    )


def sampled_kinematics(sampler, fixed, predicted):
    coefficient = sampler.coefficients(fixed, predicted)
    dt = sampler.duration / sampler.eval_points
    times = torch.linspace(
        dt, sampler.duration, sampler.eval_points,
        device=fixed.device, dtype=fixed.dtype,
    )
    velocity, acceleration = [], []
    for axis in range(3):
        values = coefficient[:, 6 * axis:6 * (axis + 1)]
        t = times[None]
        velocity.append(
            values[:, 1, None] + 2 * values[:, 2, None] * t
            + 3 * values[:, 3, None] * t.square()
            + 4 * values[:, 4, None] * t.pow(3)
            + 5 * values[:, 5, None] * t.pow(4)
        )
        acceleration.append(
            2 * values[:, 2, None] + 6 * values[:, 3, None] * t
            + 12 * values[:, 4, None] * t.square()
            + 20 * values[:, 5, None] * t.pow(3)
        )
    return (
        torch.stack(velocity, dim=-1),
        torch.stack(acceleration, dim=-1),
    )


def interpolated_future(obstacles, relative_times):
    """Interpolate recorded GT; extrapolate past its bounded recording horizon."""
    batch, obstacle_count = obstacles.positions_world.shape[:2]
    result = torch.zeros(
        batch, obstacle_count, relative_times.numel(), 3,
        device=obstacles.positions_world.device,
        dtype=obstacles.positions_world.dtype,
    )
    valid = torch.zeros(
        batch, obstacle_count, relative_times.numel(),
        device=obstacles.positions_world.device, dtype=torch.bool,
    )
    extrapolated = torch.zeros(
        batch, relative_times.numel(),
        device=obstacles.positions_world.device, dtype=torch.bool,
    )
    for row in range(batch):
        source_times = torch.cat((
            relative_times.new_zeros(1),
            (
                obstacles.future_timestamps[row]
                - obstacles.sample_timestamps[row]
            ).to(relative_times.dtype),
        ))
        source_positions = torch.cat((
            obstacles.positions_world[row, :, None],
            obstacles.future_positions_world[row],
        ), dim=1)
        source_valid = torch.cat((
            (obstacles.valid_mask[row] & obstacles.dynamic_mask[row])[:, None],
            obstacles.future_valid_mask[row],
        ), dim=1)
        for time_index, requested in enumerate(relative_times):
            if requested <= source_times[-1]:
                upper = int(torch.searchsorted(source_times, requested).item())
                upper = max(1, min(upper, source_times.numel() - 1))
                lower = upper - 1
                ratio = (
                    (requested - source_times[lower])
                    / (source_times[upper] - source_times[lower]).clamp_min(1e-8)
                )
                result[row, :, time_index] = (
                    source_positions[:, lower] * (1.0 - ratio)
                    + source_positions[:, upper] * ratio
                )
                valid[row, :, time_index] = (
                    source_valid[:, lower] & source_valid[:, upper]
                )
            else:
                extrapolated[row, time_index] = True
                last = source_positions[:, -1]
                previous = source_positions[:, -2]
                delta = (source_times[-1] - source_times[-2]).clamp_min(1e-8)
                velocity = (last - previous) / delta
                result[row, :, time_index] = (
                    last + velocity * (requested - source_times[-1])
                )
                valid[row, :, time_index] = source_valid[:, -1]
    return result, valid, extrapolated


def evaluate_bank(
    trainer,
    raw,
    primitive_ids,
    position,
    rotation,
    observation,
    map_id,
    obstacles,
    duration,
):
    batch, candidates = raw.shape[:2]
    fixed, predicted = derivatives_world(
        trainer, raw, primitive_ids, position, rotation, observation
    )
    base_sampler = trainer.dynamic_dep_loss.safety_loss.trajectory_sampler
    sampler = QuinticTrajectorySampler(
        base_sampler.coefficient_map, duration, base_sampler.eval_points
    ).to(trainer.device)
    trajectories, relative_times = sampler.grouped(fixed, predicted, batch)
    velocities, accelerations = sampled_kinematics(sampler, fixed, predicted)
    velocity_max = torch.linalg.vector_norm(velocities, dim=-1).amax(1).reshape(
        batch, candidates
    )
    acceleration_max = torch.linalg.vector_norm(
        accelerations, dim=-1
    ).amax(1).reshape(batch, candidates)
    # The current/deployed parameterization constrains terminal P/V/A through
    # tanh/raw bounds. It has no sampled intermediate norm Gate. Keep that
    # stricter check as a diagnostic so O1--O4 retain the exact O0 semantics.
    feasible = (raw.abs() <= 1.0 + 1e-6).all(dim=2)
    intermediate_limit_pass = (
        (velocity_max <= float(cfg["vel_max_train"]) + 1e-5)
        & (acceleration_max <= float(cfg["acc_max_train"]) + 1e-5)
    )

    _, static_distance = trainer.dynamic_dep_loss.safety_loss.get_distance_cost(
        trajectories.reshape(batch, -1, 3), map_id.to(trainer.device)
    )
    static_clearance = static_distance.reshape(
        batch, candidates, -1
    ).amin(dim=2)

    active = (
        obstacles.valid_mask & obstacles.dynamic_mask
        & (
            obstacles.observable_mask
            if obstacles.observable_mask is not None else obstacles.valid_mask
        )
    )
    if obstacles.max_obstacles == 0 or not bool(active.any()):
        dynamic_clearance = torch.full_like(
            static_clearance, torch.finfo(static_clearance.dtype).max
        )
        extrapolated = torch.zeros(
            batch, relative_times.numel(), dtype=torch.bool,
            device=trainer.device,
        )
    else:
        future, future_valid, extrapolated = interpolated_future(
            obstacles, relative_times
        )
        future_valid = future_valid & active[:, :, None]
        distance = torch.linalg.vector_norm(
            trajectories[:, :, None] - future[:, None], dim=-1
        )
        config = trainer.dynamic_dep_loss.dynamic_loss_config
        current_age = (
            obstacles.sample_timestamps[:, None]
            - obstacles.track_timestamps
        ).to(trajectories.dtype)
        delta_time = current_age[:, :, None] + relative_times[None, None]
        covariance = torch.linalg.eigvalsh(
            obstacles.position_covariances
        ).amax(-1).clamp_min(0)
        variance = (
            covariance[:, :, None]
            + config.covariance_growth_rate * delta_time.square()
        )
        obstacle_radius = torch.where(
            obstacles.radii > 0,
            obstacles.radii,
            torch.full_like(obstacles.radii, config.default_obstacle_radius),
        )
        safe_radius = (
            config.uav_radius + obstacle_radius[:, :, None]
            + config.covariance_sigma * variance.sqrt()
        )
        clearance = (distance - safe_radius[:, None]).masked_fill(
            ~future_valid[:, None], float("inf")
        )
        dynamic_clearance = clearance.flatten(2).amin(2)
        no_future = ~future_valid.any(dim=(1, 2))
        dynamic_clearance = torch.where(
            no_future[:, None],
            torch.full_like(
                dynamic_clearance, torch.finfo(dynamic_clearance.dtype).max
            ),
            dynamic_clearance,
        )
    dynamic_safe = dynamic_clearance >= -TOLERANCE
    static_safe = static_clearance >= -TOLERANCE
    joint_safe = dynamic_safe & static_safe & feasible
    joint_violation = (
        torch.relu(-dynamic_clearance.clamp_min(-1e3))
        + torch.relu(-static_clearance)
    )
    return {
        "joint_safe": joint_safe,
        "dynamic_safe": dynamic_safe & feasible,
        "static_safe": static_safe & feasible,
        "dynamic_clearance": dynamic_clearance,
        "static_clearance": static_clearance,
        "joint_violation": joint_violation,
        "velocity_max": velocity_max,
        "acceleration_max": acceleration_max,
        "feasible": feasible,
        "intermediate_limit_pass": intermediate_limit_pass,
        "extrapolated_time_fraction": extrapolated.float().mean(1),
    }


def sobol_bank(current_raw, count):
    if count < 15:
        raise ValueError("candidate bank count must be at least 15")
    batch = current_raw.shape[0]
    primitive_ids = torch.arange(count, device=current_raw.device) % 15
    if count == 15:
        return current_raw, primitive_ids
    engine = torch.quasirandom.SobolEngine(9, scramble=False, seed=88031)
    samples = engine.draw(count).to(current_raw.device, current_raw.dtype)
    samples = samples * 2.0 - 1.0
    samples = samples[15:count]
    extra = samples[None].expand(batch, -1, -1)
    return torch.cat((current_raw, extra), dim=1), primitive_ids


def apply_profile(raw, profile):
    value = raw.clone()
    radius = (value[:, :, 2] + 1.0) * float(profile["radius_scale"])
    value[:, :, 2] = radius - 1.0
    value[:, :, 3:6] *= float(profile["velocity_scale"])
    value[:, :, 6:9] *= float(profile["acceleration_scale"])
    return value.clamp(-1.0, 1.0)


def direct_optimize(
    trainer,
    current_raw,
    position,
    rotation,
    observation,
    map_id,
    obstacles,
    iterations,
):
    primitive_ids = torch.arange(15, device=trainer.device)
    raw = current_raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.04)
    best_violation = torch.full(
        (raw.shape[0],), float("inf"), device=trainer.device
    )
    success = torch.zeros(raw.shape[0], dtype=torch.bool, device=trainer.device)
    boundary = torch.zeros_like(success)
    completed = torch.zeros(
        raw.shape[0], dtype=torch.int64, device=trainer.device
    )
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        result = evaluate_bank(
            trainer, raw, primitive_ids, position, rotation, observation,
            map_id, obstacles, float(cfg["sgm_time"]),
        )
        violation = result["joint_violation"]
        weights = torch.softmax(-violation / 0.08, dim=1)
        objective = (weights * violation).sum(1).mean()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            raw.clamp_(-1.0, 1.0)
            minimum = violation.amin(1)
            improved = minimum < best_violation
            best_violation = torch.minimum(best_violation, minimum)
            completed[improved] = iteration + 1
            success |= result["joint_safe"].any(1)
            boundary |= (raw.abs() >= 0.999).any(dim=(1, 2))
    final = evaluate_bank(
        trainer, raw, primitive_ids, position, rotation, observation,
        map_id, obstacles, float(cfg["sgm_time"]),
    )
    success |= final["joint_safe"].any(1)
    best_violation = torch.minimum(
        best_violation, final["joint_violation"].amin(1)
    )
    return {
        "success": success,
        "best_joint_violation": best_violation,
        "iterations": completed,
        "reached_output_boundary": boundary,
        "best_dynamic_clearance": final["dynamic_clearance"].amax(1),
        "best_static_clearance": final["static_clearance"].amax(1),
    }


def record_summary(records, key):
    values = np.asarray([row[key] for row in records], dtype=bool)
    return {
        "success_count": int(values.sum()),
        "failure_count": int((~values).sum()),
        "failure_fraction_full_suite": float((~values).sum() / 2052.0),
        "success_fraction_among_o0_failures": float(values.mean()) if len(values) else 0.0,
    }


def grouped_taxonomy(records, key):
    groups = defaultdict(Counter)
    for row in records:
        groups[str(row[key])][row["taxonomy"]] += 1
    return {
        group: {"total": sum(counts.values()), **dict(counts)}
        for group, counts in sorted(groups.items())
    }


def analyze_suite(trainer, suite_name, max_failure_windows, direct_iterations):
    loader = trainer.validation_suites[suite_name]
    scenarios = scenario_lookup(loader.dataset)
    all_windows = []
    failures = []
    started = time.perf_counter()
    for batch in loader:
        details = trainer.compute_batch("dynamic", batch)
        raw = current_raw_predictions(trainer, batch)
        dynamic_clearance = details[
            "dynamic_diagnostics"
        ].candidate_min_clearance.detach()
        static_clearance = details["candidate_static_clearance"].detach()
        current_joint = (
            (dynamic_clearance >= -TOLERANCE)
            & (static_clearance >= -TOLERANCE)
        )
        for row in range(raw.shape[0]):
            metadata = {
                "sequence_id": batch["sequence_id"][row],
                "frame_index": int(batch["frame_index"][row]),
                "map_id": int(batch["map_id"][row]),
                "scenario": scenarios[batch["sequence_id"][row]],
                "category": batch["sample_category"][row],
                "actor_count": int(
                    details["dynamic_diagnostics"].dynamic_obstacle_count[row]
                ),
                "o0_dynamic_safe_count": int(
                    (dynamic_clearance[row] >= -TOLERANCE).sum()
                ),
                "o0_joint_safe_count": int(current_joint[row].sum()),
                "o0_joint_success": bool(current_joint[row].any()),
                "o0_minimum_joint_violation": float((
                    torch.relu(-dynamic_clearance[row].clamp_min(-1e3))
                    + torch.relu(-static_clearance[row])
                ).amin()),
                "o0_maximum_dynamic_clearance": float(
                    dynamic_clearance[row].amax()
                ),
                "o0_maximum_static_clearance": float(
                    static_clearance[row].amax()
                ),
            }
            all_windows.append(metadata)
            if (
                not metadata["o0_joint_success"]
                and (
                    max_failure_windows <= 0
                    or len(failures) < max_failure_windows
                )
            ):
                failures.append((metadata, raw[row:row + 1].detach(), {
                    "position": batch["position_world"][row:row + 1],
                    "rotation": batch["rotation_world_from_body"][row:row + 1],
                    "observation": batch["observation_9d"][row:row + 1],
                    "map_id": batch["map_id"][row:row + 1],
                    "obstacles": slice_obstacles(
                        batch["dynamic_obstacles"], slice(row, row + 1)
                    ).to(trainer.device),
                }))

    oracle_records = []
    oracle_batch_size = 8
    for chunk_start in range(0, len(failures), oracle_batch_size):
        chunk = failures[chunk_start:chunk_start + oracle_batch_size]
        metadata_rows = [item[0] for item in chunk]
        current_raw = torch.cat([item[1] for item in chunk], dim=0)
        position = torch.cat([item[2]["position"] for item in chunk], dim=0)
        rotation = torch.cat([item[2]["rotation"] for item in chunk], dim=0)
        observation = torch.cat(
            [item[2]["observation"] for item in chunk], dim=0
        )
        map_id = torch.cat([item[2]["map_id"] for item in chunk], dim=0)
        obstacles = concatenate_obstacles(
            [item[2]["obstacles"] for item in chunk]
        )
        print(
            f"[{suite_name} {chunk_start + 1}-"
            f"{chunk_start + len(chunk)}/{len(failures)}]",
            flush=True,
        )
        o1 = direct_optimize(
            trainer, current_raw, position, rotation, observation, map_id,
            obstacles, direct_iterations,
        )
        max_bank, primitive_ids = sobol_bank(current_raw, DENSE_LEVELS[-1])
        dense = evaluate_bank(
            trainer, max_bank, primitive_ids, position, rotation,
            observation, map_id, obstacles, float(cfg["sgm_time"]),
        )
        dense_success = {
            str(level): dense["joint_safe"][:, :level].any(dim=1)
            for level in DENSE_LEVELS
        }
        temporal_success = {}
        temporal_best = torch.full(
            (len(chunk),), float("inf"), device=trainer.device
        )
        temporal_extrapolated = {}
        for profile in TEMPORAL_PROFILES:
            profile_raw = apply_profile(max_bank[:, :256], profile)
            result = evaluate_bank(
                trainer, profile_raw, primitive_ids[:256], position, rotation,
                observation, map_id, obstacles,
                float(cfg["sgm_time"]) * float(profile["duration_scale"]),
            )
            temporal_success[profile["name"]] = result[
                "joint_safe"
            ].any(dim=1)
            temporal_best = torch.minimum(
                temporal_best, result["joint_violation"].amin(dim=1)
            )
            temporal_extrapolated[profile["name"]] = result[
                "extrapolated_time_fraction"
            ]
        extended_success = {}
        for profile in EXTENDED_PROFILES:
            profile_raw = apply_profile(max_bank[:, :256], profile)
            result = evaluate_bank(
                trainer, profile_raw, primitive_ids[:256], position, rotation,
                observation, map_id, obstacles,
                float(cfg["sgm_time"]) * float(profile["duration_scale"]),
            )
            extended_success[profile["name"]] = result[
                "joint_safe"
            ].any(dim=1)

        for row, metadata in enumerate(metadata_rows):
            o1_success = bool(o1["success"][row])
            o2_success = bool(
                dense_success[str(DENSE_LEVELS[-1])][row]
            )
            o3_success = any(
                bool(value[row]) for value in temporal_success.values()
            )
            o4_success = any(
                bool(value[row]) for value in extended_success.values()
            )
            if o1_success:
                taxonomy = "optimization_limited"
            elif o2_success:
                taxonomy = "count_limited"
            elif o3_success:
                taxonomy = "temporal_limited"
            elif o4_success:
                taxonomy = "horizon_limited"
            else:
                # O5 is conservative: without a distinct richer trajectory
                # parameterization oracle, unresolved cases remain
                # parameterization-limited rather than being falsely declared
                # physically impossible.
                taxonomy = "parameterization_limited"
            record = {
                **metadata,
                "o1_success": o1_success,
                "o1_best_joint_violation": float(
                    o1["best_joint_violation"][row]
                ),
                "o1_optimizer_iterations": int(o1["iterations"][row]),
                "o1_reached_output_boundary": bool(
                    o1["reached_output_boundary"][row]
                ),
                "o1_best_dynamic_clearance": float(
                    o1["best_dynamic_clearance"][row]
                ),
                "o1_best_static_clearance": float(
                    o1["best_static_clearance"][row]
                ),
                "o2_dense_success": {
                    key: bool(value[row])
                    for key, value in dense_success.items()
                },
                "o2_best_joint_violation": float(
                    dense["joint_violation"][row].amin()
                ),
                "o2_feasible_fraction": float(
                    dense["feasible"][row].float().mean()
                ),
                "o2_intermediate_limit_pass_fraction": float(
                    dense["intermediate_limit_pass"][row].float().mean()
                ),
                "o3_temporal_success": {
                    key: bool(value[row])
                    for key, value in temporal_success.items()
                },
                "o3_best_joint_violation": float(temporal_best[row]),
                "o3_extrapolated_time_fraction": {
                    key: float(value[row])
                    for key, value in temporal_extrapolated.items()
                },
                "o4_extended_success": {
                    key: bool(value[row])
                    for key, value in extended_success.items()
                },
                "taxonomy": taxonomy,
            }
            oracle_records.append(record)

    unaudited_failures = len(all_windows) - sum(
        row["o0_joint_success"] for row in all_windows
    ) - len(oracle_records)
    total = len(all_windows)
    o0_failures = sum(not row["o0_joint_success"] for row in all_windows)
    def full_failure(success_key):
        recovered = sum(bool(row[success_key]) for row in oracle_records)
        return (o0_failures - recovered) / total

    o2_levels = {
        str(level): (
            o0_failures - sum(
                row["o2_dense_success"][str(level)] for row in oracle_records
            )
        ) / total
        for level in DENSE_LEVELS
    }
    o3_recovered = sum(
        any(row["o3_temporal_success"].values()) for row in oracle_records
    )
    o4_recovered = sum(
        any(row["o4_extended_success"].values()) for row in oracle_records
    )
    taxonomy = Counter(row["taxonomy"] for row in oracle_records)
    result = {
        "suite": suite_name,
        "window_count": total,
        "capacity_failure_windows_audited": len(oracle_records),
        "capacity_failure_windows_unaudited": unaudited_failures,
        "formal_complete": unaudited_failures == 0,
        "o0_current_15": {
            "joint_coverage_failure_count": o0_failures,
            "joint_coverage_failure_fraction": o0_failures / total,
        },
        "o1_direct_optimization": {
            "joint_coverage_failure_fraction": full_failure("o1_success"),
            "recovered_count": sum(
                row["o1_success"] for row in oracle_records
            ),
            "iterations": direct_iterations,
        },
        "o2_dense_same_bound": {
            "joint_coverage_failure_fraction_by_count": o2_levels,
            "fixed_duration_seconds": float(cfg["sgm_time"]),
            "sobol_seed": 88031,
        },
        "o3_temporal_bank": {
            "joint_coverage_failure_fraction": (
                o0_failures - o3_recovered
            ) / total,
            "recovered_count": o3_recovered,
            "profiles": list(TEMPORAL_PROFILES),
            "recorded_future_extrapolation": (
                "linear extrapolation from the last two recorded GT samples "
                "is used only after the formal 1.667 s supervision horizon"
            ),
        },
        "o4_extended_horizon_diagnostic": {
            "joint_coverage_failure_fraction": (
                o0_failures - o4_recovered
            ) / total,
            "recovered_count": o4_recovered,
            "profiles": list(EXTENDED_PROFILES),
            "gate_eligible": False,
        },
        "taxonomy": {
            "counts": dict(taxonomy),
            "fractions_of_o0_failures": {
                key: value / max(o0_failures, 1)
                for key, value in taxonomy.items()
            },
            "by_scenario": grouped_taxonomy(oracle_records, "scenario"),
            "by_map": grouped_taxonomy(oracle_records, "map_id"),
            "by_actor_count": grouped_taxonomy(
                oracle_records, "actor_count"
            ),
            "intrinsically_infeasible_fraction": None,
            "intrinsically_infeasible_fraction_bounds": [
                0.0,
                taxonomy.get("parameterization_limited", 0) / total,
            ],
            "intrinsic_classification_status": (
                "not identifiable with the current O0-O4 parameterization; "
                "the unresolved set is an upper bound, not a claim that every "
                "window is physically infeasible"
            ),
        },
        "all_windows": all_windows,
        "failure_records": oracle_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


def write_taxonomy(report):
    estimated = report["suites"]["valid_estimated"]
    counts = estimated["taxonomy"]["counts"]
    lines = [
        "# Phase 8J-R capacity failure taxonomy",
        "",
        "The taxonomy is based on the complete frozen `valid_estimated` suite.",
        "Unresolved O1–O4 windows are conservatively labelled "
        "`parameterization_limited`; they are not claimed to be intrinsically "
        "infeasible without a richer-parameterization oracle.",
        "",
        "## Counts",
        "",
    ]
    for name in (
        "optimization_limited", "count_limited", "temporal_limited",
        "horizon_limited", "parameterization_limited",
        "intrinsically_infeasible",
    ):
        lines.append(f"- {name}: {counts.get(name, 0)}")
    lines.extend([
        "",
        "## Oracle failures",
        "",
        f"- O0 current-15: "
        f"{estimated['o0_current_15']['joint_coverage_failure_fraction']:.6f}",
        f"- O1 direct optimization: "
        f"{estimated['o1_direct_optimization']['joint_coverage_failure_fraction']:.6f}",
        f"- O2 dense-512: "
        f"{estimated['o2_dense_same_bound']['joint_coverage_failure_fraction_by_count']['512']:.6f}",
        f"- O3 temporal bank: "
        f"{estimated['o3_temporal_bank']['joint_coverage_failure_fraction']:.6f}",
        f"- O4 extended diagnostic: "
        f"{estimated['o4_extended_horizon_diagnostic']['joint_coverage_failure_fraction']:.6f}",
        "",
        "O4 is diagnostic only and is excluded from the capacity continuation Gate.",
    ])
    path = ROOT / "reports/phase8jr_capacity_failure_taxonomy.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", nargs="+", default=["valid_estimated", "valid_gt"],
        choices=("valid_estimated", "valid_gt"),
    )
    parser.add_argument(
        "--max-failure-windows", type=int, default=0,
        help="nonzero is smoke-only and intentionally cannot emit a formal report",
    )
    parser.add_argument("--direct-iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/phase8jr_capacity_oracle.json",
    )
    args = parser.parse_args()
    entry = json.loads(
        (ROOT / "reports/phase8jr_entry_gate.json").read_text()
    )
    if entry.get("status") != "PASS":
        raise RuntimeError("Phase 8J-R entry Gate is not PASS")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal Phase 8J-R capacity audit requires host CUDA; do not "
            "interpret a sandbox CUDA result"
        )
    if args.direct_iterations < 1 or args.direct_iterations > 100:
        raise ValueError("direct iterations must be in 1..100")
    trainer, frozen = make_trainer()
    suites = {
        name: analyze_suite(
            trainer, name, args.max_failure_windows, args.direct_iterations
        )
        for name in args.suites
    }
    report = {
        "status": "PASS",
        "phase": "8J-R-capacity-audit",
        "capacity_audit_complete": (
            args.max_failure_windows == 0
            and set(suites) == {"valid_estimated", "valid_gt"}
        ),
        "frozen": frozen,
        "dense_levels": list(DENSE_LEVELS),
        "temporal_profiles": list(TEMPORAL_PROFILES),
        "extended_profiles": list(EXTENDED_PROFILES),
        "suites": suites,
        "production_test_used": False,
        "network_weights_modified": False,
        "score_stage_executed": False,
    }
    if args.max_failure_windows:
        output = args.output.with_name(args.output.stem + "_smoke.json")
    else:
        output = args.output
    atomic_json(output, report)
    if args.max_failure_windows == 0 and set(suites) == {
        "valid_estimated", "valid_gt"
    }:
        write_taxonomy(report)
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "capacity_audit_complete": report["capacity_audit_complete"],
    }, indent=2))


if __name__ == "__main__":
    main()
