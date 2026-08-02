#!/usr/bin/env python3
"""Phase 8J-P offline O5/O6 richer-parameterization capacity audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from loss.dynamic_types import DynamicObstacleBatch
from policy.state_transform import rotate_body2world, state_body2world
from tools.analyze_candidate_capacity_oracle import (
    concatenate_obstacles,
    current_raw_predictions,
    derivatives_world,
    interpolated_future,
    make_trainer,
    slice_obstacles,
)
from tools.run_phase8i_failure_decomposition import scenario_lookup
from tools.phase8jp_trajectory_models import (
    integrate_piecewise_constant_jerk,
    sample_piecewise_quintic,
)


HORIZON = float(cfg["sgm_time"])
VELOCITY_LIMIT = float(cfg["vel_max_train"])
ACCELERATION_LIMIT = float(cfg["acc_max_train"])
# The repository has no controller-authoritative jerk limit. This audit-only
# value is frozen before the oracle run and must be confirmed in Phase 8J-Q.
JERK_LIMIT = 30.0
TIME_SPLITS = (0.30, 0.40, 0.50, 0.60, 0.70)
O5_ITERATIONS = 60
O6_ITERATIONS = 80
ORACLE_SEED = 89117


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def oracle_config():
    value = {
        "version": "phase8jp_o5_o6_v2",
        "horizon_seconds": HORIZON,
        "samples": 30,
        "velocity_limit_mps": VELOCITY_LIMIT,
        "acceleration_limit_mps2": ACCELERATION_LIMIT,
        "jerk_limit_mps3": JERK_LIMIT,
        "jerk_limit_status": (
            "audit-only; no controller-authoritative jerk limit exists in "
            "the current repository"
        ),
        "waypoint_body_bounds_m": {
            "forward": [0.0, 5.0],
            "lateral": [-3.0, 3.0],
            "vertical": [-2.0, 2.0],
        },
        "time_splits": list(TIME_SPLITS),
        "o5_iterations": O5_ITERATIONS,
        "o6_intervals": 8,
        "o6_starts": 8,
        "o6_iterations": O6_ITERATIONS,
        "o5d_action_families": [
            "brake_then_go",
            "hover_then_go",
            "lateral_escape_then_rejoin",
            "vertical_escape_then_rejoin",
            "yield_then_pass",
            "pass_then_recover",
        ],
        "seed": ORACLE_SEED,
        "future_gt_role": "offline safety evaluator only",
        "instance_mask_used": False,
        "production_test_used": False,
    }
    value["config_sha256"] = sha256_bytes(
        json.dumps(value, sort_keys=True).encode()
    )
    return value


def evaluate_trajectory_samples(
    trainer, samples, map_id, obstacles: DynamicObstacleBatch
):
    trajectories = samples.position
    batch, candidates, points = trajectories.shape[:3]
    _, static_distance = trainer.dynamic_dep_loss.safety_loss.get_distance_cost(
        trajectories.reshape(batch, -1, 3), map_id.to(trainer.device)
    )
    static_clearance = static_distance.reshape(
        batch, candidates, points
    ).amin(2)
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
        dynamic_cost = torch.zeros_like(static_clearance)
    else:
        future, valid, _ = interpolated_future(obstacles, samples.times)
        valid = valid & active[:, :, None]
        distance = torch.linalg.vector_norm(
            trajectories[:, :, None] - future[:, None], dim=-1
        )
        config = trainer.dynamic_dep_loss.dynamic_loss_config
        age = (
            obstacles.sample_timestamps[:, None]
            - obstacles.track_timestamps
        ).to(trajectories.dtype)
        delta = age[:, :, None] + samples.times[None, None]
        covariance = torch.linalg.eigvalsh(
            obstacles.position_covariances
        ).amax(-1).clamp_min(0)
        variance = (
            covariance[:, :, None]
            + config.covariance_growth_rate * delta.square()
        )
        radius = torch.where(
            obstacles.radii > 0,
            obstacles.radii,
            torch.full_like(obstacles.radii, config.default_obstacle_radius),
        )
        safe_radius = (
            config.uav_radius + radius[:, :, None]
            + config.covariance_sigma * variance.sqrt()
        )
        clearance = (distance - safe_radius[:, None]).masked_fill(
            ~valid[:, None], float("inf")
        )
        dynamic_clearance = clearance.flatten(2).amin(2)
        no_future = ~valid.any(dim=(1, 2))
        dynamic_clearance = torch.where(
            no_future[:, None],
            torch.full_like(
                dynamic_clearance,
                torch.finfo(dynamic_clearance.dtype).max,
            ),
            dynamic_clearance,
        )
        dynamic_cost = torch.relu(-dynamic_clearance)
    velocity_max = torch.linalg.vector_norm(
        samples.velocity, dim=-1
    ).amax(2)
    acceleration_max = torch.linalg.vector_norm(
        samples.acceleration, dim=-1
    ).amax(2)
    jerk_max = torch.linalg.vector_norm(samples.jerk, dim=-1).amax(2)
    finite = (
        torch.isfinite(trajectories).all(dim=(2, 3))
        & torch.isfinite(samples.velocity).all(dim=(2, 3))
        & torch.isfinite(samples.acceleration).all(dim=(2, 3))
        & torch.isfinite(samples.jerk).all(dim=(2, 3))
    )
    physical = (
        (velocity_max <= VELOCITY_LIMIT + 1e-5)
        & (acceleration_max <= ACCELERATION_LIMIT + 1e-5)
        & (jerk_max <= JERK_LIMIT + 1e-5)
        & finite
    )
    safe = (
        (dynamic_clearance >= -1e-6)
        & (static_clearance >= -1e-6)
        & physical
    )
    violation = (
        torch.relu(-dynamic_clearance.clamp_min(-1e3))
        + torch.relu(-static_clearance)
        + torch.relu(velocity_max - VELOCITY_LIMIT) / VELOCITY_LIMIT
        + torch.relu(acceleration_max - ACCELERATION_LIMIT)
          / ACCELERATION_LIMIT
        + torch.relu(jerk_max - JERK_LIMIT) / JERK_LIMIT
        + (~finite).to(trajectories.dtype) * 100.0
    )
    return {
        "safe": safe,
        "violation": violation,
        "dynamic_clearance": dynamic_clearance,
        "dynamic_cost": dynamic_cost,
        "static_clearance": static_clearance,
        "velocity_max": velocity_max,
        "acceleration_max": acceleration_max,
        "jerk_max": jerk_max,
        "physical": physical,
    }


def initial_waypoints(batch, starts, device, dtype):
    fixed = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, -0.8, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, -0.8],
        [-0.6, -0.8, 0.0],
        [-0.6, 0.8, 0.0],
    ], device=device, dtype=dtype)
    if starts > len(fixed):
        engine = torch.quasirandom.SobolEngine(
            3, scramble=False, seed=ORACLE_SEED
        )
        sobol = engine.draw(starts - len(fixed)).to(device, dtype) * 2 - 1
        values = torch.cat((fixed, sobol), dim=0)
    else:
        values = fixed[:starts]
    return values[None].expand(batch, -1, -1).clone()


def waypoint_world(start, rotation, waypoint_raw):
    offset = torch.stack((
        (waypoint_raw[:, :, 0] + 1.0) * 2.5,
        waypoint_raw[:, :, 1] * 3.0,
        waypoint_raw[:, :, 2] * 2.0,
    ), dim=-1)
    return (
        start[:, None, :, 0]
        + rotate_body2world(
            rotation[:, None].expand(-1, offset.shape[1], -1, -1),
            offset,
        )
    )


def projected_update(parameters, objective, learning_rate):
    gradients = torch.autograd.grad(
        objective, parameters, allow_unused=False
    )
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            scale = gradient.norm().clamp_min(1.0)
            parameter.add_(gradient, alpha=-learning_rate / float(scale))
            parameter.clamp_(-1.0, 1.0)


def update_selected(storage, name, candidates, indices, improved):
    rows = torch.arange(indices.shape[0], device=indices.device)
    selected = candidates[rows, indices].detach()
    if name not in storage:
        storage[name] = selected.clone()
    else:
        mask = improved.reshape(
            improved.shape + (1,) * (selected.ndim - improved.ndim)
        )
        storage[name] = torch.where(mask, selected, storage[name])


def optimize_piecewise(
    trainer, current_raw, position, rotation, observation, map_id, obstacles,
    mode, split=0.5, iterations=O5_ITERATIONS,
):
    batch = current_raw.shape[0]
    starts = 6 if mode == "actions" else 15
    end_raw = current_raw[:, :starts].detach().clone().requires_grad_(True)
    waypoint_raw = initial_waypoints(
        batch, starts, trainer.device, current_raw.dtype
    ).requires_grad_(True)
    velocity_raw = None
    if mode in {"velocity", "time", "actions"}:
        velocity_raw = torch.zeros(
            batch, starts, 3, device=trainer.device,
            dtype=current_raw.dtype, requires_grad=True,
        )
        if mode == "actions":
            with torch.no_grad():
                # brake, hover, lateral, vertical, yield, pass/recover.
                waypoint_raw[:] = torch.tensor([
                    [-0.6, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-0.2, 0.5, 0.0],
                    [-0.2, 0.0, 0.5],
                    [-0.7, -0.4, 0.0],
                    [0.2, 0.4, 0.0],
                ], device=trainer.device, dtype=current_raw.dtype)
                velocity_raw[:] = 0.0
                velocity_raw[:, 0, 0] = 0.10
                velocity_raw[:, 2, 0] = 0.35
                velocity_raw[:, 2, 1] = 0.25
                velocity_raw[:, 3, 0] = 0.25
                velocity_raw[:, 3, 2] = 0.20
                velocity_raw[:, 4, 1] = -0.20
                velocity_raw[:, 5, 0] = 0.65
    primitive_ids = torch.arange(starts, device=trainer.device)
    success = torch.zeros(batch, dtype=torch.bool, device=trainer.device)
    best_violation = torch.full(
        (batch,), float("inf"), device=trainer.device
    )
    best_selected = {}
    best_solution = {}
    for _ in range(iterations):
        fixed, predicted = derivatives_world(
            trainer, end_raw, primitive_ids, position, rotation, observation
        )
        start = fixed.reshape(batch, starts, 3, 3)
        end = predicted.reshape(batch, starts, 3, 3)
        midpoint_position = waypoint_world(start[:, 0], rotation, waypoint_raw)
        if velocity_raw is None:
            midpoint_velocity = 0.5 * (
                start[:, :, :, 1] + end[:, :, :, 1]
            )
        else:
            midpoint_velocity = rotate_body2world(
                rotation[:, None].expand(-1, starts, -1, -1),
                velocity_raw * VELOCITY_LIMIT,
            )
        midpoint_acceleration = 0.5 * (
            start[:, :, :, 2] + end[:, :, :, 2]
        )
        midpoint = torch.stack((
            midpoint_position, midpoint_velocity, midpoint_acceleration
        ), dim=3)
        samples = sample_piecewise_quintic(
            start, midpoint, end, HORIZON*split, HORIZON*(1.0-split), 30
        )
        metrics = evaluate_trajectory_samples(
            trainer, samples, map_id, obstacles
        )
        weights = torch.softmax(-metrics["violation"] / 0.08, dim=1)
        objective = (weights * metrics["violation"]).sum(1).mean()
        current_min, indices = metrics["violation"].min(1)
        improved = current_min < best_violation
        for key in (
            "dynamic_clearance", "static_clearance", "velocity_max",
            "acceleration_max", "jerk_max", "physical",
        ):
            selected = metrics[key].gather(
                1, indices[:, None]
            ).squeeze(1).detach()
            if key not in best_selected:
                best_selected[key] = selected.clone()
            else:
                best_selected[key] = torch.where(
                    improved, selected, best_selected[key]
                )
        waypoint_body = torch.stack((
            (waypoint_raw[:, :, 0] + 1.0) * 2.5,
            waypoint_raw[:, :, 1] * 3.0,
            waypoint_raw[:, :, 2] * 2.0,
        ), dim=-1)
        for name, candidates in (
            ("waypoint_body_m", waypoint_body),
            ("midpoint_state_world", midpoint),
            ("terminal_state_world", end),
            ("trajectory_position_world", samples.position),
            ("trajectory_velocity_world", samples.velocity),
            ("trajectory_acceleration_world", samples.acceleration),
            ("trajectory_jerk_world", samples.jerk),
        ):
            update_selected(
                best_solution, name, candidates, indices, improved
            )
        raw_parameters = [end_raw, waypoint_raw]
        if velocity_raw is not None:
            raw_parameters.append(velocity_raw)
        boundary = torch.stack(
            [value.abs().flatten(2).amax(2) for value in raw_parameters],
            dim=-1,
        ).amax(-1) >= 0.999
        update_selected(
            best_solution, "parameter_boundary_touched",
            boundary, indices, improved,
        )
        best_violation = torch.minimum(best_violation, current_min.detach())
        success |= metrics["safe"].any(1).detach()
        parameters = [end_raw, waypoint_raw]
        if velocity_raw is not None:
            parameters.append(velocity_raw)
        projected_update(parameters, objective, 0.035)
    # One exact terminal evaluation after the final projected update.
    fixed, predicted = derivatives_world(
        trainer, end_raw, primitive_ids, position, rotation, observation
    )
    start = fixed.reshape(batch, starts, 3, 3)
    end = predicted.reshape(batch, starts, 3, 3)
    midpoint_position = waypoint_world(start[:, 0], rotation, waypoint_raw)
    midpoint_velocity = (
        0.5 * (start[:, :, :, 1] + end[:, :, :, 1])
        if velocity_raw is None else rotate_body2world(
            rotation[:, None].expand(-1, starts, -1, -1),
            velocity_raw * VELOCITY_LIMIT,
        )
    )
    midpoint = torch.stack((
        midpoint_position, midpoint_velocity,
        0.5 * (start[:, :, :, 2] + end[:, :, :, 2]),
    ), dim=3)
    samples = sample_piecewise_quintic(
        start, midpoint, end, HORIZON*split, HORIZON*(1.0-split), 30
    )
    metrics = evaluate_trajectory_samples(trainer, samples, map_id, obstacles)
    best_index = metrics["violation"].argmin(1)
    terminal_violation = metrics["violation"].amin(1).detach()
    improved = terminal_violation < best_violation
    for key in best_selected:
        selected = metrics[key].gather(
            1, best_index[:, None]
        ).squeeze(1).detach()
        best_selected[key] = torch.where(
            improved, selected, best_selected[key]
        )
    waypoint_body = torch.stack((
        (waypoint_raw[:, :, 0] + 1.0) * 2.5,
        waypoint_raw[:, :, 1] * 3.0,
        waypoint_raw[:, :, 2] * 2.0,
    ), dim=-1)
    for name, candidates in (
        ("waypoint_body_m", waypoint_body),
        ("midpoint_state_world", midpoint),
        ("terminal_state_world", end),
        ("trajectory_position_world", samples.position),
        ("trajectory_velocity_world", samples.velocity),
        ("trajectory_acceleration_world", samples.acceleration),
        ("trajectory_jerk_world", samples.jerk),
    ):
        update_selected(best_solution, name, candidates, best_index, improved)
    raw_parameters = [end_raw, waypoint_raw]
    if velocity_raw is not None:
        raw_parameters.append(velocity_raw)
    boundary = torch.stack(
        [value.abs().flatten(2).amax(2) for value in raw_parameters],
        dim=-1,
    ).amax(-1) >= 0.999
    update_selected(
        best_solution, "parameter_boundary_touched",
        boundary, best_index, improved,
    )
    success |= metrics["safe"].any(1).detach()
    best_violation = torch.minimum(best_violation, terminal_violation)

    def select(name):
        return best_selected[name]
    return {
        "success": success.detach(),
        "minimum_violation": best_violation,
        "dynamic_clearance": select("dynamic_clearance"),
        "static_clearance": select("static_clearance"),
        "velocity_max": select("velocity_max"),
        "acceleration_max": select("acceleration_max"),
        "jerk_max": select("jerk_max"),
        "physical": select("physical"),
        "split": torch.full(
            (batch,), split, device=trainer.device, dtype=current_raw.dtype
        ),
        **best_solution,
        "optimizer_converged": torch.zeros(
            batch, dtype=torch.bool, device=trainer.device
        ),
        "optimizer_failed": torch.zeros(
            batch, dtype=torch.bool, device=trainer.device
        ),
        "optimizer_termination": ["max_iterations_finite"] * batch,
    }


def optimize_kinodynamic(
    trainer, position, rotation, observation, map_id, obstacles,
    iterations=O6_ITERATIONS,
):
    batch, starts, intervals = position.shape[0], 8, 8
    _, velocity, acceleration = state_body2world(
        position.to(trainer.device), rotation.to(trainer.device),
        observation[:, 6:9].to(trainer.device),
        observation[:, 0:3].to(trainer.device),
        observation[:, 3:6].to(trainer.device),
    )
    start = torch.stack((
        position.to(trainer.device), velocity, acceleration
    ), dim=2)[:, None].expand(-1, starts, -1, -1).clone()
    engine = torch.quasirandom.SobolEngine(
        intervals * 3, scramble=False, seed=ORACLE_SEED
    )
    seeds = (
        engine.draw(starts).to(trainer.device, position.dtype).reshape(
            starts, intervals, 3
        ) * 0.4 - 0.2
    )
    seeds[0] = 0.0
    seeds[1, :intervals // 2, 0] = -0.2
    seeds[1, intervals // 2:, 0] = 0.2
    jerk_raw = seeds[None].expand(batch, -1, -1, -1).clone()
    jerk_raw.requires_grad_(True)
    success = torch.zeros(batch, dtype=torch.bool, device=trainer.device)
    best_violation = torch.full(
        (batch,), float("inf"), device=trainer.device
    )
    best_selected = {}
    best_solution = {}
    for _ in range(iterations):
        samples = integrate_piecewise_constant_jerk(
            start, jerk_raw * JERK_LIMIT, HORIZON, samples_per_interval=4
        )
        metrics = evaluate_trajectory_samples(
            trainer, samples, map_id, obstacles
        )
        weights = torch.softmax(-metrics["violation"] / 0.08, dim=1)
        objective = (weights * metrics["violation"]).sum(1).mean()
        current_min, indices = metrics["violation"].min(1)
        improved = current_min < best_violation
        for key in (
            "dynamic_clearance", "static_clearance", "velocity_max",
            "acceleration_max", "jerk_max", "physical",
        ):
            selected = metrics[key].gather(
                1, indices[:, None]
            ).squeeze(1).detach()
            if key not in best_selected:
                best_selected[key] = selected.clone()
            else:
                best_selected[key] = torch.where(
                    improved, selected, best_selected[key]
                )
        for name, candidates in (
            ("jerk_control_mps3", jerk_raw * JERK_LIMIT),
            ("trajectory_position_world", samples.position),
            ("trajectory_velocity_world", samples.velocity),
            ("trajectory_acceleration_world", samples.acceleration),
            ("trajectory_jerk_world", samples.jerk),
        ):
            update_selected(
                best_solution, name, candidates, indices, improved
            )
        update_selected(
            best_solution, "control_boundary_touched",
            jerk_raw.abs().flatten(2).amax(2) >= 0.999,
            indices, improved,
        )
        success |= metrics["safe"].any(1).detach()
        best_violation = torch.minimum(best_violation, current_min.detach())
        projected_update([jerk_raw], objective, 0.025)
    samples = integrate_piecewise_constant_jerk(
        start, jerk_raw * JERK_LIMIT, HORIZON, samples_per_interval=4
    )
    final_metrics = evaluate_trajectory_samples(
        trainer, samples, map_id, obstacles
    )
    index = final_metrics["violation"].argmin(1)
    terminal_violation = final_metrics["violation"].amin(1).detach()
    improved = terminal_violation < best_violation
    for key in best_selected:
        selected = final_metrics[key].gather(
            1, index[:, None]
        ).squeeze(1).detach()
        best_selected[key] = torch.where(
            improved, selected, best_selected[key]
        )
    for name, candidates in (
        ("jerk_control_mps3", jerk_raw * JERK_LIMIT),
        ("trajectory_position_world", samples.position),
        ("trajectory_velocity_world", samples.velocity),
        ("trajectory_acceleration_world", samples.acceleration),
        ("trajectory_jerk_world", samples.jerk),
    ):
        update_selected(best_solution, name, candidates, index, improved)
    update_selected(
        best_solution, "control_boundary_touched",
        jerk_raw.abs().flatten(2).amax(2) >= 0.999,
        index, improved,
    )
    success |= final_metrics["safe"].any(1).detach()
    best_violation = torch.minimum(best_violation, terminal_violation)

    def select(name):
        return best_selected[name]
    return {
        "success": success.detach(),
        "minimum_violation": best_violation,
        "dynamic_clearance": select("dynamic_clearance"),
        "static_clearance": select("static_clearance"),
        "velocity_max": select("velocity_max"),
        "acceleration_max": select("acceleration_max"),
        "jerk_max": select("jerk_max"),
        "physical": select("physical"),
        **best_solution,
        "optimizer_converged": torch.zeros(
            batch, dtype=torch.bool, device=trainer.device
        ),
        "optimizer_failed": torch.zeros(
            batch, dtype=torch.bool, device=trainer.device
        ),
        "optimizer_termination": ["max_iterations_finite"] * batch,
    }


def gather_unresolved(trainer, suite_name, keys, limit=0):
    loader = trainer.validation_suites[suite_name]
    scenarios = scenario_lookup(loader.dataset)
    items = []
    for batch in loader:
        batch_keys = [
            (batch["sequence_id"][row], int(batch["frame_index"][row]))
            for row in range(len(batch["sequence_id"]))
        ]
        selected = [row for row, key in enumerate(batch_keys) if key in keys]
        if not selected:
            continue
        raw = current_raw_predictions(trainer, batch)
        for row in selected:
            key = batch_keys[row]
            items.append({
                "key": key,
                "suite": suite_name,
                "sequence_id": key[0],
                "frame_index": key[1],
                "map_id_value": int(batch["map_id"][row]),
                "scenario": scenarios[key[0]],
                "category": batch["sample_category"][row],
                "current_raw": raw[row:row+1].detach(),
                "position": batch["position_world"][row:row+1],
                "rotation": batch["rotation_world_from_body"][row:row+1],
                "observation": batch["observation_9d"][row:row+1],
                "map_id": batch["map_id"][row:row+1],
                "obstacles": slice_obstacles(
                    batch["dynamic_obstacles"], slice(row, row+1)
                ).to(trainer.device),
            })
            if limit and len(items) >= limit:
                return items
    return items


def tensor_result(result, row):
    payload = {}
    for key, value in result.items():
        if torch.is_tensor(value):
            item = value[row].detach().cpu()
            if item.numel() == 1:
                payload[key] = (
                    bool(item) if item.dtype == torch.bool else float(item)
                )
            else:
                payload[key] = item.tolist()
        elif isinstance(value, list):
            payload[key] = value[row]
        else:
            payload[key] = value
    return payload


def classify_outcome(a, b, c, d, k, optimizer_failure=False):
    if a["success"]:
        return "midpoint_position_limited"
    if b["success"]:
        return "midpoint_velocity_limited"
    if c["success"]:
        return "time_allocation_limited"
    if d["success"]:
        return "brake_yield_limited"
    if k["success"]:
        return "piecewise_quintic_limited"
    if optimizer_failure:
        return "optimizer_failure"
    return "kinodynamic_unresolved"


def run_chunk(trainer, chunk):
    current_raw = torch.cat([item["current_raw"] for item in chunk])
    position = torch.cat(
        [item["position"] for item in chunk]
    ).to(trainer.device)
    rotation = torch.cat(
        [item["rotation"] for item in chunk]
    ).to(trainer.device)
    observation = torch.cat(
        [item["observation"] for item in chunk]
    ).to(trainer.device)
    map_id = torch.cat(
        [item["map_id"] for item in chunk]
    ).to(trainer.device)
    obstacles = concatenate_obstacles([item["obstacles"] for item in chunk])

    timings = {}

    def timed(name, function):
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        return result

    o5a = timed("o5a", lambda: optimize_piecewise(
        trainer, current_raw, position, rotation, observation, map_id,
        obstacles, "position", 0.5,
    ))
    o5b = timed("o5b", lambda: optimize_piecewise(
        trainer, current_raw, position, rotation, observation, map_id,
        obstacles, "velocity", 0.5,
    ))
    o5c_parts = timed("o5c", lambda: [
        optimize_piecewise(
            trainer, current_raw, position, rotation, observation, map_id,
            obstacles, "time", split,
        )
        for split in TIME_SPLITS
    ])
    o5d = timed("o5d", lambda: optimize_piecewise(
        trainer, current_raw, position, rotation, observation, map_id,
        obstacles, "actions", 0.5,
    ))
    o6 = timed("o6", lambda: optimize_kinodynamic(
        trainer, position, rotation, observation, map_id, obstacles
    ))
    rows = []
    for row, item in enumerate(chunk):
        c_parts = [tensor_result(part, row) for part in o5c_parts]
        c_best = min(c_parts, key=lambda value: value["minimum_violation"])
        c_best["success"] = any(value["success"] for value in c_parts)
        c_best["tested_splits"] = list(TIME_SPLITS)
        a = tensor_result(o5a, row)
        b = tensor_result(o5b, row)
        d = tensor_result(o5d, row)
        k = tensor_result(o6, row)
        optimizer_failure = any(
            value["optimizer_failed"] for value in (a, b, c_best, d, k)
        )
        taxonomy = classify_outcome(
            a, b, c_best, d, k, optimizer_failure
        )
        rows.append({
            "completion": True,
            "suite": item["suite"],
            "sequence_id": item["sequence_id"],
            "frame_index": item["frame_index"],
            "map_id": item["map_id_value"],
            "scenario": item["scenario"],
            "category": item["category"],
            "actor_count": int(
                (
                    item["obstacles"].valid_mask
                    & item["obstacles"].dynamic_mask
                ).sum().item()
            ),
            "o5a": a,
            "o5b": b,
            "o5c": c_best,
            "o5d": d,
            "o6": k,
            "o5_union_success": any((
                a["success"], b["success"], c_best["success"], d["success"]
            )),
            "taxonomy": taxonomy,
            "optimizer_failure": optimizer_failure,
            "offline_latency_seconds": {
                key: value / len(chunk) for key, value in timings.items()
            },
        })
    return rows


def report_scheme(records, key, baseline_unresolved, full_windows=2052):
    recovered = sum(row[key]["success"] for row in records)
    physical = sum(
        row[key]["physical"] for row in records if row[key]["success"]
    )
    return {
        "status": "PASS",
        "unresolved_window_count": len(records),
        "recovered_count": recovered,
        "joint_coverage_failure_fraction": (
            baseline_unresolved - recovered
        ) / full_windows,
        "physical_feasibility_rate_of_recovered": (
            physical / recovered if recovered else 0.0
        ),
        "optimizer_failure_count": sum(
            row["optimizer_failure"] for row in records
        ),
        "records": [
            {
                "sequence_id": row["sequence_id"],
                "frame_index": row["frame_index"],
                "map_id": row["map_id"],
                "scenario": row["scenario"],
                "category": row["category"],
                "actor_count": row["actor_count"],
                **row[key],
            }
            for row in records
        ],
        "production_test_used": False,
        "network_weights_modified": False,
        "offline_latency_seconds_total": sum(
            row["offline_latency_seconds"][key] for row in records
        ),
        "offline_latency_seconds_mean_per_window": (
            sum(row["offline_latency_seconds"][key] for row in records)
            / len(records) if records else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-windows", type=int, default=0)
    parser.add_argument(
        "--completion-dir", type=Path,
        default=ROOT / "diagnostics/phase8jp/completions",
    )
    args = parser.parse_args()
    entry = json.loads((ROOT / "reports/phase8jp_entry_gate.json").read_text())
    if entry.get("status") != "PASS":
        raise RuntimeError("Phase 8J-P entry Gate is not PASS")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Phase 8J-P oracle requires host CUDA")
    prior = json.loads(
        (ROOT / "reports/phase8jr_capacity_oracle.json").read_text()
    )
    suite_keys = {}
    for suite in ("valid_estimated", "valid_gt"):
        suite_keys[suite] = {
            (row["sequence_id"], int(row["frame_index"]))
            for row in prior["suites"][suite]["failure_records"]
            if row["taxonomy"] == "parameterization_limited"
        }
    union_keys = suite_keys["valid_estimated"] | suite_keys["valid_gt"]
    trainer, frozen = make_trainer()
    config = oracle_config()
    args.completion_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records_by_suite = {}
    for suite_name in ("valid_estimated", "valid_gt"):
        items = gather_unresolved(
            trainer, suite_name, suite_keys[suite_name], args.smoke_windows
        )
        if not args.smoke_windows and len(items) != len(suite_keys[suite_name]):
            raise RuntimeError(
                f"{suite_name}: resolved {len(items)} of "
                f"{len(suite_keys[suite_name])} unresolved keys"
            )
        records_by_key = {}
        pending = []
        for item in items:
            identity = (
                f"{suite_name}:{item['sequence_id']}:{item['frame_index']}"
            )
            path = args.completion_dir / (
                hashlib.sha256(identity.encode()).hexdigest() + ".json"
            )
            if path.is_file():
                payload = json.loads(path.read_text())
                if payload.get("config_sha256") == config["config_sha256"]:
                    records_by_key[item["key"]] = payload["result"]
                    continue
            pending.append((item, path))
        for start in range(0, len(pending), 8):
            chunk_with_paths = pending[start:start+8]
            chunk = [item for item, _ in chunk_with_paths]
            print(
                f"{suite_name} [{start+1}-{start+len(chunk)}/"
                f"{len(pending)}] O5/O6",
                flush=True,
            )
            results = run_chunk(trainer, chunk)
            for (item, path), result in zip(chunk_with_paths, results):
                atomic_json(path, {
                    "config_sha256": config["config_sha256"],
                    "suite": suite_name,
                    "result": result,
                })
                records_by_key[item["key"]] = result
        records_by_suite[suite_name] = [
            records_by_key[item["key"]] for item in items
        ]
    output_suffix = "_smoke" if args.smoke_windows else ""
    estimated_records = records_by_suite["valid_estimated"]
    gt_records = records_by_suite["valid_gt"]
    for scheme, filename in (
        ("o5a", "phase8jp_o5a_midpoint_position"),
        ("o5b", "phase8jp_o5b_midpoint_velocity"),
        ("o5c", "phase8jp_o5c_time_split"),
        ("o5d", "phase8jp_o5d_brake_yield"),
        ("o6", "phase8jp_o6_kinodynamic_oracle"),
    ):
        report = report_scheme(
            estimated_records, scheme, len(suite_keys["valid_estimated"])
        )
        report["valid_gt_summary"] = report_scheme(
            gt_records, scheme, len(suite_keys["valid_gt"])
        )
        report["oracle_config"] = config
        report["frozen"] = frozen
        report["offline_compute_seconds_sum_from_completions"] = (
            report["offline_latency_seconds_total"]
            + report["valid_gt_summary"]["offline_latency_seconds_total"]
        )
        atomic_json(
            ROOT / f"reports/{filename}{output_suffix}.json", report
        )
    union_recovered_est = sum(
        row["o5_union_success"] for row in estimated_records
    )
    union_recovered_gt = sum(
        row["o5_union_success"] for row in gt_records
    )
    combined = {
        "status": "PASS",
        "parameterization_audit_complete": (
            not args.smoke_windows
            and len(estimated_records) == len(suite_keys["valid_estimated"])
            and len(gt_records) == len(suite_keys["valid_gt"])
        ),
        "oracle_config": config,
        "union_unresolved_window_count": len(union_keys),
        "valid_estimated": {
            "unresolved_window_count": len(suite_keys["valid_estimated"]),
            "o5_union_recovered_count": union_recovered_est,
            "o5_union_joint_coverage_failure_fraction": (
                len(suite_keys["valid_estimated"]) - union_recovered_est
            ) / 2052.0,
            "o6_recovered_count": sum(
                row["o6"]["success"] for row in estimated_records
            ),
            "o6_joint_coverage_failure_fraction": (
                len(suite_keys["valid_estimated"]) - sum(
                    row["o6"]["success"] for row in estimated_records
                )
            ) / 2052.0,
        },
        "valid_gt": {
            "unresolved_window_count": len(suite_keys["valid_gt"]),
            "o5_union_recovered_count": union_recovered_gt,
            "o5_union_joint_coverage_failure_fraction": (
                len(suite_keys["valid_gt"]) - union_recovered_gt
            ) / 2052.0,
            "o6_recovered_count": sum(
                row["o6"]["success"] for row in gt_records
            ),
            "o6_joint_coverage_failure_fraction": (
                len(suite_keys["valid_gt"]) - sum(
                    row["o6"]["success"] for row in gt_records
                )
            ) / 2052.0,
        },
        "taxonomy": dict(Counter(row["taxonomy"] for row in estimated_records)),
        "records_by_suite": records_by_suite,
        "completion_dir": str(args.completion_dir.resolve()),
        "completion_count": sum(map(len, records_by_suite.values())),
        "offline_compute_seconds_sum_from_completions": sum(
            sum(row["offline_latency_seconds"].values())
            for records in records_by_suite.values() for row in records
        ),
        "production_test_used": False,
        "network_weights_modified": False,
        "score_stage_executed": False,
    }
    atomic_json(
        ROOT / f"reports/phase8jp_parameterization_comparison{output_suffix}.json",
        combined,
    )
    print(json.dumps({
        "status": "PASS",
        "smoke": bool(args.smoke_windows),
        "records": sum(map(len, records_by_suite.values())),
        "comparison": str(
            ROOT / f"reports/phase8jp_parameterization_comparison{output_suffix}.json"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
