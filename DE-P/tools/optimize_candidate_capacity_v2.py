#!/usr/bin/env python3
"""C1/C3 projected multistart V2 capacity oracle.

C1 optimizes terminal P/V/A for 15 latency-aware single quintics.  C3 adds one
continuous midpoint P/V/A (two quintic segments).  The network stays frozen;
this is per-window capacity optimization, not model training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from policy.safety_evaluator_v2 import SafetyEvaluatorV2Config
from tools.analyze_candidate_capacity_v2 import (
    DECISION, LEVELS, TOLERANCE, _ConfigAdapter, atomic_json,
    candidate_trajectories, c0_masks, collect_static_failure_states,
    local_basis, metric, sobol_parameters, static_clearance,
)
from tools.evaluate_candidate_decomposition_v2 import (
    ARTIFACTS_V2, EstimatedCovariance, SequenceData, vectorized_dynamic,
)


REPORTS = ROOT / "reports"
DATASET = ROOT / "data/phase8_dynamic_production"
ARTIFACTS = ROOT / "artifacts/phase8jv2_capacity"
COUNT = 15
SEED = 8241301
SINGLE_ITERATIONS = 40
PIECEWISE_ITERATIONS = 60


def boundary_inverse(duration, device, dtype):
    t = torch.as_tensor(duration, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    matrix = torch.stack((
        torch.stack((one, zero, zero, zero, zero, zero)),
        torch.stack((zero, one, zero, zero, zero, zero)),
        torch.stack((zero, zero, 2*one, zero, zero, zero)),
        torch.stack((one, t, t**2, t**3, t**4, t**5)),
        torch.stack((zero, one, 2*t, 3*t**2, 4*t**3, 5*t**4)),
        torch.stack((zero, zero, 2*one, 6*t, 12*t**2, 20*t**3)),
    ))
    return torch.linalg.inv(matrix)


def sample_segment(start, end, duration, points, include_start):
    inverse = boundary_inverse(duration, start.device, start.dtype)
    boundary = torch.cat((start, end), dim=-1)
    coefficients = torch.einsum("...xi,ji->...xj", boundary, inverse)
    first = 0.0 if include_start else float(duration) / points
    times = torch.linspace(
        first, float(duration), points + int(include_start),
        device=start.device, dtype=start.dtype,
    )
    powers = torch.stack((
        torch.ones_like(times), times, times**2, times**3, times**4,
        times**5,
    ), -1)
    velocity_powers = torch.stack((
        torch.zeros_like(times), torch.ones_like(times), 2*times,
        3*times**2, 4*times**3, 5*times**4,
    ), -1)
    acceleration_powers = torch.stack((
        torch.zeros_like(times), torch.zeros_like(times),
        2*torch.ones_like(times), 6*times, 12*times**2, 20*times**3,
    ), -1)
    return (
        torch.einsum("...xk,tk->...tx", coefficients, powers),
        torch.einsum("...xk,tk->...tx", coefficients, velocity_powers),
        torch.einsum(
            "...xk,tk->...tx", coefficients, acceleration_powers
        ),
    )


def decode_state(first, basis, parameters):
    position = (
        first[:, None, :, 0]
        + torch.einsum("bcx,bkx->bkc", basis, parameters[..., :3])
    )
    velocity = torch.einsum(
        "bcx,bkx->bkc", basis, parameters[..., 3:6]
    )
    acceleration = torch.einsum(
        "bcx,bkx->bkc", basis, parameters[..., 6:9]
    )
    return torch.stack((position, velocity, acceleration), dim=-1)


def trajectories(current, basis, parameters, config, mode, time_split=.5):
    latency = config.latency_s
    prefix_times = torch.as_tensor(
        [0.0, latency/2, latency],
        device=current.device, dtype=current.dtype,
    )
    prefix = (
        current[:, None, :, 0]
        + prefix_times[None, :, None] * current[:, None, :, 1]
        + .5 * prefix_times[None, :, None]**2
          * current[:, None, :, 2]
    )
    first = current.clone()
    first[:, :, 0] = prefix[:, -1]
    first[:, :, 1] = (
        current[:, :, 1] + latency * current[:, :, 2]
    )
    first = first[:, None].expand(-1, COUNT, -1, -1)
    duration = config.wall_clock_horizon_s - latency
    if mode == "single":
        end = decode_state(first[:, 0], basis, parameters)
        position, velocity, acceleration = sample_segment(
            first, end, duration,
            config.controlled_samples * config.subdivisions, True,
        )
    else:
        midpoint = decode_state(
            first[:, 0], basis, parameters[..., :9]
        )
        end = decode_state(
            first[:, 0], basis, parameters[..., 9:18]
        )
        first_duration = duration * float(time_split)
        second_duration = duration - first_duration
        total_points = config.controlled_samples * config.subdivisions
        first_points = max(1, round(total_points * float(time_split)))
        second_points = total_points - first_points
        first_position, first_velocity, first_acceleration = sample_segment(
            first, midpoint, first_duration, first_points, True,
        )
        second_position, second_velocity, second_acceleration = sample_segment(
            midpoint, end, second_duration, second_points, False,
        )
        position = torch.cat((first_position, second_position), dim=2)
        velocity = torch.cat((first_velocity, second_velocity), dim=2)
        acceleration = torch.cat(
            (first_acceleration, second_acceleration), dim=2
        )
    positions = torch.cat((
        prefix[:, :-1, None].expand(-1, -1, COUNT, -1).permute(0, 2, 1, 3),
        position,
    ), dim=2)
    return positions, velocity, acceleration


def parameter_bounds(mode, device, dtype):
    lower = torch.tensor(
        [.35, -4, -2.5, 0, -2.5, -2, -4, -4, -4],
        device=device, dtype=dtype,
    )
    upper = torch.tensor(
        [9.5, 4, 2.5, 5, 2.5, 2, 4, 4, 4],
        device=device, dtype=dtype,
    )
    if mode == "single":
        return lower, upper
    midpoint_lower = lower.clone()
    midpoint_upper = upper.clone()
    midpoint_upper[0] = 5.0
    return (
        torch.cat((midpoint_lower, lower)),
        torch.cat((midpoint_upper, upper)),
    )


def initial_parameters(batch, mode, device, dtype):
    base = torch.from_numpy(sobol_parameters()[:COUNT]).to(device, dtype)
    if mode == "single":
        values = base
    else:
        midpoint = base.clone()
        midpoint[:, :3] *= .5
        values = torch.cat((midpoint, base), dim=1)
    return values[None].expand(batch, -1, -1).clone()


def dynamic_clearance_torch(
    positions, actor_positions, radii, heights, cylinders, active,
    covariances, estimated, config,
):
    if actor_positions.shape[2] == 0:
        return torch.full(
            positions.shape[:2], float("inf"),
            device=positions.device, dtype=positions.dtype,
        )
    difference = (
        positions[:, :, :, None] - actor_positions[:, None]
    )
    sphere = torch.linalg.vector_norm(difference, dim=-1) - (
        config.uav_radius_m + radii[:, None, None]
    )
    radial = torch.relu(
        torch.linalg.vector_norm(difference[..., :2], dim=-1)
        - radii[:, None, None]
    )
    vertical = torch.relu(
        difference[..., 2].abs() - .5*heights[:, None, None]
    )
    cylinder = torch.hypot(radial, vertical) - config.uav_radius_m
    clearance = torch.where(
        cylinders[:, None, None], cylinder, sphere
    )
    if estimated:
        norm = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-8)
        direction = difference / norm[..., None]
        variance = torch.einsum(
            "bctmi,bmij,bctmj->bctm",
            direction, covariances, direction,
        ).clamp_min(0)
        clearance = clearance - config.confidence_sigma * variance.sqrt()
    clearance = clearance.masked_fill(
        ~active[:, None], float("inf")
    )
    return clearance.flatten(2).amin(2)


def static_sample_clearance(loss, positions, maps, radius):
    batch, count, points = positions.shape[:3]
    _, raw = loss.safety_loss.get_distance_cost(
        positions.reshape(batch, count*points, 3), maps
    )
    return raw.reshape(batch, count, points).amin(2) - radius


def optimize_batch(
    records, mode, suite, loss, config, device, iterations, time_split,
):
    current = torch.as_tensor(
        np.stack([row["current"] for row in records]),
        device=device, dtype=torch.float32,
    )
    basis = torch.as_tensor(
        np.stack([local_basis(row["direction"]) for row in records]),
        device=device, dtype=torch.float32,
    )
    maps = torch.as_tensor(
        [row["map"] for row in records], device=device, dtype=torch.long
    )
    parameters = torch.nn.Parameter(
        initial_parameters(len(records), mode, device, torch.float32)
    )
    lower, upper = parameter_bounds(mode, device, torch.float32)
    optimizer = torch.optim.Adam([parameters], lr=.08)
    actor = records[0].get("actor_tensors")
    if actor is not None:
        actor = {
            key: torch.as_tensor(
                np.stack([row["actor_tensors"][key] for row in records]),
                device=device,
            ) for key in actor
        }
    finite_gradient = True
    zero_gradient_steps = 0
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        position, velocity, acceleration = trajectories(
            current, basis, parameters, config, mode, time_split
        )
        static = static_sample_clearance(
            loss, position, maps, config.uav_radius_m
        )
        if actor is None:
            dynamic = torch.full_like(static, float("inf"))
        else:
            dynamic = dynamic_clearance_torch(
                position, actor["positions"], actor["radii"],
                actor["heights"], actor["cylinders"], actor["active"],
                actor["covariances"], suite == "valid_estimated", config,
            )
        joint = torch.minimum(static, dynamic)
        # Smoothly approximate best-candidate maximum clearance while keeping
        # gradients on multiple starts.
        violation = torch.nn.functional.softplus(-joint * 8) / 8
        speed = torch.linalg.vector_norm(velocity, dim=-1).amax(2)
        accel = torch.linalg.vector_norm(acceleration, dim=-1).amax(2)
        violation = (
            violation + torch.relu(speed-6.0) + torch.relu(accel-6.0)
        )
        objective = torch.topk(
            violation, k=3, largest=False, dim=1
        ).values.mean()
        objective.backward()
        gradient = parameters.grad
        finite_gradient &= bool(torch.isfinite(gradient).all())
        zero_gradient_steps += int(float(gradient.norm()) <= 1e-12)
        optimizer.step()
        with torch.no_grad():
            parameters.clamp_(lower, upper)
    with torch.no_grad():
        position, velocity, acceleration = trajectories(
            current, basis, parameters, config, mode, time_split
        )
        feasible = (
            torch.linalg.vector_norm(velocity, dim=-1).amax(2)
            <= 6.0 + TOLERANCE
        ) & (
            torch.linalg.vector_norm(acceleration, dim=-1).amax(2)
            <= 6.0 + TOLERANCE
        )
    return (
        position.detach().cpu().numpy(),
        feasible.detach().cpu().numpy(),
        bool(finite_gradient),
        int(zero_gradient_steps),
    )


def actor_tensors(sequence, frame, times, covariances):
    actors_by_time = sequence.actors_at(frame, times)
    identifiers = sorted({
        int(actor["object_id"]) for actors in actors_by_time for actor in actors
        if actor.get("dynamic", True)
    })
    count = len(identifiers)
    lookup = {identifier: index for index, identifier in enumerate(identifiers)}
    positions = np.zeros((len(times), count, 3), np.float32)
    radii = np.zeros(count, np.float32)
    heights = np.zeros(count, np.float32)
    cylinders = np.zeros(count, bool)
    active = np.zeros((len(times), count), bool)
    covariance = np.zeros((count, 3, 3), np.float32)
    for time_index, actors in enumerate(actors_by_time):
        for actor in actors:
            identifier = int(actor["object_id"])
            if identifier not in lookup or not actor.get("dynamic", True):
                continue
            index = lookup[identifier]
            positions[time_index, index] = actor["position_world"]
            radii[index] = actor["radius"]
            heights[index] = actor.get("height", 2*actor["radius"])
            cylinders[index] = actor.get("type") == "vertical_cylinder"
            active[time_index, index] = actor.get("active", True)
            covariance[index] = covariances.get(
                identifier, np.zeros((3, 3))
            )
    return {
        "positions": positions, "radii": radii, "heights": heights,
        "cylinders": cylinders, "active": active,
        "covariances": covariance,
    }, actors_by_time


def pad_actor_records(records):
    maximum = max(
        row["actor_tensors"]["positions"].shape[1] for row in records
    )
    for row in records:
        value = row["actor_tensors"]
        count = value["positions"].shape[1]
        if count == maximum:
            continue
        pad = maximum-count
        value["positions"] = np.pad(
            value["positions"], ((0, 0), (0, pad), (0, 0))
        )
        value["radii"] = np.pad(value["radii"], (0, pad))
        value["heights"] = np.pad(value["heights"], (0, pad))
        value["cylinders"] = np.pad(
            value["cylinders"], (0, pad), constant_values=False
        )
        value["active"] = np.pad(
            value["active"], ((0, 0), (0, pad)), constant_values=False
        )
        value["covariances"] = np.pad(
            value["covariances"], ((0, pad), (0, 0), (0, 0))
        )


def exact_success(
    records, positions, feasible, suite, loss, config, device,
):
    static = static_clearance(
        loss, positions, [row["map"] for row in records],
        device, config.uav_radius_m,
    )
    success = np.zeros(len(records), bool)
    for row, record in enumerate(records):
        if "actors_by_time" not in record:
            dynamic = np.full(COUNT, np.inf)
        else:
            dynamic_result = vectorized_dynamic(
                _ConfigAdapter(config), positions[row],
                record["actors_by_time"], suite, record["covariances"],
            )
            dynamic = (
                dynamic_result["planning_min"]
                if suite == "valid_estimated"
                else dynamic_result["continuous_min"]
            )
        success[row] = bool((
            (static[row] >= -TOLERANCE)
            & (dynamic >= -TOLERANCE)
            & feasible[row]
        ).any())
    return success


def dynamic_records(suite, data, config, maximum):
    c0_safe, preventable = c0_masks(data, suite)
    indices = np.flatnonzero(preventable & ~c0_safe.any(1))
    if maximum:
        indices = indices[:maximum]
    cache = EstimatedCovariance()
    sequences = {}
    # All V2 candidate timelines share this fixed time grid.
    current_fixture = np.zeros((3, 3))
    _, times, _ = candidate_trajectories(
        current_fixture, [1, 0, 0], sobol_parameters()[:1], config
    )
    records = []
    for row in indices:
        sequence_id = str(data["v1_sequence"][row])
        frame = int(data["v1_frame"][row])
        sequence = sequences.setdefault(
            sequence_id, SequenceData(sequence_id)
        )
        current = sequence.current_state(frame)
        old = data["v1_trajectory"][row]
        current_actors = sequence.actors_at(frame, [0.0])[0]
        covariances = (
            cache.get(sequence_id, frame, current_actors)
            if suite == "valid_estimated" else {}
        )
        tensors, actors = actor_tensors(
            sequence, frame, times, covariances
        )
        records.append({
            "index": int(row), "current": current,
            "direction": old[:, -1].mean(0)-current[:, 0],
            "map": int(data["v1_map"][row]),
            "actor_tensors": tensors, "actors_by_time": actors,
            "covariances": covariances,
        })
    return records, int(preventable.sum()), int(
        (preventable & ~c0_safe.any(1)).sum()
    )


def run_suite(
    suite, mode, records, denominator, original_failures,
    loss, config, device, iterations, batch_size, time_split,
):
    successes = 0
    finite_gradient = True
    zero_gradient_steps = 0
    started = time.perf_counter()
    for begin in range(0, len(records), batch_size):
        batch = records[begin:begin+batch_size]
        if suite != "valid_static":
            pad_actor_records(batch)
        position, feasible, finite, zero = optimize_batch(
            batch, mode, suite, loss, config, device, iterations,
            time_split,
        )
        exact = exact_success(
            batch, position, feasible, suite, loss, config, device
        )
        successes += int(exact.sum())
        finite_gradient &= finite
        zero_gradient_steps += zero
        if (begin + len(batch)) % 128 == 0:
            print(
                f"{mode} {suite} {begin+len(batch)}/{len(records)}",
                flush=True,
            )
    return {
        "remaining_failure": metric(
            original_failures-successes, denominator,
            (
                "all 10,000 valid_static windows"
                if suite == "valid_static"
                else "t0-safe and first-controllable-safe windows"
            ),
            (
                "none" if suite == "valid_static"
                else "already-unsafe or first-controllable-unsafe windows"
            ),
        ),
        "recovered_c0_failures": successes,
        "audited_failure_count": len(records),
        "full_failure_count": original_failures,
        "gradient_finite": finite_gradient,
        "zero_gradient_step_count": zero_gradient_steps,
        "iterations": iterations,
        "candidate_count": COUNT,
        "elapsed_seconds": time.perf_counter()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("single", "piecewise"), required=True
    )
    parser.add_argument("--smoke-failures", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-split", type=float, default=.5)
    args = parser.parse_args()
    if not .2 <= args.time_split <= .8:
        raise ValueError("time split must be within [0.2,0.8]")
    if args.mode == "single" and args.time_split != .5:
        raise ValueError("time split only applies to piecewise mode")
    entry = json.loads(
        (REPORTS / "phase8jv2_entry_gate.json").read_text()
    )
    oracle_path = REPORTS / "phase8jv2_capacity_oracle.json"
    oracle = json.loads(oracle_path.read_text())
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        REPORTS / "phase8jq_controller_authoritative_envelope.json",
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    iterations = (
        SINGLE_ITERATIONS if args.mode == "single"
        else PIECEWISE_ITERATIONS
    )
    dynamic_loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=DATASET / "map_catalog.yaml",
    )
    results = {}
    for suite in ("valid_estimated", "valid_gt"):
        data = dict(np.load(
            ARTIFACTS_V2 / f"{DECISION}-{suite}.npz",
            allow_pickle=True,
        ))
        v1 = np.load(
            ROOT / f"artifacts/phase8i/{DECISION}-{suite}.npz",
            allow_pickle=True,
        )
        data["v1_trajectory"] = v1["trajectory"]
        records, denominator, original = dynamic_records(
            suite, data, config, args.smoke_failures
        )
        results[suite] = run_suite(
            suite, args.mode, records, denominator, original,
            dynamic_loss, config, device, iterations, args.batch_size,
            args.time_split,
        )
    del dynamic_loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    static_loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=ROOT / "configs/static_map_catalog.yaml",
    )
    static_artifact = dict(np.load(
        ARTIFACTS_V2 / f"{DECISION}-valid_static-v2.npz",
        allow_pickle=True,
    ))
    static_failure = ~(
        np.asarray(static_artifact["clearance"]) >= -TOLERANCE
    ).any(1)
    static_records = collect_static_failure_states(
        static_failure, args.smoke_failures
    )
    results["valid_static"] = run_suite(
        "valid_static", args.mode, static_records,
        len(static_failure), int(static_failure.sum()),
        static_loss, config, device, iterations, args.batch_size,
        args.time_split,
    )
    key = (
        "c1_direct_optimization"
        if args.mode == "single"
        else "c3_richer_parameterization"
    )
    value = {
        "status": "PASS",
        "scope": "SMOKE" if args.smoke_failures else "FULL_VALIDATION",
        "parameterization": (
            "15 projected-multistart terminal-PVA single quintics"
            if args.mode == "single"
            else "15 projected-multistart continuous midpoint-PVA piecewise quintics"
        ),
        "network_frozen": True,
        "score_frozen": True,
        "time_split": args.time_split if args.mode == "piecewise" else None,
        "results": results,
        "physical_feasibility": {
            "velocity_limit_mps": 6.0,
            "acceleration_limit_mps2": 6.0,
            "continuous_pva": True,
        },
        "production_test_used": False,
        "training_executed": False,
    }
    variant = (
        args.mode if args.mode == "single"
        else f"piecewise_{int(round(args.time_split*100)):02d}"
    )
    suffix = "_smoke" if args.smoke_failures else ""
    atomic_json(
        REPORTS / f"phase8jv2_capacity_{variant}{suffix}.json",
        {**{
            name: oracle[name] for name in (
                "evaluator_version", "config_hash", "geometry_hash",
                "timeline_hash", "uncertainty_policy_hash",
                "Simulator_geometry_hash", "semantic_determinism_hash",
                "dataset_manifest_hash", "cache_index_hash",
                "checkpoint_hash",
            )
        }, **value},
    )
    if not args.smoke_failures:
        if args.mode == "single":
            oracle[key] = value
        else:
            existing = oracle.get(key, {})
            if existing.get("status") == "PASS" and "results" in existing:
                existing = {"piecewise_50": existing}
            existing[variant] = value
            oracle[key] = existing
        atomic_json(oracle_path, oracle)
    print(json.dumps({
        "status": "PASS", "mode": args.mode, "scope": value["scope"],
        "remaining_failure": {
            suite: row["remaining_failure"]["fraction"]
            for suite, row in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
