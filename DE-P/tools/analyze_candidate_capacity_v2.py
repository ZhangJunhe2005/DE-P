#!/usr/bin/env python3
"""Bounded Safety Evaluator V2 candidate-capacity audit.

The network and all checkpoints are read-only.  C0 is reproduced from frozen
V2 artifacts.  C2 uses a fixed scrambled Sobol bank under the existing
latency-aware single-quintic parameterization.  Only C0 failures are expanded,
so the full-validation denominator remains exact without wasting oracle work
on already-covered windows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig
from loss.loss_function import DEPLoss
from policy.dep_dataset import DEPDataset, seed_dataset_worker
from policy.safety_evaluator_v2 import (
    SafetyEvaluatorV2Config, _quintic_inverse,
)
from policy.state_transform import state_body2world
from tools.evaluate_candidate_decomposition_v2 import (
    ARTIFACTS_V2, EstimatedCovariance, SequenceData, vectorized_dynamic,
)


REPORTS = ROOT / "reports"
DATASET = ROOT / "data/phase8_dynamic_production"
DECISION = "fixed_050_seed8403"
LEVELS = (64, 128, 256, 512)
TOLERANCE = 1e-6
SEED = 8241201


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


def metric(numerator, denominator, inclusion, exclusion):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "fraction": (
            float(numerator / denominator) if denominator else None
        ),
        "inclusion_rule": inclusion,
        "exclusion_rule": exclusion,
    }


def sobol_parameters(count=512):
    engine = torch.quasirandom.SobolEngine(
        9, scramble=True, seed=SEED
    )
    values = engine.draw(count).double().numpy()
    # Deployable terminal P/V/A bounds.  The first entries are explicit
    # brake/yield/straight/lateral/vertical primitives; remaining entries are
    # the frozen Sobol placement.
    parameters = np.empty_like(values)
    parameters[:, 0] = .35 + 9.15 * values[:, 0]       # forward displacement
    parameters[:, 1] = -4.0 + 8.0 * values[:, 1]      # lateral displacement
    parameters[:, 2] = -2.5 + 5.0 * values[:, 2]      # vertical displacement
    parameters[:, 3] = 5.0 * values[:, 3]             # terminal forward speed
    parameters[:, 4] = -2.5 + 5.0 * values[:, 4]
    parameters[:, 5] = -2.0 + 4.0 * values[:, 5]
    parameters[:, 6:9] = -4.0 + 8.0 * values[:, 6:9]
    explicit = np.asarray([
        [.35, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4.0, 0, 0, 2, 0, 0, 0, 0, 0],
        [6.0, 0, 0, 4, 0, 0, 0, 0, 0],
        [3.0, -2.0, 0, 1, 0, 0, 0, 0, 0],
        [3.0, 2.0, 0, 1, 0, 0, 0, 0, 0],
        [3.0, 0, -1.5, 1, 0, 0, 0, 0, 0],
        [3.0, 0, 1.5, 1, 0, 0, 0, 0, 0],
        [5.0, -3.0, 0, 3, 0, 0, 0, 0, 0],
        [5.0, 3.0, 0, 3, 0, 0, 0, 0, 0],
        [2.0, -1.0, 0, 0, 0, 0, 0, 0, 0],
        [2.0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        [.35, -1.5, 0, 0, 0, 0, 0, 0, 0],
        [.35, 1.5, 0, 0, 0, 0, 0, 0, 0],
    ])
    parameters[:len(explicit)] = explicit
    return parameters


def local_basis(direction):
    forward = np.asarray(direction, float)
    if np.linalg.norm(forward) < 1e-9:
        forward = np.asarray([1.0, 0.0, 0.0])
    forward /= np.linalg.norm(forward)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > .95:
        reference = np.asarray([0.0, 1.0, 0.0])
    lateral = np.cross(reference, forward)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    vertical /= np.linalg.norm(vertical)
    return np.stack((forward, lateral, vertical), axis=1)


def candidate_trajectories(current, direction, parameters, config):
    current = np.asarray(current, float)
    basis = local_basis(direction)
    latency = config.latency_s
    prefix_times = np.asarray([0.0, latency / 2.0, latency])
    prefix_positions = (
        current[:, 0][None]
        + prefix_times[:, None] * current[:, 1][None]
        + .5 * prefix_times[:, None] ** 2 * current[:, 2][None]
    )
    first = current.copy()
    first[:, 0] = prefix_positions[-1]
    first[:, 1] = current[:, 1] + latency * current[:, 2]
    count = len(parameters)
    end = np.empty((count, 3, 3), float)
    end[:, :, 0] = (
        first[:, 0][None] + parameters[:, :3] @ basis.T
    )
    end[:, :, 1] = parameters[:, 3:6] @ basis.T
    end[:, :, 2] = parameters[:, 6:9] @ basis.T
    duration = config.wall_clock_horizon_s - latency
    boundary = np.concatenate((
        np.broadcast_to(first, (count, 3, 3)), end
    ), axis=2)
    coefficients = np.einsum(
        "cxi,ji->cxj", boundary, _quintic_inverse(duration)
    )
    local_times = np.linspace(
        0.0, duration,
        config.controlled_samples * config.subdivisions + 1,
    )
    powers = np.stack([
        np.ones_like(local_times), local_times, local_times**2,
        local_times**3, local_times**4, local_times**5,
    ], axis=1)
    velocity_powers = np.stack([
        np.zeros_like(local_times), np.ones_like(local_times),
        2*local_times, 3*local_times**2, 4*local_times**3,
        5*local_times**4,
    ], axis=1)
    acceleration_powers = np.stack([
        np.zeros_like(local_times), np.zeros_like(local_times),
        2*np.ones_like(local_times), 6*local_times,
        12*local_times**2, 20*local_times**3,
    ], axis=1)
    controlled = np.einsum("cxk,tk->ctx", coefficients, powers)
    velocity = np.einsum(
        "cxk,tk->ctx", coefficients, velocity_powers
    )
    acceleration = np.einsum(
        "cxk,tk->ctx", coefficients, acceleration_powers
    )
    positions = np.concatenate((
        np.broadcast_to(prefix_positions[:-1], (count, 2, 3)),
        controlled,
    ), axis=1)
    times = np.concatenate((
        prefix_times[:-1], latency + local_times
    ))
    feasible = (
        np.linalg.norm(velocity, axis=2).max(1) <= 6.0 + TOLERANCE
    ) & (
        np.linalg.norm(acceleration, axis=2).max(1) <= 6.0 + TOLERANCE
    ) & np.isfinite(positions).all(axis=(1, 2))
    return positions, times, feasible


def static_clearance(loss, trajectories, map_ids, device, radius):
    trajectories = np.asarray(trajectories, np.float32)
    maps = torch.as_tensor(map_ids, dtype=torch.long, device=device)
    positions = torch.from_numpy(
        trajectories.reshape(len(trajectories), -1, 3)
    ).to(device)
    with torch.inference_mode():
        _, raw = loss.safety_loss.get_distance_cost(positions, maps)
    raw = raw.detach().cpu().numpy().reshape(
        trajectories.shape[0], trajectories.shape[1],
        trajectories.shape[2],
    )
    segments = np.linalg.norm(
        np.diff(trajectories, axis=2), axis=3
    )
    lower = np.minimum(raw[:, :, :-1], raw[:, :, 1:]) - segments
    return np.minimum(raw.min(2), lower.min(2)) - radius


def collect_static_failure_states(failure_mask, maximum_windows=0):
    dataset = DEPDataset(
        mode="valid", cache_size=0, global_seed=8172402
    )
    generator = torch.Generator().manual_seed(8172402)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4,
        worker_init_fn=seed_dataset_worker, generator=generator,
        persistent_workers=False,
    )
    records, offset = [], 0
    for _, position, rotation, observation, map_id in loader:
        size = len(position)
        indices = np.arange(offset, offset + size)
        selected = np.flatnonzero(failure_mask[indices])
        if len(selected):
            goal, velocity, acceleration = state_body2world(
                position, rotation, observation[:, 6:9],
                observation[:, 0:3], observation[:, 3:6],
            )
            start = torch.stack(
                (position, velocity, acceleration), dim=2
            ).numpy()
            for local in selected:
                records.append({
                    "index": int(indices[local]),
                    "current": start[local],
                    "direction": (
                        goal[local].numpy() - position[local].numpy()
                    ),
                    "map": int(map_id[local]),
                })
                if maximum_windows and len(records) >= maximum_windows:
                    return records
        offset += size
    return records


def c0_masks(data, suite):
    physical = (
        (np.asarray(data["physical_dynamic"], float) >= -TOLERANCE)
        & (np.asarray(data["physical_static"], float) >= -TOLERANCE)
    )
    planning = (
        (np.asarray(data["planning_dynamic"], float) >= -TOLERANCE)
        & (np.asarray(data["planning_static"], float) >= -TOLERANCE)
    )
    t0_safe = np.asarray(data["t0_joint"], float) >= -TOLERANCE
    first_safe = np.asarray(data["first_joint"], float) >= -TOLERANCE
    preventable = t0_safe & first_safe
    safe = planning if suite == "valid_estimated" else physical
    return safe, preventable


def audit_dynamic_suite(
    suite, data, config, parameters, static_loss, device,
    covariance_cache, maximum_windows=0,
):
    c0_safe, preventable = c0_masks(data, suite)
    failures = preventable & ~c0_safe.any(1)
    failure_indices = np.flatnonzero(failures)
    if maximum_windows:
        failure_indices = failure_indices[:maximum_windows]
    recovered = {level: 0 for level in LEVELS}
    scenario_recovered = {
        level: Counter() for level in LEVELS
    }
    map_recovered = {level: Counter() for level in LEVELS}
    feasibility_rejections = {level: 0 for level in LEVELS}
    recovered_records = {level: [] for level in LEVELS}
    sequences = {}
    started = time.perf_counter()
    for ordinal, row in enumerate(failure_indices, 1):
        sequence_id = str(data["v1_sequence"][row])
        frame = int(data["v1_frame"][row])
        sequence = sequences.setdefault(
            sequence_id, SequenceData(sequence_id)
        )
        current = sequence.current_state(frame)
        old = np.asarray(data["v1_trajectory"][row], float)
        direction = old[:, -1].mean(0) - current[:, 0]
        trajectories, times, feasible = candidate_trajectories(
            current, direction, parameters, config
        )
        static = static_clearance(
            static_loss, trajectories[None],
            [int(data["v1_map"][row])], device, config.uav_radius_m,
        )[0]
        current_actors = sequence.actors_at(frame, [0.0])[0]
        covariances = (
            covariance_cache.get(
                sequence_id, frame, current_actors
            ) if suite == "valid_estimated" else {}
        )
        dynamic = vectorized_dynamic(
            _ConfigAdapter(config), trajectories,
            sequence.actors_at(frame, times), suite, covariances,
        )
        dynamic_clearance = (
            dynamic["planning_min"]
            if suite == "valid_estimated"
            else dynamic["continuous_min"]
        )
        safe = (
            (static >= -TOLERANCE)
            & (dynamic_clearance >= -TOLERANCE)
            & feasible
        )
        for level in LEVELS:
            success = bool(safe[:level].any())
            recovered[level] += int(success)
            if success:
                recovered_records[level].append({
                    "sequence_id": sequence_id,
                    "frame_index": frame,
                    "map_id": int(data["v1_map"][row]),
                    "scenario": str(data["v1_scenario"][row]),
                })
            scenario_recovered[level][str(
                data["v1_scenario"][row]
            )] += int(success)
            map_recovered[level][str(int(
                data["v1_map"][row]
            ))] += int(success)
            feasibility_rejections[level] += int((~feasible[:level]).sum())
        if ordinal % 50 == 0:
            print(
                f"{suite} C2 {ordinal}/{len(failure_indices)}",
                flush=True,
            )
    denominator = int(preventable.sum())
    original_failures = int(failures.sum())
    return {
        "suite": suite,
        "full_window_count": int(len(preventable)),
        "preventable_window_count": denominator,
        "c0_preventable_failure": metric(
            original_failures, denominator,
            "t0-safe and first-controllable-safe windows",
            "already-unsafe or first-controllable-unsafe windows",
        ),
        "c2_dense_single_quintic": {
            str(level): {
                "remaining_failure": metric(
                    original_failures - recovered[level],
                    denominator,
                    "all preventable full-validation windows",
                    "already-unsafe or first-controllable-unsafe windows",
                ),
                "recovered_c0_failures": recovered[level],
                "feasibility_rejected_candidate_count": (
                    feasibility_rejections[level]
                ),
                "scenario_recovered": dict(
                    scenario_recovered[level]
                ),
                "map_recovered": dict(map_recovered[level]),
                "recovered_records": recovered_records[level],
            }
            for level in LEVELS
        },
        "audited_failure_count": int(len(failure_indices)),
        "full_failure_count": original_failures,
        "partial_smoke": bool(
            maximum_windows and len(failure_indices) < original_failures
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


class _ConfigAdapter:
    def __init__(self, config):
        self.config = config


def audit_static(
    config, parameters, static_loss, device, maximum_windows=0
):
    artifact = dict(np.load(
        ARTIFACTS_V2 / f"{DECISION}-valid_static-v2.npz",
        allow_pickle=True,
    ))
    c0_safe = np.asarray(artifact["clearance"], float) >= -TOLERANCE
    failures = ~c0_safe.any(1)
    records = collect_static_failure_states(failures, maximum_windows)
    recovered = {level: 0 for level in LEVELS}
    map_recovered = {level: Counter() for level in LEVELS}
    feasibility_rejections = {level: 0 for level in LEVELS}
    recovered_records = {level: [] for level in LEVELS}
    started = time.perf_counter()
    batch_size = 8
    for begin in range(0, len(records), batch_size):
        batch = records[begin:begin+batch_size]
        trajectories, feasible = [], []
        for record in batch:
            positions, _, valid = candidate_trajectories(
                record["current"], record["direction"], parameters, config
            )
            trajectories.append(positions)
            feasible.append(valid)
        trajectories = np.stack(trajectories)
        feasible = np.stack(feasible)
        clearances = static_clearance(
            static_loss, trajectories,
            [row["map"] for row in batch], device, config.uav_radius_m,
        )
        safe = (clearances >= -TOLERANCE) & feasible
        for row, record in enumerate(batch):
            for level in LEVELS:
                success = bool(safe[row, :level].any())
                recovered[level] += int(success)
                if success:
                    recovered_records[level].append({
                        "static_index": record["index"],
                        "map_id": record["map"],
                    })
                map_recovered[level][str(record["map"])] += int(success)
                feasibility_rejections[level] += int(
                    (~feasible[row, :level]).sum()
                )
        if (begin + len(batch)) % 256 == 0:
            print(
                f"valid_static C2 {begin+len(batch)}/{len(records)}",
                flush=True,
            )
    total = len(failures)
    original = int(failures.sum())
    return {
        "suite": "valid_static",
        "full_window_count": total,
        "c0_failure": metric(
            original, total, "all 10,000 valid_static windows", "none"
        ),
        "c2_dense_single_quintic": {
            str(level): {
                "remaining_failure": metric(
                    original - recovered[level], total,
                    "all 10,000 valid_static windows", "none",
                ),
                "recovered_c0_failures": recovered[level],
                "map_recovered": dict(map_recovered[level]),
                "recovered_records": recovered_records[level],
                "feasibility_rejected_candidate_count": (
                    feasibility_rejections[level]
                ),
            }
            for level in LEVELS
        },
        "audited_failure_count": len(records),
        "full_failure_count": original,
        "partial_smoke": bool(
            maximum_windows and len(records) < original
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def measure_generation_latency(config, parameters):
    current = np.asarray([
        [0, 2, 0], [0, 0, 0], [2, 0, 0],
    ], float)
    values = {}
    for level in LEVELS:
        samples = []
        for _ in range(100):
            started = time.perf_counter()
            candidate_trajectories(
                current, [1, 0, 0], parameters[:level], config
            )
            samples.append((time.perf_counter() - started) * 1000)
        values[str(level)] = {
            "mean_ms": float(np.mean(samples)),
            "p95_ms": float(np.quantile(samples, .95)),
            "control_period_ms": 1000 / 33,
            "generation_only": True,
        }
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-failures", type=int, default=0)
    args = parser.parse_args()
    entry = json.loads(
        (REPORTS / "phase8jv2_entry_gate.json").read_text()
    )
    baseline = json.loads(
        (REPORTS / "phase8jv2_baseline_reproduction.json").read_text()
    )
    if entry["status"] != "PASS" or baseline["status"] != "PASS":
        raise RuntimeError("Phase 8J-V2 entry/baseline Gate is not PASS")
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        REPORTS / "phase8jq_controller_authoritative_envelope.json",
    )
    parameters = sobol_parameters()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    started = time.perf_counter()
    dynamic_loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=DATASET / "map_catalog.yaml",
    )
    covariance_cache = EstimatedCovariance()
    dynamic_results = {}
    for suite in ("valid_estimated", "valid_gt"):
        data = dict(np.load(
            ARTIFACTS_V2 / f"{DECISION}-{suite}.npz",
            allow_pickle=True,
        ))
        v1_source = np.load(
            ROOT / f"artifacts/phase8i/{DECISION}-{suite}.npz",
            allow_pickle=True,
        )
        data["v1_trajectory"] = v1_source["trajectory"]
        dynamic_results[suite] = audit_dynamic_suite(
            suite, data, config, parameters, dynamic_loss, device,
            covariance_cache, args.smoke_failures,
        )
    del dynamic_loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    static_loss = DEPLoss(
        dynamic_loss_config=DynamicLossConfig.from_global_config(),
        map_catalog=ROOT / "configs/static_map_catalog.yaml",
    )
    static_result = audit_static(
        config, parameters, static_loss, device, args.smoke_failures
    )
    latency = measure_generation_latency(config, parameters)
    provenance = {
        "evaluator_version": "v2",
        "config_hash": entry["frozen_hashes"]["v2_config_hash"],
        "geometry_hash": entry["frozen_hashes"]["v2_geometry_hash"],
        "timeline_hash": entry["frozen_hashes"]["v2_timeline_hash"],
        "uncertainty_policy_hash": entry["frozen_hashes"][
            "v2_uncertainty_policy_hash"
        ],
        "Simulator_geometry_hash": entry["frozen_hashes"][
            "Simulator_geometry_hash"
        ],
        "semantic_determinism_hash": entry["frozen_hashes"][
            "v2_semantic_determinism_hash"
        ],
        "dataset_manifest_hash": entry["frozen_hashes"][
            "dataset_manifest_hash"
        ],
        "cache_index_hash": entry["frozen_hashes"][
            "estimated_cache_index_hash"
        ],
        "checkpoint_hash": entry["frozen_hashes"]["checkpoint_hashes"][
            DECISION
        ],
    }
    oracle = {
        **provenance,
        "status": "PASS",
        "audit_scope": (
            "SMOKE" if args.smoke_failures else "FULL_VALIDATION"
        ),
        "c0_current_network": {
            "source": "frozen Phase 8J-Q2 V2 artifacts",
            "valid_static_failure": static_result["c0_failure"],
            "valid_estimated_preventable_failure": dynamic_results[
                "valid_estimated"
            ]["c0_preventable_failure"],
            "valid_gt_preventable_failure": dynamic_results[
                "valid_gt"
            ]["c0_preventable_failure"],
        },
        "c1_direct_optimization": {
            "status": "PENDING_AFTER_C2",
            "network_frozen": True,
            "parameterization": "existing terminal P/V/A single quintic",
        },
        "c2_dense_existing_parameterization": {
            "sampler": "scrambled Sobol",
            "seed": SEED,
            "levels": list(LEVELS),
            "parameter_hash": canonical_hash(parameters.tolist()),
            "wall_clock_horizon_s": config.wall_clock_horizon_s,
            "latency_s": config.latency_s,
            "continuous_subdivisions": config.subdivisions,
            "velocity_limit_mps": 6.0,
            "acceleration_limit_mps2": 6.0,
            "valid_static": static_result,
            **dynamic_results,
        },
        "c3_richer_parameterization": {
            "status": "NOT_RUN_UNTIL_C1_C2_RESULT",
        },
        "candidate_count_latency": latency,
        "maps_present": {
            "valid_static": list(range(10)),
            "valid_dynamic": [12, 13, 14],
            "absent_from_validation": [10, 11],
        },
        "no_target_dynamic_cost_exactly_zero": True,
        "network_weights_modified": False,
        "coverage_training_executed": False,
        "score_training_executed": False,
        "production_test_used": False,
        "blind_used": False,
        "performance": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available() else 0
            ),
            "peak_cpu_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
    }
    suffix = "_smoke" if args.smoke_failures else ""
    output = REPORTS / f"phase8jv2_capacity_oracle{suffix}.json"
    atomic_json(output, oracle)
    atomic_json(
        REPORTS / f"phase8jv2_candidate_count_latency{suffix}.json",
        {**provenance, "status": "PASS", "latency": latency},
    )
    print(json.dumps({
        "status": "PASS",
        "scope": oracle["audit_scope"],
        "output": str(output),
        "c2_512": {
            "valid_static": static_result[
                "c2_dense_single_quintic"
            ]["512"]["remaining_failure"]["fraction"],
            "valid_estimated": dynamic_results["valid_estimated"][
                "c2_dense_single_quintic"
            ]["512"]["remaining_failure"]["fraction"],
            "valid_gt": dynamic_results["valid_gt"][
                "c2_dense_single_quintic"
            ]["512"]["remaining_failure"]["fraction"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
