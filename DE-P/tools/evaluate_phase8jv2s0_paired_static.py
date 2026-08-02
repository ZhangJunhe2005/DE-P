#!/usr/bin/env python3
"""Paired C0 inference and legacy static-certificate evaluation."""

from __future__ import annotations

import hashlib
import json
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

from policy.checkpoint_utils import load_dep_checkpoint
from policy.safety_evaluator_v2 import (
    SafetyEvaluatorV2Config, evaluate_quintic, quintic_coefficients,
)
from policy.static_v2_fixture import StaticV2FixtureDataset
from tools.run_phase8i_failure_decomposition import make_runtime


COUNT = 15
TOLERANCE = 1e-6
FIXTURE = ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1.npz"
FIXTURE_MANIFEST = (
    ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1_manifest.json"
)
ARTIFACT = ROOT / "artifacts/phase8jv2s0/fixed_050_seed8403-paired-static.npz"
REPORT = ROOT / "reports/phase8jv2s0_initial_state_envelope.json"


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metric(flags, denominator_rule):
    flags = np.asarray(flags, bool)
    return {
        "numerator": int(flags.sum()),
        "denominator": int(len(flags)),
        "fraction": float(flags.mean()),
        "inclusion_rule": denominator_rule,
        "exclusion_rule": "none",
    }


def distribution(values):
    values = np.asarray(values, float)
    return {
        "min": float(values.min()),
        "q10": float(np.quantile(values, .1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, .9)),
        "q99": float(np.quantile(values, .99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def timelines_from_states(start, end, config):
    batch = len(start)
    latency = config.latency_s
    prefix_times = np.asarray([0.0, latency / 2.0, latency])
    prefix = (
        start[:, None, :, 0]
        + prefix_times[None, :, None] * start[:, None, :, 1]
        + .5 * prefix_times[None, :, None] ** 2 * start[:, None, :, 2]
    )
    first = start.copy()
    first[:, :, 0] = prefix[:, -1]
    first[:, :, 1] = start[:, :, 1] + latency * start[:, :, 2]
    duration = config.wall_clock_horizon_s - latency
    local_times = np.linspace(
        0.0, duration,
        config.controlled_samples * config.subdivisions + 1,
    )
    output = np.empty(
        (batch, COUNT, 2 + len(local_times), 3), dtype=np.float32
    )
    output[:, :, :2] = prefix[:, None, :2]
    for row in range(batch):
        for candidate in range(COUNT):
            coefficient = quintic_coefficients(
                first[row], end[row, candidate], duration
            )
            positions, _, _ = evaluate_quintic(coefficient, local_times)
            output[row, candidate, 2:] = positions
    return output, first


def main():
    entry = json.loads(
        (ROOT / "reports/phase8jv2s0_entry_gate.json").read_text()
    )
    fixture_manifest = json.loads(FIXTURE_MANIFEST.read_text())
    checkpoint = next(
        row for row in json.loads(
            (ROOT / "reports/phase8jv2_entry_gate.json").read_text()
        )["checkpoint_audit"]
        if row["name"] == "fixed_050_seed8403"
    )
    config = SafetyEvaluatorV2Config.load(
        ROOT / "configs/safety_evaluator_v2.yaml",
        ROOT / "reports/phase8jq_controller_authoritative_envelope.json",
    )
    candidate_parameterization_hash = canonical_hash({
        name: sha256(ROOT / name) for name in (
            "policy/dep_network.py", "policy/models/head.py",
            "policy/state_transform.py", "policy/primitive.py",
            "loss/trajectory_sampler.py",
        )
    })
    key_fields = {
        "schema": "phase8jv2s0_paired_static_c0_v1",
        "static_fixture_manifest_hash": sha256(FIXTURE_MANIFEST),
        "fixture_loader_implementation_hash":
            sha256(ROOT / "policy/static_v2_fixture.py"),
        "traj_opt_hash": sha256(ROOT / "config/traj_opt.yaml"),
        "evaluator_hash": sha256(ROOT / "policy/safety_evaluator_v2.py"),
        "static_continuous_checker_hash":
            sha256(ROOT / "tools/evaluate_phase8jqv2_static.py"),
        "static_map_catalog_hash":
            sha256(ROOT / "configs/static_map_catalog.yaml"),
        "checkpoint_hash": checkpoint["observed_sha256"],
        "candidate_parameterization_hash": candidate_parameterization_hash,
        "timeline_hash": entry["source_hashes"][
            "policy/safety_evaluator_v2.py"
        ],
    }
    analysis_key = canonical_hash(key_fields)
    if ARTIFACT.is_file():
        cached = np.load(ARTIFACT, allow_pickle=False)
        if str(cached["analysis_key"]) == analysis_key:
            print(json.dumps({
                "status": "PASS", "cache_hit": True,
                "artifact": str(ARTIFACT),
            }, indent=2))
            return

    runtime = make_runtime(
        ROOT / "data/phase8_dynamic_production/map_catalog.yaml",
        ROOT / "configs/static_map_catalog.yaml",
    )
    load_dep_checkpoint(runtime.policy, checkpoint["path"], "corrected")
    runtime.policy.eval()
    dataset = StaticV2FixtureDataset(FIXTURE, cache_size=128)
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=False,
    )
    columns = {
        name: [] for name in (
            "start", "end", "predicted", "map_id", "legacy_clearance",
            "t0_raw", "first_raw",
        )
    }
    started = time.perf_counter()
    with torch.inference_mode():
        for ordinal, batch in enumerate(loader, 1):
            details = runtime._forward_details(
                *batch, dynamic_context=None, dynamic_obstacles=None,
                dep_loss=runtime.static_dep_loss,
            )
            size = len(batch[0])
            start = details["start_state_world_expanded"].reshape(
                size, COUNT, 3, 3
            )[:, 0].permute(0, 2, 1).detach().cpu().numpy()
            end = details["end_state_world_expanded"].reshape(
                size, COUNT, 3, 3
            ).permute(0, 1, 3, 2).detach().cpu().numpy()
            timelines, first = timelines_from_states(start, end, config)
            maps = details["batch_map_id"]
            positions = torch.from_numpy(
                timelines.reshape(size, -1, 3)
            ).to(runtime.device)
            _, raw = runtime.static_dep_loss.safety_loss.get_distance_cost(
                positions, maps
            )
            raw = raw.reshape(size, COUNT, -1).detach().cpu().numpy()
            segments = np.linalg.norm(
                np.diff(timelines, axis=2), axis=3
            )
            lower = np.minimum(raw[:, :, :-1], raw[:, :, 1:]) - segments
            clearance = np.minimum(raw.min(2), lower.min(2)) - 0.3
            columns["start"].append(start.astype(np.float32))
            columns["end"].append(end.astype(np.float32))
            columns["predicted"].append(
                details["predicted_score"].reshape(
                    size, COUNT
                ).detach().cpu().numpy().astype(np.float32)
            )
            columns["map_id"].append(maps.detach().cpu().numpy())
            columns["legacy_clearance"].append(clearance.astype(np.float32))
            columns["t0_raw"].append(raw[:, 0, 0].astype(np.float32))
            first_positions = torch.from_numpy(
                first[:, :, 0][:, None].astype(np.float32)
            ).to(runtime.device)
            _, first_raw = (
                runtime.static_dep_loss.safety_loss.get_distance_cost(
                    first_positions, maps
                )
            )
            columns["first_raw"].append(
                first_raw[:, 0].detach().cpu().numpy().astype(np.float32)
            )
            if ordinal % 50 == 0:
                print(
                    f"paired static C0 {min(ordinal*32, len(dataset))}/10000",
                    flush=True,
                )
    arrays = {name: np.concatenate(value) for name, value in columns.items()}
    arrays.update({
        "fixture_id": dataset.arrays["fixture_id"],
        "fixture_semantic_hash": np.asarray(dataset.semantic_hash),
        "analysis_key": np.asarray(analysis_key),
        "analysis_key_fields": np.asarray(json.dumps(key_fields, sort_keys=True)),
        "checkpoint_sha256": np.asarray(checkpoint["observed_sha256"]),
    })
    temporary = ARTIFACT.with_name(
        f".{ARTIFACT.name}.{os.getpid()}.tmp.npz"
    )
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, ARTIFACT)

    speed = np.linalg.norm(arrays["start"][:, :, 1], axis=1)
    acceleration = np.linalg.norm(arrays["start"][:, :, 2], axis=1)
    first_speed = np.linalg.norm(
        arrays["start"][:, :, 1]
        + config.latency_s * arrays["start"][:, :, 2], axis=1
    )
    first_acceleration = acceleration.copy()
    t0 = arrays["t0_raw"] - config.uav_radius_m
    first_clearance = arrays["first_raw"] - config.uav_radius_m
    outside = (speed > 6.0) | (acceleration > 6.0)
    recoverable = outside & (first_speed <= 6.0) & (first_acceleration <= 6.0)
    non_deepening = outside & (first_clearance >= t0 - TOLERANCE)
    report = {
        "status": "PASS",
        "fixture_semantic_hash": dataset.semantic_hash,
        "artifact": str(ARTIFACT.resolve()),
        "artifact_sha256": sha256(ARTIFACT),
        "analysis_key": analysis_key,
        "analysis_key_fields": key_fields,
        "window_count": len(dataset),
        "observed_initial_state": {
            "velocity_norm_mps": distribution(speed),
            "acceleration_norm_mps2": distribution(acceleration),
            "velocity_gt_6": metric(speed > 6.0, "all paired fixtures"),
            "acceleration_gt_6": metric(
                acceleration > 6.0, "all paired fixtures"
            ),
            "either_gt_6": metric(outside, "all paired fixtures"),
        },
        "first_controllable_state": {
            "time_s": config.latency_s,
            "velocity_norm_mps": distribution(first_speed),
            "acceleration_norm_mps2": distribution(first_acceleration),
            "t0_static_clearance_m": distribution(t0),
            "first_controllable_static_clearance_m":
                distribution(first_clearance),
        },
        "classification": {
            "initially_outside_nominal_envelope": metric(
                outside, "all paired fixtures"
            ),
            "recoverable_to_envelope_at_first_controllable": metric(
                recoverable, "all paired fixtures"
            ),
            "non_deepening_static_clearance": metric(
                non_deepening, "initially outside nominal envelope"
            ),
            "control_infeasible": None,
        },
        "bound_semantics": {
            "observed_initial_state": "measured/sampled input; not a command",
            "command_control_bound": {"velocity_mps": 6.0, "acceleration_mps2": 6.0},
            "terminal_state_bound": {"velocity_mps": 6.0, "acceleration_mps2": 6.0},
            "network_normalization_scale": {"velocity": 6.0, "acceleration": 6.0},
            "audit_only_bound": (
                "initial state outside nominal does not invalidate every "
                "candidate; feasibility must be evaluated after actionability"
            ),
        },
        "legacy_current_paired_coverage_failure": metric(
            ~(arrays["legacy_clearance"] >= -TOLERANCE).any(1),
            "all 10,000 immutable paired fixtures",
        ),
        "performance": {
            "device": str(runtime.device),
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_cpu_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "network_weights_modified": False,
        "training_executed": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps({
        "status": "PASS",
        "artifact": str(ARTIFACT),
        "paired_failure": report[
            "legacy_current_paired_coverage_failure"
        ]["fraction"],
        "initial_outside": report["classification"][
            "initially_outside_nominal_envelope"
        ]["fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
