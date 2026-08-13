#!/usr/bin/env python3
"""Deterministic CPU latency smoke for the V4.9 16-actor safety path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.poly_solver import Poly5Solver
from policy.runtime_profile_v4_9 import runtime_safety_mapping_v4_9
from policy.runtime_safety_v1 import (
    CandidateSafetyV1,
    RuntimeSafetyConfigV1,
    RuntimeTrajectorySafetyV1,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=49)
    return parser.parse_args()


def build_fixture(seed):
    rng = np.random.default_rng(seed)
    mapping = runtime_safety_mapping_v4_9({})
    mapping.update({
        "feasibility_projection_enabled": False,
        "enforce_camera_visibility": False,
        "enforce_minimum_progress": False,
        "enforce_goal_progress": False,
        "enforce_observed_tracking_clearance": False,
        "enforce_stopping_distance": False,
        "enforce_preferred_flight_volume_clearance": False,
        "dynamic_track_max_age_s": 1.0,
    })
    shield = RuntimeTrajectorySafetyV1(
        RuntimeSafetyConfigV1.from_mapping(mapping)
    )
    duration = 3.333
    candidates = []
    for _ in range(15):
        endpoint = np.asarray((
            7.0 + rng.uniform(-1.0, 1.0),
            rng.uniform(-4.0, 4.0),
            rng.uniform(-2.0, 2.0),
        ))
        candidates.append(tuple(
            Poly5Solver(0.0, 0.0, 0.0, endpoint[axis], 0.0, 0.0,
                        duration)
            for axis in range(3)
        ))
    tracks = []
    for track_id in range(16):
        tracks.append(SimpleNamespace(
            is_dynamic=True,
            timestamp=10.0,
            last_direct_observation_timestamp=10.0,
            position_world=np.asarray((
                rng.uniform(1.0, 8.0),
                rng.uniform(-4.0, 4.0),
                rng.uniform(-2.0, 2.0),
            )),
            velocity_world=rng.uniform(-1.0, 1.0, 3),
            observed_extent=(0.6, 0.6, 1.0),
            track_id=track_id,
            state_covariance=np.eye(6) * 0.02,
        ))
    obstacle_points = rng.uniform(
        (-1.0, -8.0, -4.0), (12.0, 8.0, 4.0), size=(3600, 3),
    )
    # Retain realistic KD-tree load without deliberately creating static
    # rejection in every candidate.
    obstacle_points[:, 1] += np.where(
        obstacle_points[:, 1] >= 0.0, 5.0, -5.0,
    )
    return shield, tuple(candidates), duration, obstacle_points, tuple(tracks)


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "maximum_ms": float(np.max(values)),
    }


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup/iterations contract is invalid")
    shield, candidates, duration, points, tracks = build_fixture(args.seed)
    evaluate = lambda: shield.evaluate(
        candidates, tuple(duration for _ in candidates), points,
        np.zeros(3), np.eye(3), dynamic_tracks=tracks,
        query_timestamp=10.0, goal_world=np.asarray((30.0, 0.0, 0.0)),
    )
    single = measure(evaluate, args.warmup, args.iterations)
    dynamic_only = tuple(
        CandidateSafetyV1(
            False, ("predicted_dynamic_clearance",), 1.0, 1.0,
            None, 1.0,
        )
        for _ in candidates
    )

    # Conservative upper bound: base evaluation plus both ordered temporal
    # trials.  Runtime exits after 1.2 whenever it finds an executable action.
    def three_stage():
        evaluate()
        for scale in (1.2, 1.4):
            pool, durations, _, _ = shield.retime_dynamic_only_candidates(
                candidates, tuple(duration for _ in candidates),
                dynamic_only, scale,
            )
            shield.evaluate(
                pool, durations, points, np.zeros(3), np.eye(3),
                dynamic_tracks=tracks, query_timestamp=10.0,
                goal_world=np.asarray((30.0, 0.0, 0.0)),
            )

    worst = measure(three_stage, args.warmup, args.iterations)
    print(json.dumps({
        "status": "PASS",
        "role": "diagnostic_latency_smoke_not_a_gate",
        "candidate_count": 15,
        "trajectory_samples": 81,
        "depth_point_count": 3600,
        "dynamic_track_count": 16,
        "iterations": args.iterations,
        "single_evaluation": single,
        "conservative_three_evaluation_upper_bound": worst,
        "note": (
            "upper bound excludes network/depth/visualization and deliberately "
            "executes 1.4 even though runtime skips it when 1.2 succeeds"
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
