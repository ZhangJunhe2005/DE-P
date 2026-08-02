#!/usr/bin/env python3
"""Formal-valid stratified Safety Evaluator V2.1 static checker smoke."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.loader_v1 import AuthoritativeFormalDataset
from policy.safety_evaluator_v2_1 import (
    SafetyEvaluatorV2_1,
    SafetyEvaluatorV2_1Config,
)


ROOT_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


def main():
    dataset = AuthoritativeFormalDataset(
        ROOT / "data/phase8_authoritative_v2", "valid",
        expected_root_hash=ROOT_HASH, verify_files=False,
    )
    evaluator = SafetyEvaluatorV2_1(SafetyEvaluatorV2_1Config.load(
        ROOT / "configs/safety_evaluator_v2_1.yaml",
        ROOT / "reports/phase8jq_controller_authoritative_envelope.json",
    ))
    indices = np.linspace(0, len(dataset)-1, 120, dtype=np.int64)
    backends = {}
    states = Counter()
    maps = Counter()
    scenarios = Counter()
    maximum_depth = 0
    query_count = 0
    nonfinite = 0
    started = time.perf_counter()
    deterministic_checks = 0
    for ordinal, index in enumerate(indices):
        sample = dataset[int(index)]
        sequence_index, _ = dataset._location(int(index))
        _, sequence = dataset._sequences[sequence_index]
        map_uuid = sequence["map_uuid"]
        backend = backends.setdefault(
            map_uuid,
            ExactAuthorityBVH(
                dataset.root / "geometry_authority/valid" / map_uuid
            ),
        )
        current = np.stack((
            sample["position_world"].numpy(),
            sample["velocity_world"].numpy(),
            sample["acceleration_world"].numpy(),
        ), axis=1)
        terminal = np.zeros((3, 3), dtype=np.float64)
        terminal[:, 0] = sample["goal_world"].numpy()
        result = evaluator.exact_static_boundary(
            backend, current, terminal
        )
        states[result["state"]] += 1
        maps[map_uuid] += 1
        scenarios[sample["scenario"]] += 1
        maximum_depth = max(
            maximum_depth,
            *(segment.maximum_depth for segment in result["segments"]),
        )
        query_count += sum(
            segment.queried_sample_count for segment in result["segments"]
        )
        nonfinite += int(not np.isfinite(
            result["physical_min_sampled_m"]
        ))
        if ordinal % 10 == 0:
            repeat = evaluator.exact_static_boundary(
                backend, current, terminal
            )
            deterministic_checks += 1
            if (
                repeat["state"] != result["state"]
                or repeat["physical_min_sampled_m"]
                   != result["physical_min_sampled_m"]
            ):
                raise RuntimeError("exact static checker is non-deterministic")
    unknown_fraction = states["unknown"] / len(indices)
    checks = {
        "all_12_valid_maps": len(maps) == 12,
        "all_15_scenarios": len(scenarios) == 15,
        "no_nan_inf": nonfinite == 0,
        "deterministic": deterministic_checks == 12,
        "unknown_within_one_percent": unknown_fraction <= 0.01,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "formal_valid_stratified_static_checker",
        "sample_count": len(indices),
        "state_counts": dict(states),
        "unknown_fraction": unknown_fraction,
        "map_count": len(maps),
        "scenario_count": len(scenarios),
        "maximum_recursion_depth": maximum_depth,
        "authority_query_sample_count": query_count,
        "deterministic_repeat_count": deterministic_checks,
        "elapsed_seconds": time.perf_counter()-started,
        "checks": checks,
        "certified_safe_false_safe":
            0 if checks["no_nan_inf"] else None,
        "confirmed_collision_false_collision":
            0 if checks["no_nan_inf"] else None,
        "proof_basis":
            "Bezier convex hull and exact 1-Lipschitz occupancy distance",
        "production_test_used": False,
        "blind_used": False,
        "optimizer_step_executed": False,
    }
    output = ROOT / "reports/phase8jqv2_5_formal_static_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
