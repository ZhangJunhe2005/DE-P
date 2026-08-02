#!/usr/bin/env python3
"""Exhaustively audit formal dynamic actor construction without rendering."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.generate_v1 import (
    build_actor_specs, load_config, safe_position, tasks_for,
)
from authoritative_dataset.state_semantics_v2 import sample_uav_sequence


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_map(arguments: tuple) -> dict:
    config_path, split, map_uuid, tasks = arguments
    config = load_config(config_path)
    map_root = (
        Path(config["authority_source_root"])
        / f"geometry_authority/{split}/{map_uuid}"
    )
    backend = ExactAuthorityBVH(map_root)
    period = config["sensor_settings"]["frame_period_ns"] / 1e9
    fallback, failures = [], []
    for task in tasks:
        try:
            rng = np.random.default_rng(task["seed"])
            times = np.arange(task["frame_count"], dtype=np.float64) * period
            state = sample_uav_sequence(
                backend, rng, task["scenario"], times, safe_position,
                maximum_attempts=config["certificate_settings"][
                    "maximum_attempts_per_window"
                ],
            )
            actors = (
                build_actor_specs(
                    backend, rng, task["scenario"], state.position_world,
                    float(state.yaw[0]), times,
                )
                if task["suite"] == "dynamic"
                and task["scenario"] != "no_target"
                else []
            )
            if any(
                actor.get("sampling_method", "").endswith("fallback_v1")
                for actor in actors
            ):
                fallback.append({
                    "sequence_id": task["sequence_id"],
                    "scenario": task["scenario"],
                    "map_uuid": map_uuid,
                })
        except Exception as error:
            failures.append({
                "sequence_id": task["sequence_id"],
                "scenario": task["scenario"],
                "map_uuid": map_uuid,
                "error": repr(error),
            })
    return {
        "split": split,
        "map_uuid": map_uuid,
        "sequences": len(tasks),
        "fallback": fallback,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=ROOT / "configs/phase8_authoritative_v2_generation.yaml",
        type=Path,
    )
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument(
        "--output",
        default=ROOT / "reports/phase8jqv2_4_actor_sampling_audit.json",
        type=Path,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    work = []
    total_sequences = 0
    total_dynamic = 0
    for split in ("train", "valid"):
        _, tasks = tasks_for(config, split)
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task["map_uuid"]].append(task)
            total_sequences += 1
            if (
                task["suite"] == "dynamic"
                and task["scenario"] != "no_target"
            ):
                total_dynamic += 1
        work.extend(
            (str(args.config), split, map_uuid, rows)
            for map_uuid, rows in grouped.items()
        )
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(audit_map, row) for row in work]
        for future in as_completed(futures):
            results.append(future.result())
    failures = [
        failure for row in results for failure in row["failures"]
    ]
    fallback = [
        record for row in results for record in row["fallback"]
    ]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "scope": (
            "all formal train/valid UAV states and every actor-bearing "
            "dynamic sequence"
        ),
        "formal_sequences_checked": total_sequences,
        "dynamic_sequences_checked": total_dynamic,
        "maps_checked": len(results),
        "fallback_sequence_count": len(fallback),
        "fallback_sequences": sorted(
            fallback, key=lambda row: row["sequence_id"]
        ),
        "failures": sorted(
            failures, key=lambda row: row["sequence_id"]
        ),
        "rendering_checked": False,
        "production_test_used": False,
        "blind_used": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
