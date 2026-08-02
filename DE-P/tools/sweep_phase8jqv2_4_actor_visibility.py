#!/usr/bin/env python3
"""GPU canary sweep for every formal map and dynamic scenario."""

import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authoritative_dataset.cuda_renderer_v1 import (
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.generate_v1 import (
    build_actor_specs, ensure_map, load_config, safe_position, tasks_for,
)


def provenance():
    q23 = json.loads(
        (ROOT/"reports/phase8jqv2_3_final_result.json").read_text())
    return {key: q23[key] for key in (
        "Simulator_geometry_hash", "cache_index_hash", "checkpoint_hash",
        "config_hash", "dataset_manifest_hash", "evaluator_version",
        "geometry_hash", "timeline_hash", "uncertainty_policy_hash")}


def main():
    config = load_config(
        ROOT/"configs/phase8_authoritative_v1_generation.yaml")
    sensor = config["sensor_settings"]
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    frame_times = np.arange(
        config["frames_per_sequence"], dtype=np.float64
    )*sensor["frame_period_ns"]/1e9
    formal = ROOT/"data/phase8_authoritative_v1"
    records, failures = [], []
    with tempfile.TemporaryDirectory(
        prefix="phase8jqv2_4_visibility_maps_", dir="/tmp"
    ) as temporary:
        temporary = Path(temporary)
        for split in ("train", "valid"):
            maps, tasks = tasks_for(config, split, False)
            selected = {}
            for task in tasks:
                if task["suite"] != "dynamic" \
                        or task["scenario"] == "no_target":
                    continue
                selected.setdefault(
                    (task["map_uuid"], task["scenario"]), task)
            map_paths = {}
            for row in maps:
                existing = (
                    formal/f"geometry_authority/{split}/{row['map_uuid']}")
                map_paths[row["map_uuid"]] = (
                    existing if existing.is_dir()
                    else ensure_map(
                        temporary, config, split, row, False))
            for (map_uuid, scenario), task in selected.items():
                backend = ExactAuthorityBVH(map_paths[map_uuid])
                rng = np.random.default_rng(task["seed"])
                base, _ = safe_position(backend, rng)
                rng.normal(size=3)  # goal-direction draw in generate_sequence
                positions = np.repeat(
                    base[None, :], len(frame_times), axis=0)
                passed = False
                for attempt in range(128):
                    yaw, actors = build_actor_specs(
                        backend, rng, scenario, base, frame_times)
                    actor_positions = np.empty(
                        (len(frame_times), len(actors), 3),
                        dtype=np.float64)
                    for actor_index, actor in enumerate(actors):
                        actor_positions[:, actor_index] = (
                            actor["start"][None, :]
                            + frame_times[:, None]*actor["velocity"])
                    _, _, pixels = renderer.render(
                        backend, positions,
                        np.full(len(frame_times), yaw),
                        actor_positions,
                        [actor["radius_m"] for actor in actors])
                    if np.any(pixels):
                        passed = True
                        records.append({
                            "split": split, "map_uuid": map_uuid,
                            "scenario": scenario,
                            "task": task["sequence_id"],
                            "attempts": attempt+1,
                            "actor_pixels": int(pixels.sum()),
                            "visible_frames": int(np.count_nonzero(pixels)),
                        })
                        break
                if not passed:
                    failures.append({
                        "split": split, "map_uuid": map_uuid,
                        "scenario": scenario,
                        "task": task["sequence_id"]})
    report = {
        **provenance(),
        "status": "PASS" if not failures else "FAIL",
        "renderer_version": RENDERER_VERSION,
        "map_count": 60, "dynamic_scenarios_per_map": 6,
        "canary_count": len(records)+len(failures),
        "passed": len(records), "failures": failures,
        "maximum_attempts_used":
            max((row["attempts"] for row in records), default=0),
        "minimum_actor_pixels":
            min((row["actor_pixels"] for row in records), default=0),
        "minimum_visible_frames":
            min((row["visible_frames"] for row in records), default=0),
        "records": records,
    }
    path = ROOT/"reports/phase8jqv2_4_visibility_sweep.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("records", "failures")
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
