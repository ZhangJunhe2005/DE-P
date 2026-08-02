#!/usr/bin/env python3
"""Development-only train/valid trace for Phase 8E perception errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import cfg
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.types import DynamicPerceptionConfig
from policy.dynamic_sequence_dataset import validate_dataset_splits
from tools.evaluate_dynamic_perception_v2 import visible
from tools.evaluate_phase8c_estimated_context import camera, load_yaml, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-scenario", type=int, default=1)
    args = parser.parse_args()
    root = args.dataset.resolve()
    _manifest, splits = validate_dataset_splits(root)
    config = DynamicPerceptionConfig.from_global_config()
    scenario_counts, traces = {}, []
    for sequence in splits[args.split]:
        directory = root / "sequences" / sequence
        metadata = load_yaml(directory / "metadata.yaml")
        scenario = str(metadata.get("scenario_type", "unknown"))
        if scenario_counts.get(scenario, 0) >= args.max_per_scenario:
            continue
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        model = camera(metadata)
        perception = DynamicPerception(config, (cfg["vertical_num"], cfg["horizon_num"]))
        for frame_index, frame in enumerate(frames):
            timestamp = float(frame["timestamp"])
            frame_pose = pose(frame, metadata)
            depth = np.load(directory / frame["depth_path"], allow_pickle=False)
            result = perception.update_depth(depth, frame_pose, timestamp, model)
            objects = json.loads((directory / frame["dynamic_objects_path"]).read_text())
            truth = [obj for obj in objects if visible(obj)]
            tracks = []
            for track in result.all_tracks:
                distances = [float(np.linalg.norm(
                    track.position_world - np.asarray(obj["position_world"], dtype=float)
                )) for obj in truth]
                tracks.append({
                    "track_id": track.track_id,
                    "dynamic": track.is_dynamic,
                    "reason": track.dynamic_reason,
                    "confidence": track.confidence,
                    "speed": float(np.linalg.norm(track.velocity_world)),
                    "missed": track.missed_count,
                    "hits": track.hit_count,
                    "nearest_visible_gt_m": min(distances) if distances else None,
                    "position_world": track.position_world.tolist(),
                    "velocity_world": track.velocity_world.tolist(),
                })
            if truth or tracks or result.diagnostics["foreground_point_count"]:
                traces.append({
                    "sequence": sequence, "scenario": scenario,
                    "frame": frame_index, "timestamp": timestamp,
                    "visible_gt_count": len(truth),
                    "foreground": result.diagnostics["foreground"],
                    "cluster_count": result.diagnostics["cluster_count"],
                    "match_details": result.diagnostics["track_manager"]["match_details"],
                    "tracks": tracks,
                })
    payload = {
        "scope": "development_only_train_valid_no_runtime_gt",
        "split": args.split,
        "scenario_counts": scenario_counts,
        "config": {name: getattr(config, name) for name in config.__dataclass_fields__},
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trace_frames": len(traces)}))


if __name__ == "__main__":
    main()
