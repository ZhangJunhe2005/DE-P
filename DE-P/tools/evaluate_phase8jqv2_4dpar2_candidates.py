#!/usr/bin/env python3
"""Evaluate four bounded DPAR2 candidates on development controls only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2"
CONFIG = ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.dynamic_perception_architecture_registry import (
    create_architecture,
)
from tools.evaluate_phase8jqv2_4dpar2_legacy import (
    development_controls, offline_score, summarize,
)


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite DPAR2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def runtime_sequence(control, key, parameters, sensor):
    root = DATASET / "controls" / control["control_id"]
    # This is the entire runtime adapter. Offline labels are not accepted.
    depths = np.load(root / control["runtime_inputs"]["depth_file"])
    positions = np.load(root / control["runtime_inputs"]["camera_positions"])
    yaws = np.load(root / control["runtime_inputs"]["camera_yaws"])
    timestamps = np.load(root / control["runtime_inputs"]["timestamps"])
    model = camera_model(sensor)
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    perception.range_foreground = create_architecture(
        key, config, parameters
    )
    rows = []
    started = time.perf_counter()
    for index, timestamp in enumerate(timestamps):
        result = perception.update_depth(
            np.asarray(depths[index], dtype=np.float32),
            _pose(positions[index], yaws[index], timestamp),
            float(timestamp), model,
        )
        rows.append({
            "frame": index,
            "observation_count": len(result.observations),
            "observation_pixels": [
                list(observation.pixel_indices)
                for observation in result.observations
            ],
            "track_count": len(result.all_tracks),
            "attention_nonzero":
                int(result.attention_map.count_nonzero().item()),
            "foreground": result.diagnostics["foreground"],
        })
    return rows, (time.perf_counter()-started)*1000/len(depths)


def main():
    if (
        DIAGNOSTICS / "holdout_access_log.jsonl"
    ).read_text().count("\n") != 1:
        raise RuntimeError("holdout access occurred before candidate freeze")
    manifest = load(DATASET / "manifest.json")
    controls = development_controls(manifest)
    document = yaml.safe_load(CONFIG.read_text())
    keys = [
        "physical_control_residual_v1",
        "visibility_aware_causal_residual_v1",
        "causal_depth_track_before_detect_v1",
        "dual_path_dynamic_perception_v1",
    ]
    summaries = []
    for index, key in enumerate(keys):
        parameters = document["candidates"][key]
        scores, runtimes = [], []
        traces = []
        for control in controls:
            rows, runtime = runtime_sequence(
                control, key, parameters, manifest["sensor"]
            )
            score = offline_score(control, rows, 3)
            scores.append(score)
            runtimes.append(runtime)
            traces.append({
                "control_id": control["control_id"],
                "frames": [{
                    "frame": row["frame"],
                    "observation_count": row["observation_count"],
                    "track_count": row["track_count"],
                    "attention_nonzero": row["attention_nonzero"],
                    "foreground": row["foreground"],
                } for row in rows],
            })
        summary = summarize(
            key, 0, parameters, controls, scores, runtimes
        )
        summary["candidate_index"] = index
        summaries.append(summary)
        write_new(
            REPORTS / f"phase8jqv2_4dpar2_candidate{index}_development.json",
            {"status": "PASS", "result": summary},
        )
        write_new(
            DIAGNOSTICS / "physical_controls" / f"candidate{index}_trace.json",
            {"candidate": key, "traces": traces},
        )
        print(json.dumps({
            "candidate": key,
            "hard_gate": summary["development_hard_gate"],
            "classification": summary["classification"],
            "failures": summary["failure_control_ids"],
            "runtime_ms": summary["average_runtime_ms_per_frame"],
        }))
    passing = [row for row in summaries if row["development_hard_gate"]]
    write_new(
        REPORTS / "phase8jqv2_4dpar2_candidate_comparison.json",
        {
            "status": "PASS",
            "candidate_count": 4,
            "results": [{
                "candidate": row["candidate"],
                "hard_gate": row["development_hard_gate"],
                "classification": row["classification"],
                "failure_control_ids": row["failure_control_ids"],
                "average_runtime_ms_per_frame":
                    row["average_runtime_ms_per_frame"],
            } for row in summaries],
            "passing_candidate_count": len(passing),
            "holdout_accessed": False,
        },
    )
    print(json.dumps({
        "status": "PASS",
        "passing_candidate_count": len(passing),
        "passing_candidates": [row["candidate"] for row in passing],
    }, indent=2))


if __name__ == "__main__":
    main()
