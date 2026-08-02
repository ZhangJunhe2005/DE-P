#!/usr/bin/env python3
"""Compare causal runtime provenance with the independent CCR1 reference."""

from __future__ import annotations

from collections import defaultdict
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
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2/provenance"
CONFIG = ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.perception_probe_v2 import _pose
from authoritative_dataset.warp_visibility_reference_v1 import classify_pair
from policy.dynamic.visibility_provenance_v1 import (
    compute_visibility_provenance,
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


def metric(runtime, reference):
    runtime = np.asarray(runtime, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    tp = int(np.sum(runtime & reference))
    fp = int(np.sum(runtime & ~reference))
    fn = int(np.sum(~runtime & reference))
    tn = int(np.sum(~runtime & ~reference))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def finish(row):
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    return {
        **row,
        "precision": tp/max(tp+fp, 1),
        "recall": tp/max(tp+fn, 1),
    }


def main():
    manifest = load(DATASET / "manifest.json")
    config = yaml.safe_load(CONFIG.read_text())
    threshold = config["runtime_provenance"]
    if manifest["manifest_hash"] != config["physical_control_manifest_hash"]:
        raise RuntimeError("control/config manifest mismatch")
    development_ids = manifest["control_ids"][:39]
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    per_control = []
    runtimes = []
    mappings = {
        "stable_overlap": ("stable_overlap", "stable_overlap"),
        "newly_visible": (
            "newly_visible_candidate", "newly_visible_from_image_fov",
        ),
        "newly_invalid": (
            "newly_invalid_candidate", "newly_invalid_to_image_fov",
        ),
        "disocclusion": (
            "disocclusion_candidate",
            "newly_visible_from_static_disocclusion",
        ),
    }
    for control_id in development_ids:
        control = load(DATASET / "controls" / control_id / "control.json")
        if control["role"] != "hard_negative":
            continue
        root = DATASET / "controls" / control_id
        depth = np.load(root / "static_depth.npy")
        positions = np.load(root / "camera_positions.npy")
        yaws = np.load(root / "camera_yaws.npy")
        timestamps = np.load(root / "timestamps.npy")
        control_totals = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        for index in range(1, len(depth)):
            started = time.perf_counter()
            runtime = compute_visibility_provenance(
                depth[index-1], depth[index],
                _pose(positions[index-1], yaws[index-1],
                      timestamps[index-1]),
                _pose(positions[index], yaws[index], timestamps[index]),
                manifest["sensor"]["intrinsics"],
                manifest["sensor"]["max_depth_m"],
                threshold["depth_consistency_tolerance_m"],
                manifest["sensor"].get("min_depth_m", 0.1),
            )
            runtimes.append((time.perf_counter()-started)*1000)
            reference = classify_pair(
                depth[index-1], depth[index],
                positions[index-1], yaws[index-1],
                positions[index], yaws[index],
                manifest["sensor"],
                threshold["depth_consistency_tolerance_m"],
            )
            for name, (runtime_key, reference_key) in mappings.items():
                values = metric(
                    getattr(runtime, runtime_key), reference[reference_key]
                )
                for key in values:
                    totals[name][key] += values[key]
                    control_totals[name][key] += values[key]
        per_control.append({
            "control_id": control_id,
            "all_depth_valid": bool(np.all(
                np.isfinite(depth)
                & (depth < manifest["sensor"]["max_depth_m"]-1e-6)
            )),
            "combined_motion": bool(
                np.any(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 0)
                and np.any(np.abs(np.diff(yaws)) > 0)
            ),
            "channels": {
                name: finish(dict(value))
                for name, value in control_totals.items()
            },
        })
    channels = {name: finish(dict(value)) for name, value in totals.items()}
    gates = {
        "stable_overlap": (
            channels["stable_overlap"]["precision"]
            >= threshold["minimum_stable_overlap_precision"]
            and channels["stable_overlap"]["recall"]
            >= threshold["minimum_stable_overlap_recall"]
        ),
        "newly_visible": (
            channels["newly_visible"]["precision"]
            >= threshold["minimum_fov_precision"]
            and channels["newly_visible"]["recall"]
            >= threshold["minimum_fov_recall"]
        ),
        "disocclusion": (
            channels["disocclusion"]["precision"]
            >= threshold["minimum_disocclusion_precision"]
            and channels["disocclusion"]["recall"]
            >= threshold["minimum_disocclusion_recall"]
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "status": status,
        "runtime_module":
            "policy/dynamic/visibility_provenance_v1.py",
        "reference_module":
            "authoritative_dataset/warp_visibility_reference_v1.py",
        "development_static_control_count": len(per_control),
        "channels": channels,
        "predeclared_thresholds": threshold,
        "gates": gates,
        "future_frames_used": 0,
        "actor_gt_runtime_input": False,
        "reference_masks_runtime_input": False,
        "authority_correspondence_runtime_input": False,
        "average_runtime_ms": float(np.mean(runtimes)),
        "p95_runtime_ms": float(np.percentile(runtimes, 95)),
    }
    write_new(
        REPORTS / "phase8jqv2_4dpar2_runtime_visibility_provenance.json",
        report,
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar2_provenance_reference_comparison.json",
        {
            "status": status,
            "controls": per_control,
            "aggregate_channels": channels,
            "runtime_and_reference_implementations_independent": True,
        },
    )
    combined = [
        row for row in per_control if row["combined_motion"]
    ]
    write_new(
        REPORTS / "phase8jqv2_4dpar2_combined_motion_validation.json",
        {
            "status": "PASS" if combined and status == "PASS" else "FAIL",
            "control_count": len(combined),
            "controls": combined,
            "yaw_nonzero": True,
            "translation_nonzero": True,
        },
    )
    write_new(DIAGNOSTICS / "comparison_summary.json", {
        "status": status, "controls": per_control,
    })
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
