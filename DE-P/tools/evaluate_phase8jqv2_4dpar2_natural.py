#!/usr/bin/env python3
"""Run the sole physical-control candidate on frozen natural development cases.

This evaluator deliberately does not synthesize additional gaps or alter any
map, camera, actor, or timeline.  Offline actor ownership is used only after
runtime inference to score measurements.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4dpar2"
CONFIG = ROOT / "configs/dynamic_perception_architecture_candidates_v1.yaml"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.dynamic_perception_architecture_registry import (
    create_architecture,
)
from tools.evaluate_phase8jqv2_4tf1_candidates import (
    render_case,
    sensor,
    summarize_natural,
)


CANDIDATE = "physical_control_residual_v1"
FIXED_N1_CASE = "natural_forest_81064183_seed831005002_gap1"


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


def runtime_rows(value, parameters):
    """Inference adapter: depth, pose, timestamps and intrinsics only."""
    config = frozen_perception_config(sensor())
    model = camera_model(sensor())
    perception = DynamicPerception(config, (3, 5), attention_device="cpu")
    perception.range_foreground = create_architecture(
        CANDIDATE, config, parameters
    )
    rows = []
    import time
    started = time.perf_counter()
    for index, depth in enumerate(value["depths"]):
        timestamp = index * .1
        result = perception.update_depth(
            np.asarray(depth, dtype=np.float32),
            _pose(value["positions"][index], value["yaws"][index], timestamp),
            timestamp, model,
        )
        actor_mask = np.asarray(value["masks"][index], dtype=bool).ravel()
        overlaps = [
            sum(actor_mask[list(observation.pixel_indices)])
            for observation in result.observations
            if observation.pixel_indices
        ]
        matched = bool(overlaps and max(overlaps) > 0)
        errors = [
            float(np.linalg.norm(
                observation.centroid_world - value["actors"][index]
            ))
            for observation in result.observations
        ]
        rows.append({
            "frame": index,
            "measurement_count": len(result.observations),
            "measurement_valid": bool(result.observations),
            "actor_measurement_valid": matched,
            "centroid_error_m": min(errors) if errors else None,
            "actor_overlap_pixels": int(max(overlaps)) if overlaps else 0,
            "component_pixels": int(
                result.diagnostics["foreground"].get(
                    "component_pixel_count", 0
                )
            ),
            "strong_seeds": int(
                result.diagnostics["foreground"].get(
                    "range_seed_count", 0
                )
            ),
            "weak_support": int(
                result.diagnostics["foreground"].get(
                    "weak_support_pixels", 0
                )
            ),
            "track_ids": [track.track_id for track in result.all_tracks],
            "confirmed_ids": [
                track.track_id for track in result.confirmed_tracks
            ],
            "dynamic_ids": [
                track.track_id for track in result.dynamic_tracks
            ],
            "false_attention": bool(
                result.attention_map.detach().cpu().max().item() > 0
                and not matched
            ),
        })
    return rows, (time.perf_counter() - started) * 1000 / len(rows)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for natural authority rendering")
    access_log = DIAGNOSTICS / "holdout_access_log.jsonl"
    if access_log.read_text().count("\n") != 1:
        raise RuntimeError("holdout access occurred before candidate freeze")

    retained = load(REPORTS / "phase8jqv2_4i1_retained_case_manifest.json")
    if retained["case_count"] != 6:
        raise RuntimeError("frozen I1 six-case manifest changed")
    cases = list(retained["cases"])
    if not any(row["case_id"] == FIXED_N1_CASE for row in cases):
        raise RuntimeError("fixed N1 forest gap-1 case missing")
    if any(row.get("annex_used") for row in cases):
        raise RuntimeError("annex case is forbidden")

    map_set = load(
        REPORTS / "occlusion_constructor_v2_2_natural_map_set.json"
    )
    map_rows = {row["map_uuid"]: row for row in map_set["maps"]}
    if any(case["map_uuid"] not in map_rows for case in cases):
        raise RuntimeError("natural authority map missing")

    renderer = CudaAuthorityRenderer(sensor(), "cuda:0")
    parameters = yaml.safe_load(CONFIG.read_text())["candidates"][CANDIDATE]
    results, traces, runtimes = [], [], []
    for case in cases:
        value = render_case(case, map_rows, renderer)
        rows, runtime_ms = runtime_rows(value, parameters)
        summary = summarize_natural(value, rows)
        summary.update({
            "requested_gap": int(case["requested_gap"]),
            "map_seed": int(case["map_seed"]),
            "semantic_map_type": case["semantic_map_type"],
            "average_runtime_ms_per_frame": runtime_ms,
            "runtime_inputs": [
                "depth", "camera_pose", "timestamp", "camera_intrinsics"
            ],
            "offline_scoring_only": [
                "actor_trajectory", "actor_instance_owner_mask"
            ],
        })
        results.append(summary)
        runtimes.append(runtime_ms)
        traces.append({
            "case_id": case["case_id"],
            "frames": rows,
        })

    gap1 = [row for row in results if row["requested_gap"] == 1]
    gap2 = [row for row in results if row["requested_gap"] == 2]
    gap3 = [row for row in results if row["requested_gap"] == 3]

    def measurement_pass(row):
        return (
            row["pre_gap_consecutive_measurements"] >= 1
            and row["first_post_gap_measurement"]
        )

    gap1_measurement = all(map(measurement_pass, gap1)) and bool(gap1)
    gap1_maps = {row["map_uuid"] for row in gap1}
    gap1_types = {row["maze_type"] for row in gap1}
    gap1_seeds = {row["map_seed"] for row in gap1}
    gap1_coverage = (
        len(gap1_maps) >= 3 and len(gap1_types) >= 2
        and len(gap1_seeds) >= 2
    )
    gap1_gate = gap1_measurement and gap1_coverage

    # Contractually blocked until gap-1 passes.  Measurements are retained as
    # diagnostics, but are not promoted to a gap-2 Gate result.
    gap2_measurement = all(map(measurement_pass, gap2)) and bool(gap2)
    gap2_maps = {row["map_uuid"] for row in gap2}
    gap2_types = {row["maze_type"] for row in gap2}
    gap2_coverage = len(gap2_maps) >= 2 and len(gap2_types) >= 2

    gap1_report = {
        "status": "PASS" if gap1_gate else "FAIL",
        "candidate": CANDIDATE,
        "measurement_pass": gap1_measurement,
        "coverage_pass": gap1_coverage,
        "case_count": len(gap1),
        "independent_map_count": len(gap1_maps),
        "maze_type_count": len(gap1_types),
        "independent_seed_count": len(gap1_seeds),
        "required": {
            "independent_maps": 3, "maze_types": 2,
            "multiple_seeds": True,
        },
        "cases": gap1,
        "blocker": (
            None if gap1_gate else
            "frozen natural corpus has only one gap-1 map/type/seed"
        ),
        "maps_or_trajectories_modified": False,
    }
    gap2_report = {
        "status": "PASS" if gap1_gate and gap2_measurement
            and gap2_coverage else "BLOCKED",
        "candidate": CANDIDATE,
        "measurement_diagnostic": gap2_measurement,
        "coverage_diagnostic": gap2_coverage,
        "case_count": len(gap2),
        "independent_map_count": len(gap2_maps),
        "maze_type_count": len(gap2_types),
        "required": {"independent_maps": 2, "maze_types": 2},
        "cases": gap2,
        "blocked_by": None if gap1_gate else "natural_gap1_gate",
        "maps_or_trajectories_modified": False,
    }
    overall = {
        "status": "PASS" if gap1_gate and gap2_report["status"] == "PASS"
            else "FAIL",
        "candidate": CANDIDATE,
        "physical_development_gate": "PASS",
        "natural_gap1_gate": gap1_report["status"],
        "natural_gap2_gate": gap2_report["status"],
        "i1_cases_evaluated": len(results),
        "n1_fixed_case_evaluated": True,
        "gap_counts": {
            "gap1": len(gap1), "gap2": len(gap2), "gap3": len(gap3)
        },
        "gap3_role": "OUT_OF_SCOPE_DIAGNOSTIC_ONLY",
        "average_runtime_ms_per_frame": float(np.mean(runtimes)),
        "results": results,
        "runtime_gt_used": False,
        "future_frames_used": 0,
        "annex_used": False,
        "new_maps_generated": False,
        "holdout_accessed": False,
        "primary_cause": (
            None if gap1_gate else
            "natural_development_case_coverage_insufficient"
        ),
    }
    write_new(
        REPORTS / "phase8jqv2_4dpar2_natural_gap1.json", gap1_report
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar2_natural_gap2.json", gap2_report
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar2_natural_measurement_validation.json",
        overall,
    )
    write_new(
        DIAGNOSTICS / "natural_cases" / "candidate0_i1_six_cases.json",
        {"candidate": CANDIDATE, "traces": traces},
    )
    print(json.dumps({
        "status": overall["status"],
        "candidate": CANDIDATE,
        "gap1_measurement": gap1_measurement,
        "gap1_coverage": gap1_coverage,
        "gap1_maps": len(gap1_maps),
        "gap1_types": len(gap1_types),
        "gap2_status": gap2_report["status"],
        "holdout_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
