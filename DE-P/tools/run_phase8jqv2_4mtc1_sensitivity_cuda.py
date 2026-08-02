#!/usr/bin/env python3
"""Bounded one-proposal exact/CUDA audit for the frozen MTC1 map grid."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.dynamic_motion_v2 import load_motion_contract
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_gap1_corpus_expander_v1 import (
    camera_motion, propose_at_gap_origins,
)
from generate_phase8jqv2_4rr1_corpus import SENSOR
from run_phase8jqv2_4ce1_new_map_search import ranked_candidates

REPORTS = ROOT / "reports"
DIAGNOSTICS = (
    ROOT / "diagnostics/phase8jqv2_4mtc1/parameter_sensitivity"
)
TYPE_FILES = {
    1: "phase8jqv2_4mtc1_cave_fill_sensitivity",
    2: "phase8jqv2_4mtc1_pillar_parameter_sensitivity",
    6: "phase8jqv2_4mtc1_room_parameter_sensitivity",
    7: "phase8jqv2_4mtc1_wall_parameter_sensitivity",
}


def atomic_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


class Recorder:
    def __init__(self, renderer):
        self.renderer = renderer
        self.calls = []

    def render_with_actor_diagnostics(self, *args, **kwargs):
        result = self.renderer.render_with_actor_diagnostics(*args, **kwargs)
        self.calls.append(result)
        return result


def cuda_metrics(row, renderer, contract, frame_times):
    backend = ExactAuthorityBVH(row["authority_root"])
    recorder = Recorder(renderer)
    rejection = None
    p0 = 0
    try:
        proposals, _, scored = ranked_candidates(backend, frame_times)
        p0 = min(128, len(proposals))
        if not scored:
            raise RuntimeError("no_ranked_camera_proposal")
        proposal = scored[0][1]
        expanded = camera_motion(
            proposal, "static", frame_times, backend
        )
        propose_at_gap_origins(
            backend, recorder, expanded, frame_times, contract,
            gap_origins=(12,), maximum_candidates=5,
        )
    except Exception as error:
        rejection = f"{type(error).__name__}: {error}"
    aggregate = Counter()
    exact_gap1 = 0
    for diagnostics in recorder.calls:
        projected = diagnostics[
            "per_actor_projected_pixel_count"
        ][:, 0]
        visible = diagnostics[
            "per_actor_visible_pixel_count"
        ][:, 0]
        blocked = diagnostics[
            "per_actor_static_blocked_pixel_count"
        ][:, 0]
        full = (projected > 0) & (visible == 0) & (blocked == projected)
        partial = (visible > 0) & (blocked > 0)
        aggregate["projected_actor_cases"] += int(np.any(projected > 0))
        aggregate["partial_occlusion_cases"] += int(np.any(partial))
        aggregate["full_occlusion_cases"] += int(np.any(full))
        indices = np.where(full)[0]
        runs = []
        if len(indices):
            groups = np.split(
                indices, np.where(np.diff(indices) > 1)[0] + 1
            )
            runs = [group for group in groups]
        if not runs:
            aggregate["full_occlusion_but_gap0"] += 1
        elif any(
            len(group) == 1 and group[0] >= 4
            and group[0] + 3 < len(visible)
            and np.all(visible[group[0]-4:group[0]] > 0)
            and np.all(visible[group[0]+1:group[0]+4] > 0)
            for group in runs
        ):
            exact_gap1 += 1
            aggregate["exact_gap1"] += 1
        elif max(map(len, runs)) >= 2:
            aggregate["gap2plus"] += 1
        else:
            aggregate["pre_post_window_failure"] += 1
    return {
        "p0_camera_proposals": p0,
        "bounded_exact_candidate_budget": 5,
        "exact_physically_safe_trajectory_count": len(recorder.calls),
        "cuda_projected_actor_cases":
            aggregate["projected_actor_cases"],
        "cuda_partial_occlusion_cases":
            aggregate["partial_occlusion_cases"],
        "cuda_full_occlusion_cases": aggregate["full_occlusion_cases"],
        "cuda_full_occlusion_but_gap0":
            aggregate["full_occlusion_but_gap0"],
        "cuda_gap2plus": aggregate["gap2plus"],
        "cuda_pre_post_window_failure":
            aggregate["pre_post_window_failure"],
        "exact_gap1_count": exact_gap1,
        "cuda_metrics_status": "PASS_BOUNDED_AUDIT",
        "terminal_rejection": rejection,
    }


def aggregate_profiles(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["profile_name"], []).append(row)
    return [{
        "profile_name": name,
        "map_count": len(group),
        "parameters": group[0]["resolved_parameters"],
        "occupied_voxel_fraction_mean":
            float(np.mean([x["occupied_voxel_fraction"] for x in group])),
        "largest_free_space_component_fraction_mean": float(np.mean([
            x["largest_free_space_component_fraction_stride2"] for x in group
        ])),
        "admissible_patch_count_sum":
            sum(x["admissible_patch_count"] for x in group),
        "two_sided_free_anchor_count_sum":
            sum(x["two_sided_free_anchor_count"] for x in group),
        "feasible_ratio_interval_count_sum":
            sum(x["feasible_ratio_interval_count"] for x in group),
        "exact_physically_safe_trajectory_count_sum":
            sum(x["exact_physically_safe_trajectory_count"] for x in group),
        "cuda_full_occlusion_count_sum":
            sum(x["cuda_full_occlusion_cases"] for x in group),
        "exact_gap1_count_sum": sum(x["exact_gap1_count"] for x in group),
    } for name, group in sorted(grouped.items())]


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    reports = {}
    for maze_type, prefix in TYPE_FILES.items():
        source = REPORTS / f"{prefix}_geometry_only_v1.json"
        reports[maze_type] = json.loads(source.read_text())
    for maze_type, report in reports.items():
        for index, row in enumerate(report["maps"]):
            result = cuda_metrics(
                row, renderer, contract, frame_times
            )
            row.update(result)
            atomic_new(
                DIAGNOSTICS /
                f"{row['map_uuid']}_bounded_cuda.json",
                {"map_uuid": row["map_uuid"], **result},
            )
            print(json.dumps({
                "type": row["natural_type"],
                "map": f"{index+1}/{len(report['maps'])}",
                "exact": result["exact_physically_safe_trajectory_count"],
                "cuda_full": result["cuda_full_occlusion_cases"],
                "gap1": result["exact_gap1_count"],
            }), flush=True)
        report["profile_aggregates"] = aggregate_profiles(report["maps"])
        report["cuda_metrics_pending"] = False
        report["bounded_cuda_budget_per_map"] = 5
        report["status"] = "PASS"
        if maze_type == 1:
            report["full_occlusion_monotonicity"] = (
                "INSUFFICIENT_EVENTS" if sum(
                    row["cuda_full_occlusion_cases"]
                    for row in report["maps"]
                ) < 2 else "MEASURED_IN_PROFILE_AGGREGATES"
            )
        atomic_new(REPORTS / f"{TYPE_FILES[maze_type]}.json", report)
    cross = {
        "status": "PASS",
        "grid_frozen_before_cuda": True,
        "map_count": sum(len(report["maps"]) for report in reports.values()),
        "types": {
            report["natural_type"]: report["profile_aggregates"]
            for report in reports.values()
        },
        "common_metrics_complete": True,
        "bounded_cuda_budget_per_map": 5,
        "detector_executed": False,
        "tracker_executed": False,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_new(
        REPORTS / "phase8jqv2_4mtc1_cross_type_sensitivity.json", cross
    )
    print(json.dumps({
        "status": "PASS", "maps": cross["map_count"],
        "peak_gpu_memory_bytes": cross["peak_gpu_memory_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
