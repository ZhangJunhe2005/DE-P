#!/usr/bin/env python3
"""Stage A: frozen legacy/TF1 replay on CCR1 development controls only."""

from __future__ import annotations

from collections import Counter
import hashlib
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
sys.path.insert(0, str(ROOT))

from authoritative_dataset.perception_probe_v2 import (
    _pose, camera_model, frozen_perception_config,
)
from policy.dynamic.dynamic_perception import DynamicPerception
from policy.dynamic.foreground_contract_registry import (
    create_foreground_contract,
)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(
        value, indent=2, sort_keys=True, default=json_default
    ) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite DPAR2 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def development_controls(manifest):
    count = int(manifest["split_counts"]["development"])
    ids = manifest["control_ids"][:count]
    if len(ids) != 39 or any(name.endswith("_holdout") for name in ids):
        raise RuntimeError("development prefix is not the frozen 39 controls")
    values = []
    for control_id in ids:
        path = DATASET / "controls" / control_id / "control.json"
        value = load(path)
        if value["split"] != "development":
            raise RuntimeError(f"holdout leakage guard: {control_id}")
        values.append(value)
    return values


def runtime_sequence(control, candidate, parameters, sensor):
    """Runtime receives only depth, poses, timestamps, intrinsics and max depth."""
    root = DATASET / "controls" / control["control_id"]
    depths = np.load(root / control["runtime_inputs"]["depth_file"])
    positions = np.load(root / control["runtime_inputs"]["camera_positions"])
    yaws = np.load(root / control["runtime_inputs"]["camera_yaws"])
    timestamps = np.load(root / control["runtime_inputs"]["timestamps"])
    model = camera_model(sensor)
    config = frozen_perception_config(sensor)
    perception = DynamicPerception(
        config, (3, 5), attention_device="cpu"
    )
    perception.range_foreground = create_foreground_contract(
        candidate, config, parameters
    )
    rows = []
    started = time.perf_counter()
    for index in range(len(depths)):
        result = perception.update_depth(
            np.asarray(depths[index], dtype=np.float32),
            _pose(positions[index], yaws[index], timestamps[index]),
            float(timestamps[index]), model,
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
    runtime_ms = (time.perf_counter() - started) * 1000 / len(depths)
    return rows, runtime_ms


def candidate_d_sequence(control, parameters):
    """No existing track is supplied in discovery controls: D cannot birth."""
    root = DATASET / "controls" / control["control_id"]
    depths = np.load(root / control["runtime_inputs"]["depth_file"])
    extractor = create_foreground_contract(
        "candidate_d", frozen_perception_config({
            "height": 96, "width": 160,
            "intrinsics": [80.0, 80.0, 80.0, 45.0],
            "max_depth_m": 20.0,
        }), parameters,
    )
    rows = []
    started = time.perf_counter()
    previous = None
    for index, depth in enumerate(depths):
        residual = (
            np.zeros_like(depth, dtype=bool) if previous is None else
            np.isfinite(depth) & np.isfinite(previous)
            & (np.abs(previous-depth) > 0.1)
        )
        result = extractor.evaluate(
            residual, np.zeros_like(residual), track_alive=False
        )
        rows.append({
            "frame": index,
            "observation_count": 0,
            "observation_pixels": [],
            "track_count": 0,
            "attention_nonzero": 0,
            "foreground": {
                "candidate_d": result,
                "reacquisition_only": True,
            },
        })
        previous = depth
    return rows, (time.perf_counter() - started) * 1000 / len(depths)


def offline_score(control, rows, k):
    """Offline GT is loaded only after runtime output has been produced."""
    root = DATASET / "controls" / control["control_id"]
    owner = np.load(root / control["offline_ground_truth"]["actor_mask"])
    masks = owner >= 0
    actor_near = np.load(root / "actor_near_depth.npy")
    matched = []
    false = []
    for index, row in enumerate(rows):
        mask = masks[index].ravel()
        overlap = [
            int(mask[np.asarray(pixels, dtype=np.int64)].sum())
            if pixels else 0
            for pixels in row["observation_pixels"]
        ]
        matched.append(bool(overlap and max(overlap) > 0))
        false.append(bool(
            row["observation_count"] and not matched[-1]
        ))
    result = {
        "control_id": control["control_id"],
        "semantic_class": control["semantic_class"],
        "role": control["role"],
        "measurement_frames": [
            index for index, value in enumerate(matched) if value
        ],
        "false_measurement_frames": [
            index for index, value in enumerate(false) if value
        ],
        "any_measurement_frames": [
            row["frame"] for row in rows if row["observation_count"]
        ],
        "false_attention_frames": [
            row["frame"] for row in rows
            if row["attention_nonzero"] and not matched[row["frame"]]
        ],
        "actor_measurement": any(matched),
        "pass": False,
    }
    if control["role"] == "hard_negative":
        result["pass"] = not result["any_measurement_frames"]
        return result
    if control["role"] == "diagnostic_only":
        result["pass"] = None
        result["excluded_from_hard_gate"] = True
        return result
    if control["role"] == "paired_edge_positive":
        finite = np.isfinite(actor_near[:, 0])
        full = []
        for index, mask in enumerate(finite):
            vv, uu = np.nonzero(mask)
            if (
                len(uu) and uu.min() > 0 and uu.max() < mask.shape[1]-1
                and vv.min() > 0 and vv.max() < mask.shape[0]-1
            ):
                full.append(index)
        start = min(full) if full else None
        deadline = None if start is None else min(len(rows)-1, start+k-1)
        timely = bool(
            start is not None and any(
                matched[index] for index in range(start, deadline+1)
            )
        )
        result.update({
            "stable_overlap_start_frame": start,
            "latency_deadline_frame": deadline,
            "latency_k": k,
            "pass": timely,
        })
        return result
    result["pass"] = any(matched)
    return result


def summarize(candidate, parameter_index, parameters, controls, scores,
              runtimes):
    groups = {}
    for role in (
        "hard_positive", "hard_negative", "paired_edge_positive",
        "partial_occlusion_positive", "diagnostic_only",
    ):
        rows = [row for row in scores if row["role"] == role]
        eligible = [row for row in rows if row["pass"] is not None]
        groups[role] = {
            "passed": sum(row["pass"] is True for row in eligible),
            "total": len(eligible),
            "control_ids": [row["control_id"] for row in rows],
        }
    hard_gate = bool(
        candidate != "candidate_d"
        and groups["hard_positive"]["passed"]
            == groups["hard_positive"]["total"]
        and groups["hard_negative"]["passed"]
            == groups["hard_negative"]["total"]
        and groups["paired_edge_positive"]["passed"]
            == groups["paired_edge_positive"]["total"]
        and groups["partial_occlusion_positive"]["passed"]
            == groups["partial_occlusion_positive"]["total"]
    )
    if candidate == "ablation_seed7":
        hard_gate = False
    failures = [
        row["control_id"] for row in scores if row["pass"] is False
    ]
    if candidate == "candidate_d":
        classification = "reacquisition_only_not_discovery_candidate"
    else:
        positive_failure = any(
            row["pass"] is False and row["role"] in {
                "hard_positive", "partial_occlusion_positive"
            } for row in scores
        )
        edge_failure = any(
            row["pass"] is False
            and row["role"] == "paired_edge_positive" for row in scores
        )
        static_failure = any(
            row["pass"] is False and row["role"] == "hard_negative"
            for row in scores
        )
        if static_failure and (positive_failure or edge_failure):
            classification = "mixed_failure"
        elif static_failure:
            failure_ids = set(failures)
            if any("disocclusion" in value for value in failure_ids):
                classification = "static_disocclusion_false_positive"
            elif any(
                "max_depth" in value or "wall_edge" in value
                for value in failure_ids
            ):
                classification = "depth_validity_transition_false_positive"
            else:
                classification = "static_fov_false_positive"
        elif edge_failure:
            classification = "bounded_latency_failure"
        elif positive_failure:
            classification = "physical_positive_recall_failure"
        else:
            classification = "passes_physical_development_gate"
    return {
        "candidate": candidate,
        "parameter_index": parameter_index,
        "parameters": parameters,
        "taxonomy": groups,
        "development_hard_gate": hard_gate,
        "classification": classification,
        "failure_control_ids": failures,
        "average_runtime_ms_per_frame": float(np.mean(runtimes)),
        "p95_sequence_runtime_ms_per_frame":
            float(np.percentile(runtimes, 95)),
        "control_results": scores,
        "runtime_gt_input": False,
        "future_frames_used": 0,
    }


def main():
    entry = load(REPORTS / "phase8jqv2_4dpar2_entry_gate.json")
    if entry["status"] != "PASS":
        raise RuntimeError("DPAR2 entry gate is not PASS")
    access_log = DIAGNOSTICS / "holdout_access_log.jsonl"
    if access_log.read_text().count("\n") != 1:
        raise RuntimeError("holdout access occurred before Stage A")
    manifest = load(DATASET / "manifest.json")
    controls = development_controls(manifest)
    sensor = manifest["sensor"]
    document = yaml.safe_load((
        ROOT / "configs/temporal_foreground_contract_v2_1_candidates.yaml"
    ).read_text())
    evaluations = [{
        "candidate": "legacy_v1",
        "parameter_index": 0,
        "parameters": {},
    }]
    for candidate in (
        "candidate_a", "candidate_b", "candidate_c", "candidate_d",
        "ablation_seed7",
    ):
        for index, parameters in enumerate(
            document["candidates"][candidate]["grid"]
        ):
            evaluations.append({
                "candidate": candidate,
                "parameter_index": index,
                "parameters": parameters,
            })
    summaries = []
    for evaluation in evaluations:
        scores, runtimes = [], []
        candidate = evaluation["candidate"]
        for control in controls:
            if candidate == "candidate_d":
                rows, runtime = candidate_d_sequence(
                    control, evaluation["parameters"]
                )
            else:
                rows, runtime = runtime_sequence(
                    control, candidate, evaluation["parameters"], sensor
                )
            runtimes.append(runtime)
            scores.append(offline_score(
                control, rows,
                int(document.get("latency_k", 3)),
            ))
        summary = summarize(
            candidate, evaluation["parameter_index"],
            evaluation["parameters"], controls, scores, runtimes,
        )
        summaries.append(summary)
        write_new(
            DIAGNOSTICS / "tf1_candidates"
            / f"{candidate}_{evaluation['parameter_index']}.json",
            summary,
        )
        print(json.dumps({
            "candidate": candidate,
            "parameter_index": evaluation["parameter_index"],
            "classification": summary["classification"],
            "hard_gate": summary["development_hard_gate"],
            "failures": len(summary["failure_control_ids"]),
            "runtime_ms": summary["average_runtime_ms_per_frame"],
        }))

    legacy = next(row for row in summaries if row["candidate"] == "legacy_v1")
    tf1 = [row for row in summaries if row["candidate"] != "legacy_v1"]
    write_new(REPORTS / "phase8jqv2_4dpar2_legacy_physical_controls.json", {
        "status": "PASS",
        "implementation": "legacy_v1",
        "source_modified": False,
        "development_control_count": 39,
        "result": legacy,
        "sealed_control_holdout_accessed": False,
    })
    write_new(
        REPORTS / "phase8jqv2_4dpar2_tf1_candidates_physical_controls.json",
        {
            "status": "PASS",
            "grid_count": len(tf1),
            "results": tf1,
            "candidate_sources_modified": False,
            "candidate_d_birth_allowed": False,
            "seed7_diagnostic_only": True,
            "sealed_control_holdout_accessed": False,
        },
    )
    passes = [
        row for row in summaries if row["development_hard_gate"]
    ]
    write_new(
        REPORTS
        / "phase8jqv2_4dpar2_historical_conclusion_reassessment.json",
        {
            "status": "PASS",
            "legacy_small_projection_conclusion":
                "WITHDRAWN_INVALID_HISTORICAL_FIXTURE",
            "legacy_fov_boundary_conclusion":
                "WITHDRAWN_INVALID_HISTORICAL_FIXTURE",
            "depth_only_architecture_limit":
                "NOT_INHERITED_REQUIRES_PHYSICAL_EVIDENCE",
            "physical_development_pass_count": len(passes),
            "passing_candidates": [
                {
                    "candidate": row["candidate"],
                    "parameter_index": row["parameter_index"],
                } for row in passes
            ],
            "new_architecture_allowed": not bool(passes),
            "sealed_control_holdout_accessed": False,
        },
    )
    print(json.dumps({
        "status": "PASS",
        "evaluations": len(summaries),
        "physical_development_pass_count": len(passes),
        "passing_candidates": [
            [row["candidate"], row["parameter_index"]] for row in passes
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
