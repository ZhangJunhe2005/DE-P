#!/usr/bin/env python3
"""Audit whether formal dynamic actors are observable as dynamic motion."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ROOT_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
ENTER_SPEED = 0.3


def main():
    root = ROOT / "data/phase8_authoritative_v2"
    manifest = json.loads((root / "manifests/dataset_manifest.json").read_text())
    velocities = []
    scenario_speeds = {}
    sequences = 0
    actor_sequences = 0
    zero_motion_sequences = 0
    for record in manifest["sequences"]:
        sequence = json.loads((root / record["path"]).read_text())
        if sequence["suite"] != "dynamic":
            continue
        sequences += 1
        base = (
            root / "dynamic" / sequence["split"] / sequence["sequence_id"]
            / "frames.jsonl"
        )
        with base.open() as stream:
            frame = json.loads(next(stream))
        actors = frame["actor_metadata"]
        if not actors:
            continue
        actor_sequences += 1
        values = [
            float(np.linalg.norm(actor["velocity_world"]))
            for actor in actors
        ]
        velocities.extend(values)
        scenario_speeds.setdefault(sequence["scenario"], []).extend(values)
        zero_motion_sequences += int(max(values) <= 1e-12)
    array = np.asarray(velocities, dtype=np.float64)
    speed_report = {
        scenario: {
            "actor_count": len(values),
            "minimum_mps": float(np.min(values)),
            "maximum_mps": float(np.max(values)),
            "mean_mps": float(np.mean(values)),
            "at_or_above_dynamic_enter_count":
                int(np.count_nonzero(np.asarray(values) >= ENTER_SPEED)),
        }
        for scenario, values in sorted(scenario_speeds.items())
    }
    selection = json.loads(
        (ROOT / "artifacts/phase8jqv2_5/r0_selection_manifest.json").read_text()
    )
    cache = json.loads(
        (ROOT / "artifacts/phase8jqv2_5/estimated_r0_cache/index.json").read_text()
    )
    selected_by_key = {
        (row["sequence_id"], row["frame_index"]): row
        for row in selection["rows"]
    }
    no_target_false = 0
    estimated_tracks = 0
    attention_frames = 0
    for entry in cache["entries"]:
        row = selected_by_key[(entry["sequence_id"], entry["frame_index"])]
        estimated_tracks += int(entry["track_count"])
        attention_frames += int(entry["attention_nonzero"] > 0)
        if row["scenario"] == "no_target" and (
            entry["track_count"] or entry["attention_nonzero"]
        ):
            no_target_false += 1
    checks = {
        "actor_motion_above_frozen_dynamic_threshold":
            bool(np.any(array >= ENTER_SPEED)),
        "nonzero_motion_in_every_actor_scenario":
            all(max(values) > 1e-12 for values in scenario_speeds.values()),
        "estimated_dynamic_recall_nonzero": estimated_tracks > 0,
        "no_target_false_attention_zero": no_target_false == 0,
    }
    report = {
        "status": "FAIL" if not all(checks.values()) else "PASS",
        "primary_cause": "authoritative_dynamic_motion_contract",
        "dataset_version": manifest["dataset_version"],
        "root_manifest_hash": ROOT_HASH,
        "dynamic_sequence_count": sequences,
        "actor_sequence_count": actor_sequences,
        "actor_count": len(array),
        "zero_motion_actor_sequence_count": zero_motion_sequences,
        "frozen_perception_dynamic_enter_speed_mps": ENTER_SPEED,
        "speed_distribution_mps": {
            "minimum": float(array.min()),
            "q10": float(np.quantile(array, .1)),
            "median": float(np.median(array)),
            "q90": float(np.quantile(array, .9)),
            "maximum": float(array.max()),
            "at_or_above_dynamic_enter_count":
                int(np.count_nonzero(array >= ENTER_SPEED)),
            "at_or_above_dynamic_enter_fraction":
                float(np.mean(array >= ENTER_SPEED)),
        },
        "scenario_breakdown": speed_report,
        "r0_estimated_cache": {
            "entry_count": cache["entry_count"],
            "dynamic_track_total": estimated_tracks,
            "nonzero_attention_frame_count": attention_frames,
            "no_target_false_nonzero_count": no_target_false,
            "camera_semantics_validated": False,
            "usable_for_rebaseline": False,
        },
        "checks": checks,
        "impact": [
            "estimated dynamic attention has no motion signal matching the frozen detector",
            "estimated-context surrogate/exact alignment cannot be established",
            "candidate capacity on valid_estimated would be non-authoritative",
        ],
        "required_repair": [
            "version a corrected dynamic actor motion contract",
            "use scenario motion above the frozen dynamic-enter threshold",
            "ensure multi-target and occluded-but-tracked actors actually move",
            "regenerate dynamic train/valid sequences under a new dataset root hash",
            "rerun causal estimated-context and Q2.5 gates",
        ],
        "formal_dataset_modified": False,
        "production_test_used": False,
        "blind_used": False,
        "optimizer_step_executed": False,
    }
    output = ROOT / "reports/phase8jqv2_5_dynamic_motion_contract.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
