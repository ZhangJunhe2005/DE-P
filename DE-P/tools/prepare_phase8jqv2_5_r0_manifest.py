#!/usr/bin/env python3
"""Freeze the deterministic 10,000-frame authoritative R0 selection."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.loader_v1 import AuthoritativeFormalDataset


ROOT_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"
COUNT = 10_000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main():
    dataset = AuthoritativeFormalDataset(
        ROOT / "data/phase8_authoritative_v2",
        "valid",
        expected_root_hash=ROOT_HASH,
        verify_files=False,
    )
    if len(dataset) != 100_000:
        raise RuntimeError("R0 selection requires exactly 100,000 valid frames")
    selected = [5 + 10*index for index in range(COUNT)]
    selected_set = set(selected)
    rows = []
    global_offset = 0
    for _, sequence in dataset._sequences:
        base = (
            dataset.root / sequence["suite"] / "valid"
            / sequence["sequence_id"]
        )
        frames = [
            json.loads(line)
            for line in (base / "frames.jsonl").read_text().splitlines()
        ]
        for frame in frames:
            global_index = global_offset + int(frame["frame_index"])
            if global_index not in selected_set:
                continue
            action = frame["actionability"]
            actors = frame["actor_metadata"]
            rows.append({
                "global_index": global_index,
                "sequence_id": frame["sequence_id"],
                "frame_index": frame["frame_index"],
                "map_uuid": frame["map_uuid"],
                "suite": frame["suite"],
                "scenario": frame["scenario_class"],
                "goal_type": frame["goal_type"],
                "actionability_class": action["actionability_class"],
                "nominal": bool(action["nominal"]),
                "stress": bool(action["stress"]),
                "recoverable": bool(action["recoverable"]),
                "near_boundary": (
                    frame["scenario_class"] == "near_boundary_recovery_stress"
                ),
                "actor_count": len(actors),
                "occluded_actor_count": sum(
                    bool(actor.get("occluded", False)) for actor in actors
                ),
            })
        global_offset += int(sequence["frame_count"])
    if len(rows) != COUNT:
        raise RuntimeError(f"R0 selection size mismatch: {len(rows)}")
    breakdown = {
        key: dict(sorted(Counter(row[key] for row in rows).items()))
        for key in (
            "map_uuid", "suite", "scenario", "goal_type",
            "actionability_class",
        )
    }
    coverage = {
        "all_valid_maps": len(breakdown["map_uuid"]) == 12,
        "static_and_dynamic": set(breakdown["suite"]) == {"static", "dynamic"},
        "nominal": any(row["nominal"] for row in rows),
        "recovery": any(row["recoverable"] for row in rows),
        "near_boundary": any(row["near_boundary"] for row in rows),
        "no_target": "no_target" in breakdown["scenario"],
        "multi_target": "multi_target" in breakdown["scenario"],
        "occluded_but_tracked":
            "occluded_but_tracked" in breakdown["scenario"],
    }
    payload = {
        "schema": "phase8jqv2_5_r0_selection_v1",
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": ROOT_HASH,
        "selection": "global_indices_5_plus_10k_for_k_0_through_9999",
        "frame_count": len(rows),
        "coverage": coverage,
        "breakdown": breakdown,
        "rows": rows,
        "production_test_used": False,
        "blind_used": False,
    }
    payload["selection_manifest_hash"] = canonical_hash(payload)
    status = "PASS" if all(coverage.values()) else "FAIL"
    payload["status"] = status
    output = ROOT / "artifacts/phase8jqv2_5/r0_selection_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    report = {
        key: value for key, value in payload.items() if key != "rows"
    }
    (ROOT / "reports/phase8jqv2_5_r0_selection_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
