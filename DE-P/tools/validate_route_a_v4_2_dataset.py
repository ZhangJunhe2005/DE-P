#!/usr/bin/env python3
"""Fail-closed V4.2 scene-height and derived-dataset validation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TYPES = {"cave", "pillar", "forest", "room", "wall"}
RAW_VERSION = "route_a_v4_2_raw_static_v1"
DERIVED_VERSION = "route_a_v4_2_static_yopo"
SPATIAL_VERSION = "route_a_scene_spatial_sampling_v1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "data/route_a_v4_2_raw_static"
    )
    parser.add_argument(
        "--derived", type=Path,
        default=ROOT / "data/route_a_v4_2_static_yopo",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/route_a_v4_2_dataset_validation.json",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raw, derived = args.raw.resolve(), args.derived.resolve()
    errors: list[str] = []

    raw_manifest_path = raw / "manifests/dataset_manifest.json"
    raw_manifest = load_json(raw_manifest_path)
    if raw_manifest.get("dataset_version") != RAW_VERSION:
        errors.append("raw dataset version mismatch")
    if raw_manifest.get("scene_spatial_sampling_contract_version") != SPATIAL_VERSION:
        errors.append("scene spatial contract version mismatch")
    spatial_hash = raw_manifest.get("scene_spatial_sampling_contract_hash")
    if not spatial_hash:
        errors.append("scene spatial contract hash missing")
    if not (raw / "generation_state/completion/FULL_GENERATION_COMPLETE").is_file():
        errors.append("raw FULL_GENERATION_COMPLETE missing")

    altitude = defaultdict(list)
    observability = defaultdict(list)
    map_ids = defaultdict(set)
    sequence_count = defaultdict(int)
    for manifest_path in sorted((raw / "manifests/sequences").glob("*.json")):
        row = load_json(manifest_path)
        split = row["split"]
        sequence_count[split] += 1
        map_ids[split].add(row["map_uuid"])
        if row.get("scene_spatial_sampling_contract_hash") != spatial_hash:
            errors.append(f"sequence spatial hash mismatch: {row['sequence_id']}")
            continue
        sequence_root = raw / row["suite"] / split / row["sequence_id"]
        diagnostics = load_json(sequence_root / "render_diagnostics.json")
        obs = diagnostics.get("scene_spatial_observability")
        if not obs or not obs.get("passed"):
            errors.append(f"observability gate missing/failed: {row['sequence_id']}")
            continue
        key = (split, obs["map_type"])
        observability[key].append(
            (float(obs["return_fraction"]), float(obs["near_obstacle_fraction"]))
        )
        for line in (sequence_root / "frames.jsonl").read_text().splitlines():
            frame = json.loads(line)
            spatial = frame.get("scene_spatial_sampling", {})
            if spatial.get("contract_hash") != spatial_hash:
                errors.append(
                    f"frame spatial hash mismatch: {row['sequence_id']}"
                )
                break
            altitude[key].append(float(spatial["relative_altitude"]))

    if not args.smoke:
        if sequence_count != {"train": 5000, "valid": 1000}:
            errors.append(f"formal sequence counts mismatch: {dict(sequence_count)}")
        if {key: len(value) for key, value in map_ids.items()} != {
            "train": 48, "valid": 12
        }:
            errors.append(
                "formal map counts mismatch: "
                f"{ {key: len(value) for key, value in map_ids.items()} }"
            )

    metrics = {}
    for split in ("train", "valid"):
        present = {kind for (part, kind) in altitude if part == split}
        if present != EXPECTED_TYPES:
            errors.append(f"{split} map-type coverage mismatch: {sorted(present)}")
        metrics[split] = {}
        for kind in sorted(EXPECTED_TYPES):
            values = np.asarray(altitude[(split, kind)], dtype=np.float64)
            obs = np.asarray(observability[(split, kind)], dtype=np.float64)
            if values.size == 0:
                continue
            metrics[split][kind] = {
                "frames": int(values.size),
                "relative_altitude_p10_p50_p90": [
                    float(v) for v in np.quantile(values, [0.1, 0.5, 0.9])
                ],
                "low_fraction_le_0_40": float(np.mean(values <= 0.40)),
                "high_fraction_gt_0_68": float(np.mean(values > 0.68)),
                "return_fraction_min": float(obs[:, 0].min()),
                "near_obstacle_fraction_min": float(obs[:, 1].min()),
            }
            if kind in {"forest", "pillar", "wall"}:
                if float(np.mean(values <= 0.40)) < 0.40:
                    errors.append(f"{split}/{kind}: insufficient low-altitude coverage")
                if float(np.mean(values > 0.68)) > 0.22:
                    errors.append(f"{split}/{kind}: high-altitude bias remains")

    derived_manifest_path = derived / "manifests/dataset_manifest.json"
    derived_manifest = load_json(derived_manifest_path)
    if derived_manifest.get("status") != "COMPLETE_FROZEN":
        errors.append("derived dataset is not COMPLETE_FROZEN")
    if derived_manifest.get("dataset_version") != DERIVED_VERSION:
        errors.append("derived dataset version mismatch")
    if derived_manifest.get("source_dataset_manifest_hash") != sha256(raw_manifest_path):
        errors.append("derived/raw manifest identity mismatch")
    if set(derived_manifest.get("map_type_sample_counts", {}).get("train", {})) != EXPECTED_TYPES:
        errors.append("derived train map-type coverage mismatch")
    if set(derived_manifest.get("map_type_sample_counts", {}).get("validation", {})) != EXPECTED_TYPES:
        errors.append("derived validation map-type coverage mismatch")
    if derived_manifest.get("actor_input_used") is not False:
        errors.append("static dataset unexpectedly uses actor input")
    if derived_manifest.get("composed_depth_used") is not False:
        errors.append("static dataset unexpectedly uses composed depth")
    if not args.smoke:
        split_counts = derived_manifest.get("split_counts", {})
        if not (
            0 < int(split_counts.get("train", 0)) <= 300000
            and 0 < int(split_counts.get("validation", 0)) <= 60000
        ):
            errors.append(f"formal derived split counts invalid: {split_counts}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "raw": str(raw),
        "derived": str(derived),
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "derived_manifest_sha256": sha256(derived_manifest_path),
        "scene_spatial_sampling_contract_hash": spatial_hash,
        "sequences": dict(sequence_count),
        "maps": {key: len(value) for key, value in map_ids.items()},
        "height_and_observability": metrics,
        "errors": errors,
        "training_started": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
