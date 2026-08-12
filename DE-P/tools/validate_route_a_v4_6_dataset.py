#!/usr/bin/env python3
"""Fail-closed validation for the four-scene Route-A V4.6 dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase8jqv2_4m1_mixed_maps import profile_hash, source_hash
from tools.prepare_route_a_v4_6_maps import load_profiles
from authoritative_dataset.generate_v1 import load_config, tasks_for

RAW = ROOT / "data/route_a_v4_6_raw_static"
DERIVED = ROOT / "data/route_a_v4_6_static_yopo"
REPORT = ROOT / "reports/route_a_v4_6_dataset_validation.json"
CONFIG = ROOT / "configs/route_a_v4_6_raw_static_resolved.yaml"
EXPECTED_TYPES = {"cave", "pillar", "forest", "wall"}
RAW_VERSION = "route_a_v4_6_raw_static_v1"
DERIVED_VERSION = "route_a_v4_6_static_yopo"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    errors = []
    raw_manifest_path = RAW / "manifests/dataset_manifest.json"
    derived_manifest_path = DERIVED / "manifests/dataset_manifest.json"
    raw_manifest = load_json(raw_manifest_path)
    derived_manifest = load_json(derived_manifest_path)
    if raw_manifest.get("dataset_version") != RAW_VERSION:
        errors.append("raw dataset version mismatch")
    if not (RAW / "generation_state/completion/FULL_GENERATION_COMPLETE").is_file():
        errors.append("raw FULL_GENERATION_COMPLETE missing")
    if raw_manifest.get("scene_spatial_sampling_contract_version") \
            != "route_a_scene_spatial_sampling_v1":
        errors.append("scene spatial contract version mismatch")
    spatial_hash = raw_manifest.get("scene_spatial_sampling_contract_hash")
    if not spatial_hash:
        errors.append("scene spatial contract hash missing")

    sequences = Counter()
    raw_frames = Counter()
    sequence_rows = defaultdict(dict)
    maps = defaultdict(set)
    sequence_types = defaultdict(Counter)
    altitude = defaultdict(list)
    observability = defaultdict(list)
    for manifest_path in sorted((RAW / "manifests/sequences").glob("*.json")):
        row = load_json(manifest_path)
        split = row["split"]
        sequence_id = str(row["sequence_id"])
        if sequence_id in sequence_rows[split]:
            errors.append(f"duplicate sequence manifest: {sequence_id}")
        sequence_rows[split][sequence_id] = row
        sequences[split] += 1
        frame_count = int(row.get("frame_count", -1))
        raw_frames[split] += frame_count
        if frame_count != 60:
            errors.append(
                f"unexpected frame count: {row['sequence_id']}={frame_count}"
            )
        maps[split].add(row["map_uuid"])
        if row.get("suite") != "static":
            errors.append(f"non-static sequence: {row['sequence_id']}")
        if row.get("scene_spatial_sampling_contract_hash") != spatial_hash:
            errors.append(f"sequence spatial hash mismatch: {row['sequence_id']}")
            continue
        root = RAW / f"static/{split}/{row['sequence_id']}"
        diagnostics = load_json(root / "render_diagnostics.json")
        observed = diagnostics.get("scene_spatial_observability", {})
        map_type = observed.get("map_type")
        if map_type not in EXPECTED_TYPES or not observed.get("passed"):
            errors.append(f"invalid observability: {row['sequence_id']}")
            continue
        sequence_types[split][map_type] += 1
        observability[(split, map_type)].append((
            float(observed["return_fraction"]),
            float(observed["near_obstacle_fraction"]),
        ))
        for line in (root / "frames.jsonl").read_text().splitlines():
            frame = json.loads(line)
            spatial = frame.get("scene_spatial_sampling", {})
            if spatial.get("contract_hash") != spatial_hash:
                errors.append(f"frame spatial hash mismatch: {row['sequence_id']}")
                break
            altitude[(split, map_type)].append(
                float(spatial["relative_altitude"])
            )

    if dict(sequences) != {"train": 5000, "valid": 1000}:
        errors.append(f"sequence counts mismatch: {dict(sequences)}")
    if dict(raw_frames) != {"train": 300000, "valid": 60000}:
        errors.append(f"raw frame totals mismatch: {dict(raw_frames)}")
    generation_config = load_config(CONFIG)
    planned_sequence_types = {}
    for split in ("train", "valid"):
        planned_maps, planned_tasks = tasks_for(generation_config, split)
        planned_by_uuid = {
            row["map_uuid"]: {
                1: "cave", 2: "pillar", 5: "forest", 7: "wall",
            }[int(row["maze_type"])]
            for row in planned_maps
        }
        planned_sequence_types[split] = Counter(
            planned_by_uuid[task["map_uuid"]] for task in planned_tasks
        )
        planned_by_id = {
            str(task["sequence_id"]): task for task in planned_tasks
        }
        if set(sequence_rows[split]) != set(planned_by_id):
            errors.append(
                f"{split} sequence identity set differs from generation plan"
            )
        else:
            identity_fields = (
                # The frozen sequence manifest does not duplicate the plan
                # seed; it is already bound by the generation-plan/config
                # hashes. Compare every task identity field persisted there.
                "suite", "scenario", "map_uuid", "frame_count",
            )
            for sequence_id, planned in planned_by_id.items():
                actual = sequence_rows[split][sequence_id]
                if any(actual.get(name) != planned.get(name)
                       for name in identity_fields):
                    errors.append(
                        f"sequence differs from plan: {sequence_id}"
                    )
                    break
        if sequence_types[split] != planned_sequence_types[split]:
            errors.append(
                f"{split} sequence type distribution differs from plan: "
                f"actual={dict(sequence_types[split])}, "
                f"planned={dict(planned_sequence_types[split])}"
            )
    expected_map_counts = {
        "train": Counter({name: 12 for name in EXPECTED_TYPES}),
        "valid": Counter({name: 3 for name in EXPECTED_TYPES}),
    }
    actual_map_counts = {}
    current_profiles = {
        row["profile_name"]: profile_hash(row) for row in load_profiles()
    }
    current_generator_hash = source_hash()
    for split in ("train", "valid"):
        counts = Counter()
        for map_uuid in maps[split]:
            provenance = load_json(
                RAW / f"geometry_authority/{split}/{map_uuid}"
                / "mixed_scene_provenance.json"
            )
            map_type = provenance["resolved_parameters"].get("map_type")
            if map_type not in EXPECTED_TYPES or int(provenance["maze_type"]) == 6:
                errors.append(f"room/unknown authority map: {map_uuid}")
                continue
            profile_name = provenance.get("profile_name")
            if profile_name not in current_profiles \
                    or provenance.get("profile_hash") != current_profiles[profile_name]:
                errors.append(f"stale map profile identity: {map_uuid}")
            if provenance.get("generator_source_hash") != current_generator_hash:
                errors.append(f"stale map generator identity: {map_uuid}")
            counts[map_type] += 1
            if map_type == "wall":
                parameters = provenance["resolved_parameters"]
                size_class = parameters.get("size_class")
                if size_class not in {"large", "narrow"}:
                    errors.append(f"wall size class mismatch: {map_uuid}")
                    continue
                expected_walls = 100 if size_class == "large" else 25
                expected = {
                    "wall_number": expected_walls,
                    "wall_width_min": 0.5,
                    "wall_width_max": 6.0,
                    "wall_thick": 0.5,
                    "wall_ceiling": 1,
                    "x_length": 60 if size_class == "large" else 30,
                    "y_length": 60 if size_class == "large" else 30,
                    "z_length": 15 if size_class == "large" else 10,
                }
                for key, value in expected.items():
                    if float(parameters[key]) != float(value):
                        errors.append(f"wall profile mismatch {map_uuid}: {key}")
        actual_map_counts[split] = counts
        if counts != expected_map_counts[split]:
            errors.append(f"{split} map balance mismatch: {dict(counts)}")

    height_metrics = {}
    for split in ("train", "valid"):
        present = {kind for (part, kind) in altitude if part == split}
        if present != EXPECTED_TYPES:
            errors.append(f"{split} map-type coverage mismatch: {sorted(present)}")
        height_metrics[split] = {}
        for kind in sorted(EXPECTED_TYPES):
            values = np.asarray(altitude[(split, kind)], dtype=np.float64)
            observed = np.asarray(
                observability[(split, kind)], dtype=np.float64
            )
            if not values.size or not observed.size:
                continue
            height_metrics[split][kind] = {
                "frames": int(values.size),
                "relative_altitude_p10_p50_p90": [
                    float(v) for v in np.quantile(values, [0.1, 0.5, 0.9])
                ],
                "low_fraction_le_0_40": float(np.mean(values <= 0.40)),
                "high_fraction_gt_0_68": float(np.mean(values > 0.68)),
                "return_fraction_min": float(observed[:, 0].min()),
                "near_obstacle_fraction_min": float(observed[:, 1].min()),
            }
            if kind in {"forest", "pillar", "wall"}:
                if float(np.mean(values <= 0.40)) < 0.40:
                    errors.append(f"{split}/{kind}: insufficient low altitude")
                if float(np.mean(values > 0.68)) > 0.22:
                    errors.append(f"{split}/{kind}: excessive high altitude")

    if derived_manifest.get("status") != "COMPLETE_FROZEN":
        errors.append("derived dataset is not COMPLETE_FROZEN")
    if derived_manifest.get("dataset_version") != DERIVED_VERSION:
        errors.append("derived dataset version mismatch")
    if derived_manifest.get("source_dataset_manifest_hash") != sha(raw_manifest_path):
        errors.append("derived/raw identity mismatch")
    for split in ("train", "validation"):
        types = set(derived_manifest.get("map_type_sample_counts", {}).get(split, {}))
        if types != EXPECTED_TYPES:
            errors.append(f"derived {split} map-type coverage mismatch")
    if derived_manifest.get("actor_input_used") is not False \
            or derived_manifest.get("composed_depth_used") is not False:
        errors.append("V4.6 derived dataset is not actor-free static depth")
    split_counts = derived_manifest.get("split_counts", {})
    if not (0 < int(split_counts.get("train", 0)) <= 300000
            and 0 < int(split_counts.get("validation", 0)) <= 60000):
        errors.append(f"derived split counts invalid: {split_counts}")

    authority_rows = load_json(
        DERIVED / "manifests/map_authority.json"
    ).get("maps", [])
    authority_by_id = {int(row["map_id"]): row for row in authority_rows}
    derived_map_counts = {}
    derived_sample_counts = {}
    for split in ("train", "validation"):
        expected_source_split = "train" if split == "train" else "valid"
        authority_counts = Counter(
            row.get("map_type") for row in authority_rows
            if row.get("source_split") == expected_source_split
        )
        derived_map_counts[split] = authority_counts
        expected_authority = Counter({
            name: 12 if split == "train" else 3 for name in EXPECTED_TYPES
        })
        if authority_counts != expected_authority:
            errors.append(
                f"derived {split} map authority mismatch: "
                f"{dict(authority_counts)}"
            )
        map_ids = np.load(
            DERIVED / f"indices/{split}/map_id.npy", mmap_mode="r"
        )
        if len(map_ids) != int(split_counts.get(split, -1)):
            errors.append(f"derived {split} map_id length mismatch")
        sample_types = Counter()
        unknown_ids = set()
        for map_id, count in zip(*np.unique(map_ids, return_counts=True)):
            authority = authority_by_id.get(int(map_id))
            if authority is None:
                unknown_ids.add(int(map_id))
                continue
            sample_types[str(authority.get("map_type"))] += int(count)
        if unknown_ids:
            errors.append(
                f"derived {split} has unknown map ids: {sorted(unknown_ids)}"
            )
        derived_sample_counts[split] = sample_types
        if set(sample_types) != EXPECTED_TYPES \
                or any(sample_types[name] <= 0 for name in EXPECTED_TYPES) \
                or sum(sample_types.values()) != int(split_counts.get(split, -1)):
            errors.append(
                f"derived {split} actual sample types invalid: "
                f"{dict(sample_types)}"
            )
        declared = Counter({
            str(name): int(count) for name, count in derived_manifest.get(
                "map_type_sample_counts", {}
            ).get(split, {}).items()
        })
        if declared != sample_types:
            errors.append(
                f"derived {split} declared/actual sample counts differ"
            )

    result = {
        "status": "PASS" if not errors else "FAIL",
        "raw": str(RAW),
        "derived": str(DERIVED),
        "raw_manifest_sha256": sha(raw_manifest_path),
        "derived_manifest_sha256": sha(derived_manifest_path),
        "map_types": sorted(EXPECTED_TYPES),
        "room_samples": int(sum(
            values.get("room", 0) for values in derived_sample_counts.values()
        )),
        "sequences": dict(sequences),
        "raw_frames": dict(raw_frames),
        "sequence_types": {
            split: dict(values) for split, values in sequence_types.items()
        },
        "planned_sequence_types": {
            split: dict(values)
            for split, values in planned_sequence_types.items()
        },
        "map_counts": {
            split: dict(values) for split, values in actual_map_counts.items()
        },
        "derived_map_counts": {
            split: dict(values) for split, values in derived_map_counts.items()
        },
        "derived_sample_counts": {
            split: dict(values) for split, values in derived_sample_counts.items()
        },
        "height_and_observability": height_metrics,
        "errors": errors,
        "training_started": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
