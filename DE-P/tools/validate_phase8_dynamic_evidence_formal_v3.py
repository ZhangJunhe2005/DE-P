#!/usr/bin/env python3
"""Full storage and training-readiness audit for Formal V3 evidence data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.dynamic_evidence_v3 import (  # noqa: E402
    load_numeric_shard,
    sha256_file,
)


EXPECTED_ARRAYS = {
    "features", "validity_mask", "time_mask", "labels",
    "actual_speed_mps", "authority_codes", "chain_ids",
}
EXPECTED_SPLITS = {"train", "calibration", "validation", "internal_test"}
REQUIRED_PROVENANCE = {
    "config_hashes", "generator_hashes", "map_catalog_hash",
    "scene_map_matrix_hash", "split_manifest_hash",
}


def audit_unit(arguments):
    root, entry = arguments
    unit_path = root / entry["path"]
    errors = []
    if sha256_file(unit_path) != entry["sha256"]:
        return {"errors": [f"work-unit hash mismatch: {unit_path}"]}
    unit = json.loads(unit_path.read_text())
    shard_path = root / unit["shard"]
    try:
        arrays = load_numeric_shard(shard_path, unit["shard_sha256"])
    except Exception as error:
        return {"errors": [f"{shard_path}: {error}"]}
    if set(arrays) != EXPECTED_ARRAYS:
        errors.append(
            f"{shard_path}: arrays {sorted(arrays)} != "
            f"{sorted(EXPECTED_ARRAYS)}")
    features = arrays["features"]
    validity = arrays["validity_mask"]
    time_mask = arrays["time_mask"]
    labels = arrays["labels"]
    speeds = arrays["actual_speed_mps"]
    authority = arrays["authority_codes"]
    chain_ids = arrays["chain_ids"]
    count = int(unit["sample_count"])
    expected_shapes = {
        "features": (count, 4, 32),
        "validity_mask": (count, 4, 32),
        "time_mask": (count, 4),
        "labels": (count,),
        "actual_speed_mps": (count,),
        "authority_codes": (count,),
        "chain_ids": (count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            errors.append(
                f"{shard_path}: {name} {arrays[name].shape} != {shape}")
    if features.dtype != np.float32:
        errors.append(f"{shard_path}: feature dtype is {features.dtype}")
    if validity.dtype != np.bool_ or time_mask.dtype != np.bool_:
        errors.append(f"{shard_path}: mask dtype mismatch")
    if labels.dtype.kind not in "ui" or not np.all(labels < 4):
        errors.append(f"{shard_path}: invalid labels")
    if not np.isfinite(features).all() or not np.isfinite(speeds).all():
        errors.append(f"{shard_path}: non-finite numeric data")
    if not np.all(authority == labels + 1):
        errors.append(f"{shard_path}: label/authority mismatch")
    if np.any((labels == 0) & (speeds < 0.3 - 1e-6)):
        errors.append(f"{shard_path}: sub-threshold dynamic label")
    if len(unit.get("chain_hashes", ())) != count:
        errors.append(f"{unit_path}: chain hash count mismatch")
    label_counts = Counter(map(int, labels.tolist()))
    declared = {
        int(key): int(value)
        for key, value in unit.get("label_counts", {}).items()
    }
    if label_counts != Counter(declared):
        errors.append(f"{unit_path}: declared label counts mismatch")
    valid_count = validity.sum(axis=(0, 1), dtype=np.int64)
    valid_nonzero = (
        (validity & (features != 0)).sum(axis=(0, 1), dtype=np.int64)
    )
    return {
        "errors": errors,
        "sequence_id": unit["sequence_id"],
        "split": unit["split"],
        "source_split": unit["source_split"],
        "map_uuid": unit["map_uuid"],
        "scenario": unit["scenario"],
        "suite": unit["suite"],
        "samples": count,
        "label_counts": dict(label_counts),
        "valid_count": valid_count.tolist(),
        "valid_nonzero": valid_nonzero.tolist(),
        "chain_ids": [bytes(value) for value in chain_ids],
        "minimum_dynamic_speed": (
            float(speeds[labels == 0].min())
            if np.any(labels == 0) else None
        ),
        "maximum_speed": float(speeds.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "data/phase8_dynamic_evidence_formal_v3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/phase8_dynamic_evidence_formal_v3_audit.json")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest_path = root / "manifests/dataset_manifest.json"
    completion_path = (
        root / "generation_state/FORMAL_V3_GENERATION_COMPLETE.json")
    errors = []
    blockers = []
    if not manifest_path.is_file() or not completion_path.is_file():
        raise SystemExit("Formal V3 completion metadata is missing")
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    if completion.get("status") != "COMPLETE":
        errors.append("root completion status is not COMPLETE")
    if completion.get("manifest_sha256") != manifest_hash:
        errors.append("root completion manifest hash mismatch")
    if manifest.get("status") != "COMPLETE":
        errors.append("dataset manifest status is not COMPLETE")
    if manifest.get("sample_count") != 420000:
        errors.append("dataset sample count is not 420000")
    entries = manifest.get("work_units", [])
    if len(entries) != 7000:
        errors.append(f"work-unit count is {len(entries)}, expected 7000")
    if len({row["path"] for row in entries}) != len(entries):
        errors.append("duplicate work-unit paths")
    declared_tree = hashlib.sha256(
        "".join(row["sha256"] for row in entries).encode()
    ).hexdigest()
    if manifest.get("sample_hash_tree_root") != declared_tree:
        errors.append("sample hash-tree root mismatch")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(
            audit_unit, ((root, entry) for entry in entries)))
    unit_errors = [
        error for row in rows for error in row.get("errors", ())]
    errors.extend(unit_errors)
    valid_rows = [row for row in rows if "samples" in row]

    split_counts = Counter()
    label_counts = defaultdict(Counter)
    scenario_counts = defaultdict(Counter)
    maps_by_split = defaultdict(set)
    valid_count = np.zeros(32, dtype=np.int64)
    valid_nonzero = np.zeros(32, dtype=np.int64)
    chain_ids = set()
    duplicate_chain_ids = 0
    minimum_dynamic_speed = None
    maximum_speed = 0.0
    for row in valid_rows:
        split = row["split"]
        split_counts[split] += row["samples"]
        label_counts[split].update(row["label_counts"])
        scenario_counts[split][row["scenario"]] += row["samples"]
        maps_by_split[split].add(row["map_uuid"])
        valid_count += np.asarray(row["valid_count"], dtype=np.int64)
        valid_nonzero += np.asarray(row["valid_nonzero"], dtype=np.int64)
        for chain_id in row["chain_ids"]:
            if chain_id in chain_ids:
                duplicate_chain_ids += 1
            chain_ids.add(chain_id)
        value = row["minimum_dynamic_speed"]
        if value is not None:
            minimum_dynamic_speed = (
                value if minimum_dynamic_speed is None
                else min(minimum_dynamic_speed, value))
        maximum_speed = max(maximum_speed, row["maximum_speed"])
        if row["source_split"] == "train" and split != "train":
            errors.append(f"{row['sequence_id']}: train source split changed")
        if row["source_split"] == "valid" and split == "train":
            errors.append(f"{row['sequence_id']}: valid leaked into train")
    if dict(split_counts) != manifest.get("split_counts"):
        errors.append("computed split counts differ from manifest")
    if set(split_counts) != EXPECTED_SPLITS:
        errors.append(f"split set is {sorted(split_counts)}")
    for left in EXPECTED_SPLITS:
        for right in EXPECTED_SPLITS:
            if left < right and maps_by_split[left] & maps_by_split[right]:
                errors.append(f"map leakage between {left} and {right}")
    if duplicate_chain_ids:
        errors.append(f"duplicate chain IDs: {duplicate_chain_ids}")

    raw = root / "raw_authoritative"
    raw_manifests = list((raw / "manifests/sequences").glob("*.json"))
    raw_states = list(
        (raw / "generation_state/sequence_state").glob("*.json"))
    for marker in (
        "TRAIN_SPLIT_GENERATION_COMPLETE",
        "VALID_SPLIT_GENERATION_COMPLETE",
        "FULL_GENERATION_COMPLETE",
    ):
        if not (raw / f"generation_state/completion/{marker}").is_file():
            errors.append(f"missing raw completion marker: {marker}")
    if len(raw_manifests) != 7000 or len(raw_states) != 7000:
        errors.append(
            f"raw sequence counts are manifests={len(raw_manifests)}, "
            f"states={len(raw_states)}")

    missing_provenance = sorted(REQUIRED_PROVENANCE-set(manifest))
    if missing_provenance:
        blockers.append(
            "root manifest lacks required frozen provenance fields: "
            + ", ".join(missing_provenance))
    if manifest.get("normalization_statistics") == (
            "NOT_COMPUTED_BY_GENERATION"):
        blockers.append(
            "train-split-only robust normalization statistics are absent")
    feature_contract_violations = []
    for index in range(32):
        if valid_count[index] and not valid_nonzero[index]:
            feature_contract_violations.append({
                "feature_index": index,
                "valid_values": int(valid_count[index]),
                "nonzero_valid_values": 0,
                "problem": "storage fill is marked valid for every occurrence",
            })
        if not valid_count[index]:
            feature_contract_violations.append({
                "feature_index": index,
                "valid_values": 0,
                "nonzero_valid_values": 0,
                "problem": "feature is never available",
            })
    if feature_contract_violations:
        blockers.append(
            "32-feature binding is incomplete or violates missing-value masks")
    evaluation_class_gaps = {}
    for split in ("calibration", "validation", "internal_test"):
        missing = sorted(set(range(4))-set(label_counts[split]))
        if missing:
            evaluation_class_gaps[split] = missing
    if evaluation_class_gaps:
        blockers.append(
            "fixed evaluation split class coverage is incomplete: "
            + json.dumps(evaluation_class_gaps, sort_keys=True))
    first_unit = json.loads((root / entries[0]["path"]).read_text())
    first_shard = load_numeric_shard(
        root / first_unit["shard"], first_unit["shard_sha256"])
    if "dynamic_actionability" not in first_shard:
        blockers.append(
            "frozen dynamic_actionability auxiliary target is absent from shards")

    storage_status = "PASS" if not errors else "FAIL"
    training_status = (
        "PASS" if storage_status == "PASS" and not blockers else "FAIL")
    payload = {
        "status": training_status,
        "storage_integrity": storage_status,
        "training_ready": training_status == "PASS",
        "root": str(root),
        "manifest_sha256": manifest_hash,
        "sample_hash_tree_root": manifest.get("sample_hash_tree_root"),
        "work_units": len(entries),
        "raw_sequence_manifests": len(raw_manifests),
        "raw_sequence_states": len(raw_states),
        "samples": int(sum(split_counts.values())),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {
            split: {str(key): value for key, value in sorted(counts.items())}
            for split, counts in sorted(label_counts.items())
        },
        "scenario_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(scenario_counts.items())
        },
        "map_counts": {
            split: len(values)
            for split, values in sorted(maps_by_split.items())
        },
        "unique_chain_ids": len(chain_ids),
        "duplicate_chain_ids": duplicate_chain_ids,
        "minimum_dynamic_speed_mps": minimum_dynamic_speed,
        "maximum_speed_mps": maximum_speed,
        "feature_valid_count": valid_count.tolist(),
        "feature_valid_nonzero_count": valid_nonzero.tolist(),
        "feature_contract_violations": feature_contract_violations,
        "evaluation_class_gaps": evaluation_class_gaps,
        "missing_provenance_fields": missing_provenance,
        "errors": errors[:200],
        "error_count": len(errors),
        "training_blockers": blockers,
        "test_blind_production_access": manifest.get(
            "test_blind_production_access"),
        "training_executed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if training_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
