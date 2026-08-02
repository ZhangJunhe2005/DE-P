#!/usr/bin/env python3
"""Freeze independent Phase-8C train/valid/test diagnostic windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from phase8b_common import ROOT, collate, evaluate_batch, jsonable, load_config, make_trainer
from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits
from policy.dynamic_training_config import DynamicTrainingConfig


OUTPUT_NAMES = (
    "production_train_hard_risk", "production_valid_hard_risk",
    "production_test_hard_risk", "production_valid_no_target",
    "production_test_no_target", "production_valid_temporal_separation",
    "production_test_temporal_separation", "production_valid_occluded_tracked",
    "production_test_occluded_tracked",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representative(indices, limit):
    if not indices:
        return []
    positions = np.linspace(0, len(indices) - 1, min(limit, len(indices)), dtype=int)
    return [indices[int(position)] for position in positions]


def window_entry(dataset, index, metrics, root):
    sequence, current, category = dataset._windows[index]
    _directory, metadata, frames = dataset._sequences[sequence]
    frame = frames[current]
    scenario = str(metadata["scenario_type"])
    return {
        "dataset_root": str(root), "split": dataset.split,
        "sequence_id": sequence, "frame_index": int(frame["frame_index"]),
        "map_id": int(frame["map_id"]), "seed": int(metadata["random_seed"]),
        "scenario": scenario, "sample_category": category,
        "coverage_tags": {
            "visible_high_or_low_risk": scenario in {"crossing", "head_on", "multi_target"},
            "temporal_separation": scenario == "temporal_separation",
            "occluded_but_tracked": scenario == "occluded_but_tracked",
            "multi_target": scenario == "multi_target",
            "waypoint_reversal": scenario in {"multi_target", "occluded_but_tracked"},
            "delayed_start": scenario in {
                "head_on", "multi_target", "temporal_separation", "occluded_but_tracked"
            },
            "held_out_actor_speed_and_timing": dataset.split == "test",
        },
        "baseline": jsonable(metrics),
    }


def scenario_indices(dataset):
    result = defaultdict(list)
    for index, (sequence, _current, _category) in enumerate(dataset._windows):
        result[str(dataset._sequences[sequence][1]["scenario_type"])].append(index)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_dynamic_production.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics")
    args = parser.parse_args()
    paths = {name: args.output_dir / f"{name}.json" for name in OUTPUT_NAMES}
    manifest_path = args.output_dir / "phase8c_production_risk_set_manifest.json"
    existing = [str(path) for path in (*paths.values(), manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"risk sets are immutable; refusing to overwrite: {existing}")
    config = load_config(args.config)
    root = Path(config["dataset_root"]).resolve()
    manifest, split_sequences = validate_dataset_splits(root)
    training = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
    trainer = make_trainer(config, data_root=root)
    pools = {}
    baseline_scan_counts = {}
    for split in ("train", "valid", "test"):
        dataset = DynamicSequenceDataset(root, split, training_config=training)
        by_scenario = scenario_indices(dataset)
        scanned = []
        for scenario, indices in sorted(by_scenario.items()):
            if scenario == "no_target":
                continue
            for index in representative(indices, 10):
                metrics = evaluate_batch(trainer, collate([dataset[index]]))
                scanned.append(window_entry(dataset, index, metrics, root))
        scanned.sort(key=lambda item: (
            item["baseline"]["top1_min_clearance"] is None,
            (item["baseline"]["top1_min_clearance"]
             if item["baseline"]["top1_min_clearance"] is not None else 1e9),
            -item["baseline"]["oracle_regret"],
        ))
        selected, per_scenario = [], defaultdict(int)
        for item in scanned:
            if per_scenario[item["scenario"]] < 4:
                selected.append(item)
                per_scenario[item["scenario"]] += 1
            if len(selected) >= 24:
                break
        pools[f"production_{split}_hard_risk"] = selected
        baseline_scan_counts[split] = len(scanned)

        for scenario, suffix, limit in (
            ("no_target", "no_target", 12),
            ("temporal_separation", "temporal_separation", 12),
            ("occluded_but_tracked", "occluded_tracked", 12),
        ):
            if (split, scenario) not in {
                ("valid", "no_target"), ("test", "no_target"),
                ("valid", "temporal_separation"), ("test", "temporal_separation"),
                ("valid", "occluded_but_tracked"), ("test", "occluded_but_tracked"),
            }:
                continue
            values = []
            for index in representative(by_scenario[scenario], limit):
                metrics = evaluate_batch(trainer, collate([dataset[index]]))
                values.append(window_entry(dataset, index, metrics, root))
            pools[f"production_{split}_{suffix}"] = values

    checkpoint = Path(config["initialization_checkpoint"])
    common = {
        "schema_version": 1, "immutable": True,
        "selection_rule": "frozen corrected baseline; lists must never be re-mined after training",
        "baseline_checkpoint": str(checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256(checkpoint),
        "config": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "dataset_manifest_sha256": sha256(root / "dataset_manifest.yaml"),
        "map_splits": manifest["map_splits"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        windows = pools.get(name, [])
        if not windows:
            raise RuntimeError(f"no windows selected for {name}")
        path.write_text(json.dumps({**common, "windows": windows}, indent=2) + "\n")
    file_hashes = {path.name: sha256(path) for path in paths.values()}
    manifest_payload = {
        "status": "PASS", **common, "baseline_scan_counts": baseline_scan_counts,
        "split_sequence_counts": {key: len(value) for key, value in split_sequences.items()},
        "file_sha256": file_hashes,
        "validation_and_test_used_by_training_sampler": False,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    print(json.dumps(manifest_payload, indent=2))


if __name__ == "__main__":
    main()
