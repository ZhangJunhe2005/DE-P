#!/usr/bin/env python3
"""Full-window formal risk distribution and sampler audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from phase8b_common import load_config, make_trainer
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits
from policy.dynamic_training_config import DynamicTrainingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["dataset_root"]).resolve()
    manifest, splits = validate_dataset_splits(root)
    training = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
    trainer = make_trainer(config, data_root=root)
    split_results = {}
    failures = []
    for split in ("train", "valid", "test"):
        dataset = DynamicSequenceDataset(root, split, training_config=training)
        loader = DataLoader(
            dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0,
            collate_fn=dynamic_sequence_collate,
        )
        raw_values, clearances = [], []
        category_counts = Counter(dataset.sample_categories)
        scenario_counts = Counter(
            dataset._sequences[sequence][1]["scenario_type"] for sequence in splits[split]
        )
        no_target_max_abs = 0.0
        target_window_count = 0
        high_risk_candidate_count = 0
        windows_by_map = Counter()
        actor_count = visible_count = occluded_count = active_invisible_count = 0
        sequence_weight = Counter()
        total_weight = 0.0
        for sequence, current, category in dataset._windows:
            directory, _metadata, frames = dataset._sequences[sequence]
            frame = frames[current]
            windows_by_map[int(frame["map_id"])] += 1
            weight = float(training.sampler_weights[category])
            sequence_weight[sequence] += weight
            total_weight += weight
            objects = dataset._objects(directory / frame["dynamic_objects_path"])
            actor_count += len(objects)
            visible_count += sum(int(float(obj["visibility"]) > 0) for obj in objects)
            occluded_count += sum(int(bool(obj["occluded"])) for obj in objects)
            active_invisible_count += sum(int(
                bool(obj.get("active", True)) and float(obj["visibility"]) <= 0
            ) for obj in objects)
        for batch in loader:
            with torch.inference_mode():
                details = trainer.compute_batch("dynamic", batch)
            raw = details["candidate_dynamic_cost_raw"].reshape(-1, 15).detach().cpu().numpy()
            diagnostics = details["dynamic_diagnostics"]
            target = diagnostics.dynamic_obstacle_count.detach().cpu().numpy() > 0
            clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()
            high_risk_candidate_count += int(
                diagnostics.candidate_high_risk_mask.sum().detach().cpu()
            )
            target_window_count += int(target.sum())
            if bool(target.any()):
                raw_values.append(raw[target].reshape(-1))
                clearances.append(clearance[target].reshape(-1))
            if bool((~target).any()):
                no_target_max_abs = max(no_target_max_abs, float(np.abs(raw[~target]).max()))
        raw = np.concatenate(raw_values) if raw_values else np.zeros(1)
        clearance = np.concatenate(clearances) if clearances else np.full(1, np.inf)
        finite_clearance = clearance[np.isfinite(clearance)]
        split_results[split] = {
            "sequence_count": len(splits[split]), "window_count": len(dataset),
            "target_bearing_window_count": target_window_count,
            "actor_instance_count": actor_count,
            "visible_actor_ratio": visible_count / max(actor_count, 1),
            "occluded_actor_ratio": occluded_count / max(actor_count, 1),
            "active_but_invisible_actor_ratio": active_invisible_count / max(actor_count, 1),
            "category_counts": dict(sorted(category_counts.items())),
            "scenario_sequence_counts": dict(sorted(scenario_counts.items())),
            "dynamic_cost_quantiles": {
                str(value): float(np.quantile(raw, value)) for value in (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
            },
            "positive_violation_candidate_fraction": float((clearance < 0).mean()),
            "safe_candidate_fraction": float((clearance >= 0).mean()),
            "high_risk_candidate_count": high_risk_candidate_count,
            "clearance_quantiles_m": {
                str(value): float(np.quantile(finite_clearance, value))
                for value in (0, 0.1, 0.5, 0.9, 1)
            } if finite_clearance.size else {},
            "no_target_dynamic_max_abs": no_target_max_abs,
            "validation_sampler_is_sequential": bool(
                split == "train" or isinstance(loader.sampler, SequentialSampler)
            ),
            "windows_by_map": {str(key): value for key, value in sorted(windows_by_map.items())},
            "expected_sequence_sampling_probability": {
                key: value / total_weight for key, value in sorted(sequence_weight.items())
            },
            "maximum_expected_sequence_sampling_probability": (
                max(sequence_weight.values()) / total_weight
            ),
        }
        required_categories = {
            "no_target", "visible_low_risk", "visible_high_risk",
            "temporally_separated", "occluded_but_tracked", "multi_target",
        }
        missing = required_categories - set(category_counts)
        if missing:
            failures.append(f"{split}: missing window categories {sorted(missing)}")
        if no_target_max_abs != 0.0:
            failures.append(f"{split}: no-target dynamic cost is not exactly zero")
        if high_risk_candidate_count == 0:
            failures.append(f"{split}: high-risk candidate set is empty")
        if category_counts.get("no_target", 0) / len(dataset) > 0.70:
            failures.append(f"{split}: no-target windows exceed 70 percent")
        if split == "train" and max(sequence_weight.values()) / total_weight > 0.03:
            failures.append("one training sequence exceeds 3% expected sampler probability")

    sampled_categories, sampled_sequences = Counter(), Counter()
    for batch in trainer.train_loaders["dynamic"]:
        sampled_categories.update(batch["sample_category"])
        sampled_sequences.update(batch["sequence_id"])
    total_sampled = sum(sampled_categories.values())
    sampler_audit = {
        "sample_count": total_sampled,
        "category_counts": dict(sorted(sampled_categories.items())),
        "category_fractions": {key: value / total_sampled
                               for key, value in sorted(sampled_categories.items())},
        "maximum_single_sequence_fraction": max(sampled_sequences.values()) / total_sampled,
        "expected_probability_source": "per-window configured category weight",
        "valid_uses_weighted_sampler": False, "test_uses_weighted_sampler": False,
    }
    if sampler_audit["maximum_single_sequence_fraction"] > 0.03:
        failures.append("one training sequence exceeds 3% of sampled windows")
    payload = {
        "status": "PASS" if not failures else "FAIL", "dataset": str(root),
        "manifest": manifest, "splits": split_results,
        "training_sampler_audit": sampler_audit,
        "pointcloud_training_allowed": False, "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
