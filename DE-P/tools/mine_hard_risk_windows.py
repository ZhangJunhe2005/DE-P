#!/usr/bin/env python3
"""Mine Phase-8B windows once, then refuse to mutate the immutable lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase8b_common import ROOT, collate, evaluate_batch, jsonable, load_config, make_trainer


SOURCES = (
    (ROOT / "data/phase8_preflight_multimap", None),
    (ROOT / "data/phase7_dynamic_pilot", ROOT / "configs/maps_phase7_control.yaml"),
)


def entry(dataset, index, metrics, root):
    sequence, current, category = dataset._windows[index]
    _directory, metadata, frames = dataset._sequences[sequence]
    frame = frames[current]
    result = {
        "dataset_root": str(root.resolve()), "split": dataset.split,
        "sequence_id": sequence, "frame_index": int(frame["frame_index"]),
        "map_id": int(frame["map_id"]), "seed": int(metadata["random_seed"]),
        "scenario": str(metadata.get("scenario_type", frame["scenario_id"])),
        "sample_category": category, "baseline": jsonable(metrics),
    }
    return result


def candidate_indices(dataset, category, limit):
    values = [i for i, value in enumerate(dataset.sample_categories) if value == category]
    if not values:
        return []
    positions = np.linspace(0, len(values) - 1, min(limit, len(values)), dtype=int)
    return [values[int(position)] for position in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase8b_gradient_diagnostics.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics")
    args = parser.parse_args()
    outputs = {name: args.output_dir / f"fixed_{name}.json" for name in (
        "no_target", "hard_risk", "temporal_separation", "occluded_tracked"
    )}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"immutable diagnostic lists already exist; refusing to re-mine: {existing}")
    config = load_config(args.config)
    pools = {name: [] for name in outputs}
    skipped_invalid_windows = []
    for root, map_catalog in SOURCES:
        trainer = make_trainer(config, data_root=root, map_catalog=map_catalog)
        for split in ("train", "valid", "test"):
            dataset = trainer.train_loaders["dynamic"].dataset if split == "train" else None
            if dataset is None or dataset.split != split:
                from policy.dynamic_sequence_dataset import DynamicSequenceDataset
                from policy.dynamic_training_config import DynamicTrainingConfig
                dataset = DynamicSequenceDataset(
                    root, split,
                    training_config=DynamicTrainingConfig.from_mapping(config["dynamic_training"]),
                )
            category_map = {
                "no_target": "no_target",
                "temporal_separation": "temporally_separated",
                "occluded_tracked": "occluded_but_tracked",
            }
            for name, category in category_map.items():
                for index in candidate_indices(dataset, category, 2):
                    try:
                        metrics = evaluate_batch(trainer, collate([dataset[index]]))
                    except ValueError as error:
                        skipped_invalid_windows.append({"root": str(root), "split": split,
                                                        "index": index, "error": str(error)})
                        continue
                    pools[name].append(entry(dataset, index, metrics, root))
            if split == "train" and root.name == "phase8_preflight_multimap":
                # Training-only hard set.  Scan representative categories and
                # retain physical collision / few-safe / rank-conflict cases.
                for category in ("visible_high_risk", "multi_target", "temporally_separated",
                                 "visible_low_risk", "occluded_but_tracked"):
                    for index in candidate_indices(dataset, category, 5):
                        metrics = evaluate_batch(trainer, collate([dataset[index]]))
                        item = entry(dataset, index, metrics, root)
                        conflict = metrics["oracle_regret"] > 1e-6 or metrics["spearman"] < 0
                        hard = (metrics["top1_min_clearance"] is not None and
                                metrics["top1_min_clearance"] < config["risk_metrics"]["hard_window_threshold"])
                        few_safe = metrics["safe_candidate_count_mean"] <= 7
                        if hard or conflict or few_safe:
                            pools["hard_risk"].append(item)
    # Preserve scenario diversity while keeping bounded tools fast.
    hard_sorted = sorted(pools["hard_risk"], key=lambda item: (
        item["baseline"]["top1_min_clearance"] is None,
        item["baseline"]["top1_min_clearance"] or 1e9,
        -item["baseline"]["oracle_regret"],
    ))
    selected, scenarios = [], set()
    for item in hard_sorted:
        key = (Path(item["dataset_root"]).name, item["scenario"])
        if key not in scenarios or len(selected) < 8:
            selected.append(item); scenarios.add(key)
        if len(selected) >= 12:
            break
    pools["hard_risk"] = selected
    if not pools["hard_risk"]:
        raise RuntimeError("no hard-risk windows met the physical/ranking criteria")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in outputs.items():
        payload = {
            "schema_version": 1, "immutable": True,
            "mined_checkpoint": config["initialization_checkpoint"],
                    "selection_rule": "baseline corrected model; never re-mine for comparisons",
            "skipped_invalid_window_count": len(skipped_invalid_windows),
            "windows": pools[name],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(f"{path}: {len(pools[name])} windows")


if __name__ == "__main__":
    main()
