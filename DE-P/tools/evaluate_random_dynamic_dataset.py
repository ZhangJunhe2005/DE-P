#!/usr/bin/env python3
"""Offline held-out map/actor comparison for static and dynamic checkpoints."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_trainer import DepTrainer


def state_dict(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    values = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    return values, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    static_state, static_meta = state_dict(args.static_checkpoint)
    dynamic_state, dynamic_meta = state_dict(args.dynamic_checkpoint)
    if dynamic_meta.get("checkpoint_role") != "dynamic_training":
        raise ValueError("dynamic checkpoint is not a managed dynamic-training checkpoint")
    if dynamic_meta.get("backbone_variant") != "corrected":
        raise ValueError("dynamic checkpoint must use corrected architecture")
    trainer = DepTrainer(
        batch_size=1, tensorboard_path=str(args.output_dir / "tb"),
        checkpoint_path=str(args.static_checkpoint), backbone_variant="corrected",
        dataset_mode="dynamic", dynamic_data_root=str(args.dataset),
        dynamic_loss_enabled=True, num_workers=0,
    )
    rows = []
    loader = trainer.val_loaders["dynamic"] if "valid" in str(args.dataset) else None
    # Formal evaluation is always the manifest's held-out test split.
    from policy.dynamic_sequence_dataset import DynamicSequenceDataset
    from policy.dynamic_collate import dynamic_sequence_collate
    dataset = DynamicSequenceDataset(args.dataset, "test")
    limit = len(dataset) if not args.max_windows else min(len(dataset), args.max_windows)
    for index in range(limit):
        batch = dynamic_sequence_collate([dataset[index]])
        per_model = {}
        for name, weights in (("static", static_state), ("dynamic", dynamic_state)):
            trainer.policy.load_state_dict(weights, strict=True)
            trainer.policy.eval()
            with torch.inference_mode():
                details = trainer.compute_batch("dynamic", batch)
                context = None if name == "static" else batch["dynamic_context"].to(trainer.device)
                endstate, scores = trainer.policy.inference(
                    batch["current_depth"].to(trainer.device),
                    batch["observation_9d"].to(trainer.device), dynamic_context=context,
                )
            labels = details["score_label"].detach().cpu().numpy()
            predicted = scores.reshape(-1).detach().cpu().numpy()
            top = int(np.argmin(predicted))
            oracle = float(labels.min())
            per_model[name] = {
                "endstate_l1": float(endstate.abs().mean().cpu()),
                "top1": top, "top1_true_cost": float(labels[top]),
                "oracle_regret": float(labels[top] - oracle),
                "collision_risk": float(labels[top] > 1.0),
                "spearman": float(spearmanr(predicted, labels).statistic),
                "kendall": float(kendalltau(predicted, labels).statistic),
                "true_future_dynamic_cost_mean": float(details["dynamic_safety_loss"]),
                "minimum_future_distance_proxy": float(details["min_dynamic_distance"]),
            }
        rows.append({"sequence_id": batch["sequence_id"][0],
                     "frame_index": int(batch["frame_index"][0]),
                     "category": batch["sample_category"][0],
                     "no_target_score_perturbation": abs(
                         per_model["dynamic"]["top1_true_cost"] - per_model["static"]["top1_true_cost"]
                     ) if batch["sample_category"][0] == "no_target" else 0.0,
                     **{f"{model}_{key}": value for model, values in per_model.items()
                        for key, value in values.items()}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {key: float(np.nanmean([row[key] for row in selected]))
                                for key in keys if key not in {"sequence_id", "frame_index", "category"}}
    result = {"status": "PASS", "held_out_split": "test", "window_count": len(rows),
              "static_checkpoint": str(args.static_checkpoint.resolve()),
              "dynamic_checkpoint": str(args.dynamic_checkpoint.resolve()),
              "categories": categories, "formal_pointcloud_training_allowed": False}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    labels = list(categories)
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
    axis.bar(x - 0.2, [categories[name]["static_oracle_regret"] for name in labels],
             width=0.4, label="static corrected")
    axis.bar(x + 0.2, [categories[name]["dynamic_oracle_regret"] for name in labels],
             width=0.4, label="dynamic trained")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("oracle regret")
    axis.legend(); figure.tight_layout()
    figure.savefig(args.output_dir / "oracle_regret_by_category.png", dpi=160)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
