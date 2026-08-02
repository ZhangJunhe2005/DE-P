#!/usr/bin/env python3
"""Component and parameter-group gradient conflict diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from phase8b_common import ROOT, collate, load_config, load_fixed_samples, make_trainer


COMPONENTS = ("smooth", "static", "guidance", "dynamic_raw", "dynamic_weighted", "score", "total")


def group_vector(named, gradients, group):
    pieces = []
    for (name, parameter), gradient in zip(named, gradients):
        if gradient is None:
            continue
        if group == "late_mobile" and name.startswith("image_backbone.backbone.0."):
            pieces.append(gradient.reshape(-1))
        elif group == "output_1x1" and name.startswith("image_backbone.backbone.1."):
            pieces.append(gradient.reshape(-1))
        elif group == "shared_head_trunk" and name.startswith(("dep_head.model.0.", "dep_head.model.2.")):
            pieces.append(gradient.reshape(-1))
        elif group == "trajectory_output_branch" and name.startswith("dep_head.model.4."):
            pieces.append(gradient[:9].reshape(-1))
        elif group == "score_output_branch" and name.startswith("dep_head.model.4."):
            pieces.append(gradient[9:].reshape(-1))
        elif group == "all":
            pieces.append(gradient.reshape(-1))
    if not pieces:
        return torch.zeros(1, device=named[0][1].device)
    return torch.cat(pieces)


def cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0:
        return 0.0
    return float(torch.dot(left, right).div(denominator).detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase8b_gradient_diagnostics.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/phase8b_gradient_diagnostics/gradients")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = make_trainer(config)
    samples = (
        load_fixed_samples(config, ROOT / "diagnostics/fixed_hard_risk.json", training_only=True)
        + load_fixed_samples(config, ROOT / "diagnostics/fixed_no_target.json", training_only=True)
    )
    configured_root = Path(config["dataset_root"]).resolve()
    samples = [sample for sample in samples
               if Path(sample["fixed_entry"]["dataset_root"]).resolve() == configured_root]
    named = [(name, parameter) for name, parameter in trainer.policy.named_parameters()
             if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    groups = ("late_mobile", "output_1x1", "trajectory_output_branch",
              "score_output_branch", "shared_head_trunk", "all")
    rows, cosine_rows = [], []
    for sample_index, sample in enumerate(samples):
        trainer.set_training_mode()
        details = trainer.compute_batch("dynamic", collate([sample]))
        values = {
            "smooth": details["smooth_loss"], "static": details["static_safety_loss"],
            "guidance": details["guidance_loss"], "dynamic_raw": details["raw_dynamic_safety_loss"],
            "dynamic_weighted": details["dynamic_safety_loss"], "score": details["score_loss"],
            "total": details["trajectory_loss"] + details["score_loss"],
        }
        gradients, vectors = {}, {}
        for component in COMPONENTS:
            gradients[component] = torch.autograd.grad(
                values[component], parameters, retain_graph=True, allow_unused=True
            )
            for group in groups:
                vector = group_vector(named, gradients[component], group)
                vectors[(component, group)] = vector
                rows.append({
                    "sample_index": sample_index, "sequence_id": sample["sequence_id"],
                    "frame_index": sample["frame_index"], "category": sample["sample_category"],
                    "component": component, "parameter_group": group,
                    "gradient_norm": float(torch.linalg.vector_norm(vector).detach().cpu()),
                    "max_abs_gradient": float(vector.abs().max().detach().cpu()),
                    "finite": bool(torch.isfinite(vector).all()),
                })
        total_norm = max(float(torch.linalg.vector_norm(vectors[("total", "all")]).cpu()), 1e-30)
        for row in rows[-len(COMPONENTS) * len(groups):]:
            row["component_total_norm_ratio"] = row["gradient_norm"] / total_norm
        for group in groups:
            for left_index, left in enumerate(COMPONENTS):
                for right in COMPONENTS[left_index + 1:]:
                    cosine_rows.append({
                        "sample_index": sample_index, "sequence_id": sample["sequence_id"],
                        "category": sample["sample_category"], "parameter_group": group,
                        "left": left, "right": right,
                        "cosine": cosine(vectors[(left, group)], vectors[(right, group)]),
                    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("gradient_norms.csv", rows), ("gradient_cosines.csv", cosine_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    pairs = (("dynamic_weighted", "static"), ("dynamic_weighted", "guidance"),
             ("dynamic_weighted", "score"), ("dynamic_weighted", "smooth"))
    summary = {}
    for left, right in pairs:
        values = [row["cosine"] for row in cosine_rows
                  if row["parameter_group"] == "all"
                  and {row["left"], row["right"]} == {left, right}]
        summary[f"{left}_vs_{right}"] = {
            "mean": float(np.mean(values)), "negative_fraction": float(np.mean(np.asarray(values) < 0)),
        }
    norms = {component: [row["gradient_norm"] for row in rows
                         if row["parameter_group"] == "all" and row["component"] == component]
             for component in COMPONENTS}
    summary["gradient_norms"] = {key: {"mean": float(np.mean(value)),
                                               "median": float(np.median(value))}
                                 for key, value in norms.items()}
    try:
        import matplotlib.pyplot as plt
        labels = list(COMPONENTS[:-1])
        matrix = np.eye(len(labels))
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if i >= j:
                    continue
                values = [row["cosine"] for row in cosine_rows if row["parameter_group"] == "all"
                          and row["left"] == left and row["right"] == right]
                matrix[i, j] = matrix[j, i] = np.mean(values)
        figure, axis = plt.subplots(figsize=(7, 6)); image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels); figure.colorbar(image, ax=axis)
        figure.tight_layout(); figure.savefig(args.output_dir / "gradient_cosine_heatmap.png", dpi=160)
        plt.close(figure)
    except ImportError:
        summary["heatmap"] = "matplotlib unavailable; CSV is authoritative"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "samples": len(samples), **summary}, indent=2))
    if not all(row["finite"] and math.isfinite(row["gradient_norm"]) for row in rows):
        raise FloatingPointError("non-finite gradient diagnostic")


if __name__ == "__main__":
    main()
