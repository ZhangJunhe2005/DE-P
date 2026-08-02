#!/usr/bin/env python3
"""Bounded, same-data/same-initialization Phase-8B objective ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from phase8b_common import (ROOT, collate, evaluate_batch, jsonable, load_config,
                            load_fixed_samples, make_trainer)
from loss.dynamic_types import DynamicObjectiveConfig


MATRIX = {
    "A_dynamic_only": ("dynamic",),
    "B_dynamic_score": ("dynamic", "score"),
    "C_dynamic_static": ("dynamic", "static"),
    "D_dynamic_guidance": ("dynamic", "guidance"),
    "E_dynamic_static_guidance": ("dynamic", "static", "guidance"),
    "F_full": ("dynamic", "smooth", "static", "guidance", "score"),
    "G_full_no_attention": ("dynamic", "smooth", "static", "guidance", "score"),
}


def train_objective(details, components):
    values = {
        "dynamic": details["dynamic_training_objective"],
        "smooth": details["smooth_loss"], "static": details["static_safety_loss"],
        "guidance": details["guidance_loss"], "score": details["score_loss"],
    }
    return sum((values[name] for name in components), details["dynamic_training_objective"] * 0)


def run(config, name, components, samples, no_target, steps, seed, objective=None,
        freeze_policy=None):
    trainer = make_trainer(config, seed=seed, objective=objective,
                           attention=name != "G_full_no_attention", freeze_policy=freeze_policy)
    evaluation_batch = collate(samples)
    no_target_batch = collate(no_target)
    before = evaluate_batch(trainer, evaluation_batch)
    no_target_before = evaluate_batch(trainer, no_target_batch)
    training_samples = list(samples)
    np.random.default_rng(seed).shuffle(training_samples)
    batches = [collate(training_samples[index:index + int(config["batch_size"])])
               for index in range(0, len(training_samples), int(config["batch_size"]))]
    losses, gradient_norms = [], []
    for step in range(steps):
        trainer.set_training_mode()
        trainer.optimizer.zero_grad(set_to_none=True)
        details = trainer.compute_batch("dynamic", batches[step % len(batches)])
        objective_value = train_objective(details, components)
        if not bool(torch.isfinite(objective_value)):
            raise FloatingPointError(f"{name}: non-finite objective")
        parameters = [parameter for parameter in trainer.policy.parameters() if parameter.requires_grad]
        if (trainer.dynamic_objective_config.gradient_strategy == "dynamic_priority_pcgrad"
                and "dynamic" in components and len(components) > 1):
            trainer._dynamic_priority_pcgrad(
                details["dynamic_training_objective"],
                objective_value - details["dynamic_training_objective"], parameters,
                trainer.dynamic_objective_config.pcgrad_other_norm_ratio,
            )
        else:
            objective_value.backward()
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad) for parameter in parameters
            if parameter.grad is not None
        ]))
        torch.nn.utils.clip_grad_norm_(parameters, trainer.max_grad_norm)
        trainer.optimizer.step()
        losses.append(float(objective_value.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))
    after = evaluate_batch(trainer, evaluation_batch)
    no_target_after = evaluate_batch(trainer, no_target_batch)
    primitive_delta = (after["raw"] - before["raw"]).mean(axis=0).tolist()
    trainer.epoch_i = 0
    tracked = next(parameter for parameter in trainer.policy.parameters() if parameter.requires_grad)
    saved = tracked.detach().clone()
    checkpoint = Path(tempfile.mkdtemp(prefix="dep-phase8b-resume-")) / "resume.pt"
    trainer.save_checkpoint(checkpoint)
    with torch.no_grad():
        tracked.add_(1.0)
    trainer.resume_training(checkpoint)
    resume_exact = bool(torch.equal(saved, tracked.detach()))
    result = {
        "name": name, "components": list(components), "seed": seed, "steps": steps,
        "before": jsonable(before), "after": jsonable(after),
        "delta": {key: jsonable(after)[key] - jsonable(before)[key]
                  for key in jsonable(before) if isinstance(jsonable(before)[key], (int, float))
                  and jsonable(before)[key] is not None and jsonable(after)[key] is not None},
        "no_target_before": jsonable(no_target_before), "no_target_after": jsonable(no_target_after),
        "primitive_dynamic_mean_delta": primitive_delta,
        "objective_first": losses[0], "objective_last": losses[-1],
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "finite": bool(np.isfinite(losses + gradient_norms).all()),
        "checkpoint_resume_exact": resume_exact,
    }
    trainer.tensorboard_log.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase8b_gradient_diagnostics.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--only", choices=tuple(MATRIX))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/phase8b_gradient_diagnostics/objective_ablation.json")
    parser.add_argument("--reference-scale", type=float)
    parser.add_argument("--cvar-coefficient", type=float)
    parser.add_argument("--dynamic-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gradient-strategy",
                        choices=("weighted_sum", "dynamic_priority_pcgrad"))
    parser.add_argument("--pcgrad-other-norm-ratio", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dynamic_weight is not None:
        config["dynamic_loss"]["weight"] = args.dynamic_weight
    if args.reference_scale is not None:
        config["dynamic_objective"]["reference_scale"] = args.reference_scale
    if args.cvar_coefficient is not None:
        config["dynamic_objective"]["cvar_coefficient"] = args.cvar_coefficient
    if args.gradient_strategy is not None:
        config["dynamic_objective"]["gradient_strategy"] = args.gradient_strategy
    if args.pcgrad_other_norm_ratio is not None:
        config["dynamic_objective"]["pcgrad_other_norm_ratio"] = args.pcgrad_other_norm_ratio
    objective = DynamicObjectiveConfig.from_mapping(config["dynamic_objective"])
    steps = int(args.steps or config["steps"])
    if not 30 <= steps <= 100:
        raise ValueError("Phase-8B bounded ablation steps must be in [30,100]")
    samples = load_fixed_samples(config, ROOT / "diagnostics/fixed_hard_risk.json", training_only=True)
    no_target = load_fixed_samples(config, ROOT / "diagnostics/fixed_no_target.json", training_only=True)
    configured_root = Path(config["dataset_root"]).resolve()
    no_target = [sample for sample in no_target
                 if Path(sample["fixed_entry"]["dataset_root"]).resolve() == configured_root]
    if not samples or not no_target:
        raise RuntimeError("immutable hard-risk and no-target train windows are required")
    selected = {args.only: MATRIX[args.only]} if args.only else MATRIX
    results = [run(config, name, components, samples, no_target, steps,
                   int(args.seed if args.seed is not None else config["random_seed"]),
                   objective=objective) for name, components in selected.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "PASS", "fixed_hard_risk": str((ROOT / "diagnostics/fixed_hard_risk.json").resolve()),
               "same_initialization": config["initialization_checkpoint"],
               "same_batch_order": True, "dynamic_loss": config["dynamic_loss"],
               "dynamic_objective": config["dynamic_objective"], "results": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
