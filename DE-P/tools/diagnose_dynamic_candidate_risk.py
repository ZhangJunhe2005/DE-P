#!/usr/bin/env python3
"""Emit candidate/primitive-level physical risk for immutable Phase-8B windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from phase8b_common import (ROOT, collate, evaluate_batch, jsonable, load_config,
                            load_fixed_samples, make_trainer)


def ranks(values, descending=False):
    order = np.argsort(-values if descending else values, kind="stable")
    result = np.empty_like(order)
    result[order] = np.arange(len(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase8b_gradient_diagnostics.yaml")
    parser.add_argument("--fixed", type=Path, default=ROOT / "diagnostics/fixed_hard_risk.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/phase8b_gradient_diagnostics/candidate_risk")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["initialization_checkpoint"] = str(args.checkpoint.resolve())
    trainer = make_trainer(config)
    samples = load_fixed_samples(config, args.fixed)
    rows, sample_summaries = [], []
    for sample in samples:
        metrics = evaluate_batch(trainer, collate([sample]))
        details = metrics["details"]
        diagnostics = details["dynamic_diagnostics"]
        score = metrics["score"][0]
        risk = metrics["raw"][0]
        static = details["candidate_static_cost"].reshape(1, 15).detach().cpu().numpy()[0]
        guidance = details["candidate_guidance_cost"].reshape(1, 15).detach().cpu().numpy()[0]
        distance = diagnostics.candidate_min_distance[0].detach().cpu().numpy()
        clearance = diagnostics.candidate_min_clearance[0].detach().cpu().numpy()
        times = diagnostics.candidate_time_of_min_clearance[0].detach().cpu().numpy()
        risk_rank, score_rank = ranks(risk, descending=True), ranks(score)
        selected = int(score.argmin())
        fixed = sample["fixed_entry"]
        for candidate in range(15):
            rows.append({
                "sequence_id": fixed["sequence_id"], "frame_index": fixed["frame_index"],
                "map_id": fixed["map_id"], "scenario": fixed["scenario"],
                "sample_category": fixed["sample_category"], "candidate_id": candidate,
                "predicted_score": float(score[candidate]), "dynamic_cost": float(risk[candidate]),
                "static_cost": float(static[candidate]), "guidance_cost": float(guidance[candidate]),
                "minimum_distance": float(distance[candidate]),
                "minimum_clearance": float(clearance[candidate]),
                "time_of_minimum_clearance": float(times[candidate]),
                "risk_rank": int(risk_rank[candidate]), "score_rank": int(score_rank[candidate]),
                "selected_by_score": candidate == selected,
            })
        summary = jsonable(metrics)
        summary.update({key: fixed[key] for key in (
            "sequence_id", "frame_index", "map_id", "scenario", "sample_category"
        )})
        summary["most_dangerous_candidate"] = int(risk.argmax())
        summary["safest_candidate"] = int(risk.argmin())
        summary["score_selected_candidate"] = selected
        sample_summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {}
    keys = ("raw_dynamic_mean", "weighted_dynamic_mean", "dynamic_cvar", "dynamic_median",
            "dynamic_p75", "dynamic_p90", "safe_candidate_fraction",
            "high_risk_candidate_count_mean", "minimum_clearance", "top1_raw_risk",
            "top1_collision_fraction", "oracle_regret", "score_weighted_expected_risk",
            "spearman", "kendall", "static", "guidance")
    for key in keys:
        values = [item[key] for item in sample_summaries if item[key] is not None]
        aggregate[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                          "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .9))}
    payload = {"checkpoint": config["initialization_checkpoint"],
               "fixed_list": str(args.fixed.resolve()), "aggregate": aggregate,
               "samples": sample_summaries}
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "samples": len(samples), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
