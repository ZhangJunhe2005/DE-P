#!/usr/bin/env python3
"""GPU target-consistency audit for repeated no-return complete inputs only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from policy.static_yopo_training_v1 import MixedSceneStaticYOPOObjectiveV1, MixedSceneStaticYOPOV1

REPORTS = ROOT / "reports"
OLD = ROOT / "artifacts/phase8_mixed_scene_static_yopo_derived_v1_failed_cross_split_gate"


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def collate(samples, device):
    result = {}
    for key in samples[0]:
        if torch.is_tensor(samples[0][key]):
            result[key] = torch.stack([row[key] for row in samples]).to(device)
        else:
            result[key] = [row[key] for row in samples]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("target audit requires host CUDA")
    rows = json.load(open(REPORTS / "phase8jqv2_5snrectr1_no_return_rows.json"))["rows"]
    tolerance_groups = defaultdict(list)
    for row in rows:
        tolerance_groups[row["tolerance_input_hash"]].append(row)
    repeated = {key: value for key, value in tolerance_groups.items() if len(value) > 1}
    wanted = {row["sample_id"] for values in repeated.values() for row in values}
    datasets = {split: StaticYOPODatasetV1(OLD, split) for split in ("train", "validation")}
    selected = []
    used_map_ids = set()
    for split, dataset in datasets.items():
        ids = dataset.arrays["sample_id"]
        for index, raw in enumerate(ids):
            sample_id = bytes(raw).decode()
            if sample_id in wanted:
                selected.append((sample_id, split, index))
                used_map_ids.add(int(dataset.arrays["map_id"][index]))
    full_catalog = YAML(typ="safe").load(OLD.joinpath("map_catalog.yaml"))
    maps = [row for row in full_catalog["maps"] if int(row["map_id"]) in used_map_ids]
    for row in maps:
        row["static_ply"] = str(
            (OLD / "maps" / Path(row["static_ply"]).name).resolve()
        )
    catalog_path = REPORTS / "phase8jqv2_5snrectr1_target_audit_map_catalog.yaml"
    yaml = YAML()
    with open(catalog_path, "w", encoding="utf-8") as stream:
        yaml.dump({"catalog_version": 1, "maps": maps}, stream)
    device = torch.device("cuda:0")
    model = MixedSceneStaticYOPOV1(ROOT / "saved/DEP_0/epoch10.pth").to(device).eval()
    objective = MixedSceneStaticYOPOObjectiveV1(catalog_path).to(device).eval()
    outputs = {}
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            chunk = selected[start:start + args.batch_size]
            samples = [datasets[split][index] for _, split, index in chunk]
            details = objective(model, collate(samples, device))
            labels = details["score_label"].reshape(len(chunk), 15).float().cpu().numpy()
            static = details["candidate_static_cost"].reshape(len(chunk), 15).float().cpu().numpy()
            smooth = details["candidate_smooth_cost"].reshape(len(chunk), 15).float().cpu().numpy()
            guidance = details["candidate_guidance_cost"].reshape(len(chunk), 15).float().cpu().numpy()
            for offset, (sample_id, split, index) in enumerate(chunk):
                outputs[sample_id] = {
                    "split": split, "index": index,
                    "total": labels[offset].tolist(),
                    "static": static[offset].tolist(),
                    "smooth": smooth[offset].tolist(),
                    "guidance": guidance[offset].tolist(),
                }
    group_results = []
    conflict_groups = 0
    for key, group in repeated.items():
        matrix = np.asarray([outputs[row["sample_id"]]["total"] for row in group])
        static = np.asarray([outputs[row["sample_id"]]["static"] for row in group])
        safe = static <= 1.0e-6
        ranking = np.argsort(matrix, axis=1)
        max_range = float(np.ptp(matrix, axis=0).max())
        safe_disagreement = bool(np.any(safe != safe[0]))
        top1_disagreement = len(set(np.argmin(matrix, axis=1).tolist())) > 1
        ranking_disagreement = bool(np.any(ranking != ranking[0]))
        conflict = max_range > 1.0e-5 or safe_disagreement or top1_disagreement
        conflict_groups += int(conflict)
        group_results.append({
            "tolerance_input_hash": key, "samples": len(group),
            "map_uuids": sorted({row["map_uuid"] for row in group}),
            "sequences": sorted({row["sequence_id"] for row in group}),
            "maximum_candidate_cost_range": max_range,
            "maximum_target_variance": float(np.var(matrix, axis=0).max()),
            "safe_mask_disagreement": safe_disagreement,
            "top1_disagreement": top1_disagreement,
            "ranking_disagreement": ranking_disagreement,
            "conflict": conflict,
        })
    result = {
        "status": "PASS" if conflict_groups == 0 else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "samples_evaluated": len(selected),
        "tolerance_groups_evaluated": len(repeated),
        "conflict_groups": conflict_groups,
        "cost_tolerance": 1.0e-5,
        "safe_cost_tolerance": 1.0e-6,
        "irreducible_label_conflict_rate": (
            conflict_groups / len(repeated) if repeated else 0.0
        ),
        "all_repeated_groups_single_map": all(
            len(row["map_uuids"]) == 1 for row in group_results
        ),
        "runtime_gt_used": False,
        "groups": group_results,
    }
    atomic(REPORTS / "phase8jqv2_5snrectr1_target_consistency_raw.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
