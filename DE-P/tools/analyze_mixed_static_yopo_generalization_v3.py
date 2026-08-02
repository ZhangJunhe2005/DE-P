#!/usr/bin/env python3
"""Read-only per-map generalization audit for a completed static YOPO run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from data.static_yopo_loader_v1 import make_static_yopo_loader_v1
from policy.static_yopo_training_v1 import (
    MixedSceneStaticYOPOObjectiveV1,
    MixedSceneStaticYOPOV1,
)


COMPONENTS = ("total", "trajectory", "score", "smoothness", "static_safety", "guidance")


def _text(value):
    return bytes(value).decode("utf-8")


def _accumulate(target, key, values, agreement):
    row = target[key]
    count = int(values["total"].numel())
    row["samples"] += count
    for name in COMPONENTS:
        row[name] += float(values[name].sum())
    row["score_top1_matches"] += int(agreement.sum())


def _finish(values):
    result = {}
    for key, row in values.items():
        count = row["samples"]
        result[key] = {
            "samples": count,
            **{name: row[name] / count for name in COMPONENTS},
            "score_top1_label_agreement": row["score_top1_matches"] / count,
        }
    return dict(sorted(result.items()))


def evaluate(model, objective, loader, dataset, type_by_uuid, device):
    totals = lambda: defaultdict(
        lambda: {"samples": 0, "score_top1_matches": 0,
                 **{name: 0.0 for name in COMPONENTS}}
    )
    by_type, by_map = totals(), totals()
    captured = {}

    def capture(_module, _inputs, output):
        captured["score"] = output[1].detach().float()

    hook = model.register_forward_hook(capture)
    offset = 0
    model.eval()
    with torch.inference_mode():
        for batch_no, batch in enumerate(loader):
            batch_size = int(batch["depth"].shape[0])
            uuids = [
                _text(value)
                for value in dataset.arrays["map_uuid"][offset:offset + batch_size]
            ]
            offset += batch_size
            batch = {
                key: (value.to(device, non_blocking=True)
                      if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=True):
                details = objective(model, batch)
            count = batch_size
            candidate_count = details["score_label"].numel() // count
            labels = details["score_label"].reshape(count, candidate_count)
            predicted = captured["score"].reshape(count, candidate_count)
            score = F.smooth_l1_loss(predicted, labels, reduction="none").mean(dim=1)
            smoothness = details["candidate_smooth_cost"].reshape(
                count, candidate_count
            ).mean(dim=1)
            static_safety = details["candidate_static_cost"].reshape(
                count, candidate_count
            ).mean(dim=1)
            guidance = details["candidate_guidance_cost"].reshape(
                count, candidate_count
            ).mean(dim=1)
            trajectory = smoothness + static_safety + guidance
            values = {
                "total": trajectory + score,
                "trajectory": trajectory,
                "score": score,
                "smoothness": smoothness,
                "static_safety": static_safety,
                "guidance": guidance,
            }
            agreement = predicted.argmin(dim=1).eq(labels.argmin(dim=1))
            for uuid in sorted(set(uuids)):
                indices = torch.tensor(
                    [index for index, value in enumerate(uuids) if value == uuid],
                    device=device,
                )
                selected = {name: value[indices] for name, value in values.items()}
                selected_agreement = agreement[indices]
                _accumulate(by_map, uuid, selected, selected_agreement)
                _accumulate(by_type, type_by_uuid[uuid], selected, selected_agreement)
            if batch_no == 0 or (batch_no + 1) % 100 == 0:
                print(json.dumps({
                    "event": "validation_audit_progress",
                    "batch": batch_no + 1, "batches": len(loader),
                }), flush=True)
    hook.remove()
    if offset != len(dataset):
        raise RuntimeError(f"validation audit length mismatch: {offset} != {len(dataset)}")
    return {"by_map_type": _finish(by_type), "by_map_uuid": _finish(by_map)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Checkpoint to audit; may be repeated. Use LABEL=initial for the "
            "unmodified legacy initialization. If omitted, retain the original "
            "v3.1 audit set."
        ),
    )
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    derived = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v3"
    authority = json.load(open(derived / "manifests/map_authority.json"))["maps"]
    type_by_uuid = {row["map_uuid"]: row["map_type"] for row in authority}
    dataset = StaticYOPODatasetV1(derived, "validation", mmap_cache_size=8)
    loader, _ = make_static_yopo_loader_v1(
        dataset, batch_size=32, seed=82502, num_workers=8, shuffle=False,
        prefetch_factor=2, pin_memory=True, drop_last=False,
    )
    device = torch.device("cuda:0")
    model = MixedSceneStaticYOPOV1(ROOT / "saved/DEP_0/epoch10.pth").to(device)
    objective = MixedSceneStaticYOPOObjectiveV1(derived / "map_catalog.yaml").to(device)
    if args.checkpoint:
        checkpoints = {}
        for specification in args.checkpoint:
            if "=" not in specification:
                raise ValueError(
                    f"--checkpoint must be LABEL=PATH, received: {specification}"
                )
            label, value = specification.split("=", 1)
            if not label or label in checkpoints:
                raise ValueError(f"invalid or duplicate checkpoint label: {label!r}")
            checkpoints[label] = (
                None if value == "initial"
                else Path(value).expanduser().resolve()
            )
    else:
        checkpoints = {
            "initial_legacy": None,
            "best_epoch_002": run / "checkpoints/best.pth",
            "last_epoch_012": run / "checkpoints/epoch_012.pth",
        }
    result = {
        "status": "PASS",
        "run": str(run),
        "dataset": str(derived),
        "validation_samples": len(dataset),
        "checkpoints": {},
    }
    initial = {key: value.detach().cpu().clone()
               for key, value in model.state_dict().items()}
    for label, checkpoint in checkpoints.items():
        if checkpoint is None:
            model.load_state_dict(initial, strict=True)
        else:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
        print(json.dumps({"event": "checkpoint_audit_start", "label": label}), flush=True)
        result["checkpoints"][label] = evaluate(
            model, objective, loader, dataset, type_by_uuid, device
        )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
