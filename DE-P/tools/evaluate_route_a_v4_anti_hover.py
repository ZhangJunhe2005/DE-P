#!/usr/bin/env python3
"""Evaluate checkpoint selection behavior before any closed-loop promotion."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from policy.static_yopo_wide_state_v1 import StaticYOPOWideStateDatasetV1
from policy.static_yopo_training_v1 import MixedSceneStaticYOPOV1


DERIVED = ROOT / "data/route_a_v4_static_yopo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument(
        "--report-only", action="store_true",
        help="record a baseline without treating its gates as promotion gates",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    base = StaticYOPODatasetV1(DERIVED, "validation", mmap_cache_size=8)
    dataset = StaticYOPOWideStateDatasetV1(base, seed=82502)
    count = min(int(args.samples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )
    authority = json.loads(
        (DERIVED / "manifests/map_authority.json").read_text()
    )
    map_type = {int(row["map_id"]): row["map_type"] for row in authority["maps"]}
    model = MixedSceneStaticYOPOV1(args.checkpoint).to(device).eval()
    values = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for batch in loader:
            depth = batch["depth"].to(device, non_blocking=True)
            observation = batch["observation"].to(device, non_blocking=True)
            endstate, score = model(depth, observation)
            candidates = endstate.permute(0, 2, 3, 1).reshape(-1, 15, 9)
            selected_index = score.reshape(-1, 15).argmin(dim=1)
            selected = candidates[
                torch.arange(len(candidates), device=device), selected_index
            ]
            distance = selected[:, :3].norm(dim=1).cpu().numpy()
            speed = selected[:, 3:6].norm(dim=1).cpu().numpy()
            hover = distance < 0.75
            for name, array in (
                ("endpoint_distance_m", distance),
                ("endpoint_speed_mps", speed),
                ("hover", hover),
            ):
                values[name].extend(array.tolist())
            for index, external_id in enumerate(batch["map_id"].tolist()):
                name = map_type[int(external_id)]
                by_type[name]["endpoint_distance_m"].append(float(distance[index]))
                by_type[name]["endpoint_speed_mps"].append(float(speed[index]))
                by_type[name]["hover"].append(bool(hover[index]))

    def summarize(group):
        distance = np.asarray(group["endpoint_distance_m"], dtype=float)
        speed = np.asarray(group["endpoint_speed_mps"], dtype=float)
        hover = np.asarray(group["hover"], dtype=bool)
        return {
            "samples": int(len(distance)),
            "selected_endpoint_distance_mean_m": float(distance.mean()),
            "selected_endpoint_distance_p10_m": float(np.quantile(distance, 0.10)),
            "selected_endpoint_speed_mean_mps": float(speed.mean()),
            "hover_selection_rate": float(hover.mean()),
        }

    metrics = summarize(values)
    gates = {
        "endpoint_distance_mean": metrics[
            "selected_endpoint_distance_mean_m"] >= 3.0,
        "endpoint_distance_p10": metrics[
            "selected_endpoint_distance_p10_m"] >= 1.0,
        "endpoint_speed_mean": metrics[
            "selected_endpoint_speed_mean_mps"] >= 2.0,
        "hover_selection_rate": metrics["hover_selection_rate"] <= 0.20,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "metrics": metrics,
        "by_map_type": {
            name: summarize(group) for name, group in sorted(by_type.items())
        },
        "gates": gates,
        "closed_loop_gate_still_required": True,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    if result["status"] != "PASS" and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
