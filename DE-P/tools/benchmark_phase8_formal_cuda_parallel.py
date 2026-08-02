#!/usr/bin/env python3
"""Bounded CUDA process-pool benchmark for the Formal V3 raw generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/phase8_dynamic_evidence_formal_v3_generation.json"
RESOLVED = (
    ROOT / "data/phase8_dynamic_evidence_formal_v3/"
    "protocol/resolved_raw_generation.yaml"
)


def semantic_signature(output):
    rows = []
    for path in sorted((output / "manifests/sequences").glob("*.json")):
        value = json.loads(path.read_text())
        rows.append({
            "sequence_id": value["sequence_id"],
            "files": value["files"],
        })
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--sequences", type=int, default=18)
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--scratch",
        default="/tmp/phase8_formal_cuda_parallel_benchmark",
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--report", default=(
        "reports/phase8_formal_cuda_parallel_benchmark.json"))
    args = parser.parse_args()
    if args.sequences < 6 or args.sequences % 6:
        parser.error("--sequences must be a positive multiple of six")
    if args.frames_per_sequence < 1:
        parser.error("--frames-per-sequence must be positive")
    if any(value < 1 for value in args.workers):
        parser.error("worker counts must be positive")
    contract = json.loads(CONTRACT.read_text())
    config = yaml.safe_load(RESOLVED.read_text())
    allowed = contract["formal_dynamic_actor_map_uuids"]["train"]
    maps_by_uuid = {
        row["map_uuid"]: row
        for row in config["formal_splits"]["train"]["maps"]
    }
    missing = [value for value in allowed if value not in maps_by_uuid]
    if missing:
        raise RuntimeError(f"benchmark maps missing: {missing}")
    config["formal_splits"]["train"]["maps"] = [
        maps_by_uuid[value] for value in allowed
    ]
    config["static_scenarios"] = ["normal_progress"]
    config["dynamic_scenarios"] = [
        "crossing",
        "head_on",
        "multi_target",
        "temporal_separation",
        "static_dynamic_joint_constraint",
    ]
    config["formal_dynamic_actor_map_uuids"]["train"] = allowed
    config["smoke_settings"] = {
        **config.get("smoke_settings", {}),
        "frames_per_split":
            args.sequences * args.frames_per_sequence,
        "frames_per_sequence": args.frames_per_sequence,
        "map_count": len(allowed),
    }
    scratch = Path(args.scratch).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    results = []
    reference = None
    try:
        for workers in args.workers:
            output = scratch / f"workers_{workers}"
            config_path = scratch / f"workers_{workers}.yaml"
            current = dict(config)
            current["output_root"] = str(output)
            config_path.write_text(yaml.safe_dump(current, sort_keys=False))
            started = time.perf_counter()
            subprocess.run([
                sys.executable, "-m", "authoritative_dataset.generate_v1",
                "--config", str(config_path),
                "--output", str(output),
                "--split", "train",
                "--smoke",
                "--workers", str(workers),
                "--device", str(args.device),
                "--fail-fast",
            ], check=True, cwd=ROOT)
            elapsed = time.perf_counter()-started
            signature, count = semantic_signature(output)
            if count != args.sequences:
                raise RuntimeError(
                    f"worker {workers} produced {count}/{args.sequences}")
            if reference is None:
                reference = signature
            if signature != reference:
                raise RuntimeError(
                    f"worker {workers} changed sequence semantics")
            results.append({
                "workers": workers,
                "sequences": count,
                "elapsed_seconds": elapsed,
                "sequences_per_second": count/elapsed,
                "semantic_signature": signature,
                "semantic_match": True,
            })
        baseline = results[0]["sequences_per_second"]
        for row in results:
            row["speedup_vs_first"] = (
                row["sequences_per_second"]/baseline)
        best = max(results, key=lambda row: row["sequences_per_second"])
        report = {
            "status": "PASS",
            "renderer": "canonical_occupancy_cuda_raycast_v1",
            "device": f"cuda:{args.device}",
            "results": results,
            "recommended_workers": best["workers"],
            "all_semantics_identical": True,
            "formal_generation_started": False,
            "training_started": False,
        }
        report_path = ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
