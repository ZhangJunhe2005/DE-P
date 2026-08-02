#!/usr/bin/env python3
"""Render every formal sequence that requires the actor sampling fallback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.generate_v1 import (
    generate_sequence, load_config, tasks_for,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=ROOT / "configs/phase8_authoritative_v2_generation.yaml",
        type=Path,
    )
    parser.add_argument(
        "--audit",
        default=ROOT / "reports/phase8jqv2_4_actor_sampling_audit.json",
        type=Path,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output",
        default=(
            ROOT
            / "reports/phase8jqv2_4_actor_sampling_cuda_validation.json"
        ),
        type=Path,
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "PASS":
        raise RuntimeError("actor sampling audit is not PASS")
    config = load_config(args.config)
    task_index = {}
    for split in ("train", "valid"):
        _, tasks = tasks_for(config, split)
        task_index.update((task["sequence_id"], task) for task in tasks)
    results = []
    with tempfile.TemporaryDirectory(
        prefix="phase8jqv2_4_actor_cuda_"
    ) as temporary:
        root = Path(temporary)
        for name in (
            "manifests/sequences", "generation_state/sequence_state",
            ".staging",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
        for record in audit["fallback_sequences"]:
            task = task_index[record["sequence_id"]]
            split = "train" if "_train_" in task["sequence_id"] else "valid"
            map_root = (
                Path(config["authority_source_root"])
                / f"geometry_authority/{split}/{task['map_uuid']}"
            )
            generate_sequence(
                root, config, split, task, map_root,
                f"cuda:{int(args.device)}",
            )
            base = root / task["suite"] / split / task["sequence_id"]
            diagnostic = json.loads(
                (base / "render_diagnostics.json").read_text()
            )
            frames = [
                json.loads(line)
                for line in (base / "frames.jsonl").read_text().splitlines()
            ]
            certificates = [
                json.loads(line)
                for line in (
                    base / "certificates.jsonl"
                ).read_text().splitlines()
            ]
            static_collisions = sum(
                actor["static_collision"]
                or actor["future_static_collision"]
                for frame in frames for actor in frame["actor_metadata"]
            )
            unsafe = sum(
                not certificate["dynamic_joint_safe"]
                for certificate in certificates
            )
            results.append({
                "sequence_id": task["sequence_id"],
                "scenario": task["scenario"],
                "actor_count": diagnostic["actor_count"],
                "actor_depth_pixels_total":
                    diagnostic["actor_depth_pixels_total"],
                "frames_with_actor_depth":
                    diagnostic["frames_with_actor_depth"],
                "static_or_future_collisions": static_collisions,
                "dynamic_joint_unsafe_frames": unsafe,
            })
    failures = [
        row for row in results
        if row["actor_depth_pixels_total"] <= 0
        or row["frames_with_actor_depth"] <= 0
        or row["static_or_future_collisions"] != 0
        or row["dynamic_joint_unsafe_frames"] != 0
    ]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "device": f"cuda:{int(args.device)}",
        "fallback_sequences_rendered": len(results),
        "results": results,
        "failures": failures,
        "production_test_used": False,
        "blind_used": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
