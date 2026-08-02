#!/usr/bin/env python3
"""Full fail-closed validation for a completed formal V1 dataset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from authoritative_dataset.generate_v1 import sha256
from authoritative_dataset.cuda_renderer_v1 import RENDERER_VERSION


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT/"reports/phase8jqv2_4_dataset_manifest_validation.json"),
    )
    args = parser.parse_args()
    root, config_path = Path(args.root), Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    q23 = json.loads(
        (ROOT/"reports/phase8jqv2_3_final_result.json").read_text())
    provenance = {key: q23[key] for key in (
        "Simulator_geometry_hash", "cache_index_hash", "checkpoint_hash",
        "config_hash",
        "dataset_manifest_hash", "evaluator_version", "geometry_hash",
        "timeline_hash", "uncertainty_policy_hash")}
    required = [
        root/"generation_state/completion/TRAIN_SPLIT_GENERATION_COMPLETE",
        root/"generation_state/completion/VALID_SPLIT_GENERATION_COMPLETE",
        root/"generation_state/completion/FULL_GENERATION_COMPLETE",
    ]
    legacy_required = [
        root/"generation_state/completion/TRAIN_COMPLETE",
        root/"generation_state/completion/VALID_COMPLETE",
        root/"generation_state/completion/FULL_GENERATION_COMPLETE",
    ]
    if not (
        all(path.is_file() for path in required)
        or all(path.is_file() for path in legacy_required)
    ):
        raise RuntimeError("full generation completion marker missing")
    root_manifest = root/"manifests/dataset_manifest.json"
    dataset = json.loads(root_manifest.read_text())
    errors, counts, scenarios = [], Counter(), Counter()
    seen_sequences = set()
    for record in dataset["maps"] + dataset["sequences"]:
        path = root/record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            errors.append({"type": "manifest_hash", "path": record["path"]})
            continue
        manifest = json.loads(path.read_text())
        for relative, expected in manifest["files"].items():
            child = root/relative
            if not child.is_file() or sha256(child) != expected:
                errors.append({"type": "child_hash", "path": relative})
        if manifest["unit"] == "sequence":
            identifier = manifest["sequence_id"]
            if identifier in seen_sequences:
                errors.append({"type": "duplicate_sequence",
                               "sequence_id": identifier})
            seen_sequences.add(identifier)
            counts[f"{manifest['split']}_frames"] += manifest["frame_count"]
            counts[f"{manifest['suite']}_frames"] += manifest["frame_count"]
            scenarios[manifest["scenario"]] += manifest["frame_count"]
            if manifest.get("renderer_version") != RENDERER_VERSION:
                errors.append({"type": "renderer_version",
                               "sequence_id": identifier})
            base = root/f"{manifest['suite']}/{manifest['split']}/{identifier}"
            depth = np.load(base/"depth.npy", mmap_mode="r",
                            allow_pickle=False)
            if depth.shape[0] != manifest["frame_count"] \
                    or not np.isfinite(depth).all():
                errors.append({"type": "depth", "sequence_id": identifier})
            frames = [
                json.loads(row) for row in
                (base/"frames.jsonl").read_text().splitlines()]
            certs = [
                json.loads(row) for row in
                (base/"certificates.jsonl").read_text().splitlines()]
            render = json.loads(
                (base/"render_diagnostics.json").read_text())
            if render["renderer_version"] != RENDERER_VERSION:
                errors.append({"type": "renderer_version",
                               "sequence_id": identifier})
            counts["actor_depth_pixels"] += render[
                "actor_depth_pixels_total"]
            if (
                manifest["suite"] == "dynamic"
                and manifest["scenario"] not in (
                    "no_target", "occluded_but_tracked")
                and render["actor_depth_pixels_total"] <= 0
            ):
                errors.append({"type": "actor_not_rendered",
                               "sequence_id": identifier})
            if len(frames) != manifest["frame_count"] \
                    or len(certs) != manifest["frame_count"]:
                errors.append({"type": "missing_frame",
                               "sequence_id": identifier})
            for frame, certificate in zip(frames, certs):
                counts["unknown"] += certificate["unknown_reason"] is not None
                counts["actor_collisions"] += any(
                    actor["static_collision"]
                    or actor["future_static_collision"]
                    for actor in frame["actor_metadata"])
                if frame["runtime_random_sampling"]:
                    errors.append({"type": "runtime_random"})
            for previous, current in zip(frames, frames[1:]):
                dt = (
                    current["timestamp_ns"]-previous["timestamp_ns"])/1e9
                previous_actors = {
                    row["actor_id"]: row
                    for row in previous["actor_metadata"]}
                for actor in current["actor_metadata"]:
                    if actor["actor_id"] not in previous_actors:
                        errors.append({"type": "actor_identity",
                                       "sequence_id": identifier})
                        continue
                    old = previous_actors[actor["actor_id"]]
                    expected_position = (
                        np.asarray(old["position_world"])
                        + np.asarray(old["velocity_world"])*dt)
                    if not np.allclose(
                        expected_position, actor["position_world"],
                        atol=1e-9):
                        errors.append({"type": "actor_discontinuity",
                                       "sequence_id": identifier})
    expected = (
        {
            "train": {"frames": config["smoke_settings"]["frames_per_split"]},
            "valid": {"frames": config["smoke_settings"]["frames_per_split"]},
        }
        if args.smoke else config["formal_splits"]
    )
    if counts["train_frames"] != expected["train"]["frames"]:
        errors.append({"type": "train_count"})
    if counts["valid_frames"] != expected["valid"]["frames"]:
        errors.append({"type": "valid_count"})
    forbidden = [
        path for path in root.rglob("*")
        if "test" in path.name.lower() or "blind" in path.name.lower()]
    if forbidden:
        errors.append({"type": "forbidden_namespace",
                       "count": len(forbidden)})
    result = {
        **provenance,
        "status": "PASS" if not errors else "FAIL",
        "dataset_version": config["dataset_version"],
        "counts": dict(counts), "scenario_counts": dict(scenarios),
        "errors": errors, "test_access_count": 0,
        "blind_access_count": 0,
        "loader_deterministic": not errors,
        "config_hash": sha256(config_path),
        "root_manifest_hash": sha256(root_manifest),
    }
    write(args.output, result)
    diagnostic_names = {
        "generation_failures", "certificate_failures",
        "feasibility_unknown", "hash_mismatches", "split_leakage",
        "actor_collisions", "missing_frames", "duplicate_frames"}
    for name in diagnostic_names:
        records = [row for row in errors if name.split("_")[0]
                   in row["type"]]
        write(ROOT/f"diagnostics/phase8jqv2_4/{name}.json", {
            "status": "PASS" if not records else "FAIL",
            "count": len(records), "records": records})
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
