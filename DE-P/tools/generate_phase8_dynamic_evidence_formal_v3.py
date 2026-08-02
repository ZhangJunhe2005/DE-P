#!/usr/bin/env python3
"""User-authorized Formal V3 generation orchestrator; never trains a model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.dynamic_evidence_v3 import (  # noqa: E402
    ExclusiveRootLock, atomic_json, save_numeric_shard, sha256_file,
)

CONFIG = ROOT / "configs/phase8_dynamic_evidence_formal_v3_generation.json"
FORMAL_ROOT = ROOT / "data/phase8_dynamic_evidence_formal_v3"
MAP_TOOL = ROOT / "tools/phase8jqv2_4m1_mixed_maps.py"


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True, cwd=ROOT)


def prepare_raw_config(root, contract):
    source = ROOT / contract["raw_base_config"]
    value = yaml.safe_load(source.read_text())
    value["formal_splits"]["train"]["frames"] = int(contract["raw_frames"]["train"])
    value["formal_splits"]["valid"]["frames"] = int(contract["raw_frames"]["valid"])
    value["evidence_authority_outputs"] = True
    value["formal_occlusion_gap_frames"] = list(
        contract["formal_occlusion_gap_frames"])
    for key in (
        "formal_occlusion_constructor",
        "formal_occlusion_map_semantic_names",
        "formal_occlusion_map_uuids",
        "formal_occlusion_maximum_proposals",
        "formal_occlusion_maximum_patch_checks",
        "formal_occlusion_actor_candidate_budget",
    ):
        value[key] = contract[key]
    value["formal_dynamic_actor_map_uuids"] = contract[
        "formal_dynamic_actor_map_uuids"]
    value["output_root"] = str(root / "raw_authoritative")
    target = root / "protocol/resolved_raw_generation.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    os.replace(temporary, target)
    return target


def component_features(depth, static, owner, index, selected_owner=None):
    height, width = depth.shape
    actor = owner >= 0 if selected_owner is None else owner == int(selected_owner)
    valid = np.isfinite(depth) & (depth > 0)
    if actor.any():
        support = actor
    else:
        gy, gx = np.gradient(depth.astype(np.float64))
        support = valid & (np.hypot(gx, gy) > 0.12)
        if not support.any():
            support[height//2-2:height//2+2, width//2-2:width//2+2] = True
    yy, xx = np.nonzero(support)
    values = np.zeros(32, dtype=np.float32)
    mask = np.zeros(32, dtype=np.bool_)
    if len(xx):
        z = depth[support].astype(np.float64)
        x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
        values[:7] = [len(z), len(z), x1-x0+1, y1-y0+1,
                      len(z)/max(1, (x1-x0+1)*(y1-y0+1)), z.min(), z.max()]
        depth_span = float(np.ptp(z))
        values[7:15] = [0.1*(x1-x0+1), 0.1*(y1-y0+1), depth_span,
                        np.linalg.norm([.1*(x1-x0+1), .1*(y1-y0+1), depth_span]),
                        .1*(x1-x0+1), .1*(y1-y0+1), max(.05, depth_span),
                        np.arctan2(max(x1-x0, y1-y0), 80.0)]
        values[15:19] = [float(np.mean((xx < 3)|(xx >= width-3)|(yy < 3)|(yy >= height-3))),
                         0.0, 1.0, len(z)/max(1.0, float((x1-x0+1)*(y1-y0+1)))]
        values[19:23] = [min(index+1, 4), 1.0, 1.0, 0.0]
        values[25:29] = [min(index+1, 4), 1.0, 0.1, 0.0]
        mask[:29] = True
    return values, mask


def extract_sequence(raw_root, output_root, manifest_path, split_contract):
    manifest = json.loads(manifest_path.read_text())
    sequence_id = manifest["sequence_id"]
    source_split = manifest["split"]
    map_uuid = manifest["map_uuid"]
    if source_split == "train":
        target_split = "train"
    else:
        target_split = ("calibration", "validation", "internal_test")[
            int(hashlib.sha256(map_uuid.encode()).hexdigest()[:8], 16) % 3]
    base = raw_root / manifest["suite"] / source_split / sequence_id
    depth = np.load(base / "depth.npy", allow_pickle=False)
    static = np.load(base / "static_depth.npy", allow_pickle=False)
    owner = np.load(base / "actor_owner.npy", allow_pickle=False)
    frames = [json.loads(row) for row in (base / "frames.jsonl").read_text().splitlines()]
    count = len(frames)
    features = np.zeros((count, 4, 32), dtype=np.float32)
    validity = np.zeros((count, 4, 32), dtype=np.bool_)
    time_mask = np.zeros((count, 4), dtype=np.bool_)
    labels = np.zeros(count, dtype=np.uint8)
    authority_codes = np.zeros(count, dtype=np.uint8)
    speeds = np.zeros(count, dtype=np.float32)
    chain_ids = np.empty(count, dtype="S64")
    chain_hashes = []
    for anchor in range(count):
        visible_owners = sorted(int(value) for value in np.unique(owner[anchor]) if value >= 0)
        selected_owner = visible_owners[anchor % len(visible_owners)] if visible_owners else None
        start = max(0, anchor-3)
        causal = list(range(start, anchor+1))
        offset = 4-len(causal)
        for slot, frame_index in enumerate(causal, offset):
            features[anchor, slot], validity[anchor, slot] = component_features(
                depth[frame_index], static[frame_index], owner[frame_index], frame_index,
                selected_owner)
            time_mask[anchor, slot] = True
        actors = frames[anchor].get("actor_metadata", [])
        selected_actor = next((row for row in actors if int(row["actor_id"]) == selected_owner), None)
        speed = float(np.linalg.norm(selected_actor["velocity_world"])) if selected_actor else max(
            (np.linalg.norm(row["velocity_world"]) for row in actors), default=0.0)
        speeds[anchor] = speed
        recent_actor_support = bool(np.any(owner[max(0, anchor-3):anchor+1] >= 0))
        if actors and recent_actor_support and speed >= 0.3:
            label = 0
            authority_codes[anchor] = 1  # L1 actor rendering/current-or-causal history
        elif anchor % 4 == 0:
            label = 1
            authority_codes[anchor] = 2  # L2 static map authority
        elif anchor % 4 == 1:
            label = 2
            authority_codes[anchor] = 3  # L3 deterministic segmentation artifact injection
            # A causal, recorded false split: halve support/pixel/compactness only at
            # the anchor. The pristine renderer remains available in raw_authoritative.
            features[anchor, -1, 1] *= 0.5
            features[anchor, -1, 18] *= 0.5
            features[anchor, -1, 25] = 2.0
        else:
            label = 3
            authority_codes[anchor] = 4  # insufficient/exclusive authority closure
        labels[anchor] = label
        identity = f"{sequence_id}:{anchor}:{map_uuid}:{target_split}"
        chain_id = hashlib.sha256(identity.encode()).hexdigest()
        chain_ids[anchor] = chain_id.encode()
        payload = features[anchor].tobytes()+validity[anchor].tobytes()+bytes([label])
        chain_hashes.append(hashlib.sha256(payload).hexdigest())
    shard = output_root / f"shards/{target_split}/{sequence_id}.npz"
    shard_hash = save_numeric_shard(shard, {
        "features": features, "validity_mask": validity,
        "time_mask": time_mask, "labels": labels, "actual_speed_mps": speeds,
        "authority_codes": authority_codes, "chain_ids": chain_ids,
    })
    unit = output_root / f"manifests/work_units/{sequence_id}.json"
    atomic_json(unit, {
        "status": "complete", "sequence_id": sequence_id,
        "source_split": source_split, "split": target_split,
        "map_uuid": map_uuid, "scenario": manifest["scenario"],
        "suite": manifest["suite"], "sample_count": count,
        "shard": str(shard.relative_to(output_root)), "shard_sha256": shard_hash,
        "chain_hashes": chain_hashes,
        "label_counts": {str(index): int(np.count_nonzero(labels == index)) for index in range(4)},
        "authority_code_legend": {"1":"L1_DYNAMIC_ACTOR_RENDERING_AUTHORITY","2":"L2_STATIC_MAP_AUTHORITY","3":"L3_SENSOR_RENDERER_CONTRACT_ARTIFACT","4":"UNKNOWN_CONFLICT_OR_INSUFFICIENT"},
    })
    return unit


def _extract_sequence_worker(arguments):
    return extract_sequence(*arguments)


def extract_all(root, contract, workers):
    raw = root / "raw_authoritative"
    manifests = sorted((raw / "manifests/sequences").glob("*.json"))
    expected = sum(int(value) for value in contract["raw_frames"].values())
    if not manifests:
        raise RuntimeError("raw generation produced no sequence manifests")
    units = [None] * len(manifests)
    pending = {}
    for index, path in enumerate(manifests, 1):
        unit_path = root / f"manifests/work_units/{json.loads(path.read_text())['sequence_id']}.json"
        if unit_path.is_file():
            unit = json.loads(unit_path.read_text())
            shard = root / unit["shard"]
            if not shard.is_file() or sha256_file(shard) != unit["shard_sha256"]:
                raise RuntimeError(f"corrupt committed work unit: {unit_path}")
            units[index-1] = unit_path
            continue
        pending[index-1] = (
            raw, root, path, contract["formal_splits"])
    completed = len(manifests)-len(pending)
    started = os.times().elapsed
    if workers == 1:
        for slot, arguments in pending.items():
            units[slot] = _extract_sequence_worker(arguments)
            completed += 1
            if completed % 250 == 0:
                print(json.dumps({
                    "status": "EXTRACTING", "units": completed,
                    "total": len(manifests), "workers_active": 1,
                }), flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_extract_sequence_worker, arguments): slot
                for slot, arguments in pending.items()
            }
            for future in as_completed(futures):
                units[futures[future]] = future.result()
                completed += 1
                if completed % 250 == 0:
                    elapsed = max(os.times().elapsed-started, 1e-9)
                    print(json.dumps({
                        "status": "EXTRACTING", "units": completed,
                        "total": len(manifests),
                        "workers_active": workers,
                        "units_per_second": completed/elapsed,
                    }), flush=True)
    if any(path is None for path in units):
        raise RuntimeError("parallel extraction left incomplete work units")
    rows = [json.loads(path.read_text()) for path in units]
    if sum(row["sample_count"] for row in rows) != expected:
        raise RuntimeError("formal chain count mismatch")
    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0)+row["sample_count"]
    manifest = {
        "status": "COMPLETE", "dataset_version": contract["dataset_version"],
        "schema_version": contract["schema_version"],
        "label_authority": contract["label_authority"],
        "sample_count": expected, "split_counts": split_counts,
        "future_holdout": "PLACEHOLDER_NOT_GENERATED",
        "normalization_statistics": "NOT_COMPUTED_BY_GENERATION",
        "work_units": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in units],
        "test_blind_production_access": False, "training_executed": False,
    }
    manifest["sample_hash_tree_root"] = hashlib.sha256(
        "".join(row["sha256"] for row in manifest["work_units"]).encode()).hexdigest()
    atomic_json(root / "manifests/dataset_manifest.json", manifest)
    atomic_json(root / "generation_state/FORMAL_V3_GENERATION_COMPLETE.json", {
        "status": "COMPLETE", "manifest_sha256": sha256_file(root / "manifests/dataset_manifest.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-formal-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not args.authorize_formal_generation:
        parser.error("explicit --authorize-formal-generation is required")
    contract = json.loads(CONFIG.read_text())
    root = FORMAL_ROOT.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError("non-empty formal root requires --resume")
    root.mkdir(parents=True, exist_ok=True)
    with ExclusiveRootLock(root):
        run([sys.executable, MAP_TOOL, "formal-maps", "--split", "train"])
        run([sys.executable, MAP_TOOL, "formal-maps", "--split", "valid"])
        resolved = prepare_raw_config(root, contract)
        raw = root / "raw_authoritative"
        for split in ("train", "valid"):
            command = [sys.executable, "-m", "authoritative_dataset.generate_v1",
                       "--config", resolved, "--output", raw, "--split", split,
                       "--workers", args.workers, "--device", args.device,
                       "--fail-fast"]
            if raw.exists():
                command.append("--resume")
            run(command)
        extract_all(root, contract, args.workers)
    print(json.dumps({"status":"PASS","root":str(root),"training_started":False}, indent=2))


if __name__ == "__main__":
    main()
