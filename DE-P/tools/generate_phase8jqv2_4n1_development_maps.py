#!/usr/bin/env python3
"""Generate bounded trainlike/validlike N1 maps from original YOPO profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.phase8jqv2_4m1_mixed_maps import build_one


PROFILE_PATH = ROOT / "configs/mixed_scene_map_profiles_v2.yaml"
TRAINLIKE = ROOT / "data/phase8_authoritative_v3_profile_repair_trainlike"
VALIDLIKE = ROOT / "data/phase8_authoritative_v3_profile_repair_validlike"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-maps", type=int, default=30)
    return parser.parse_args()


def resolved_profiles():
    document = yaml.safe_load(PROFILE_PATH.read_text())
    if document["profiles_version"] != "mixed_scene_map_profiles_v2":
        raise RuntimeError("profile version mismatch")
    if document["annex_allowed"]:
        raise RuntimeError("annex must be disabled")
    common = document["common"]
    rows = []
    for name, override in document["profiles"].items():
        row = {**common, **override, "profile_name": name}
        if row.get("occlusion_annex") is not False:
            raise RuntimeError(f"annex enabled in {name}")
        rows.append(row)
    rows.sort(key=lambda row: (int(row["maze_type"]), row["profile_name"]))
    return rows


def main():
    args = parse_args()
    profiles = resolved_profiles()
    planned = len(profiles) * 2
    if planned > args.maximum_maps:
        raise RuntimeError(f"bounded plan {planned} exceeds {args.maximum_maps}")
    started = time.perf_counter()
    outputs = []
    for namespace, root, seed_base in (
        ("profile_debug", TRAINLIKE, 841000),
        ("profile_holdout", VALIDLIKE, 851000),
    ):
        for index, profile in enumerate(profiles):
            seed = seed_base + int(profile["maze_type"]) * 100 + index
            row = build_one(
                root, namespace, profile, seed, index, development=True
            )
            outputs.append({
                **row, "seed_namespace": namespace,
                "dataset_root": str(root),
            })
            print(json.dumps({
                "status": row["status"], "namespace": namespace,
                "profile": profile["profile_name"], "seed": seed,
            }))
    manifest = {
        "status": "PASS" if all(row["status"] == "PASS" for row in outputs)
        else "PARTIAL",
        "version": "phase8jqv2_4n1_development_map_manifest_v1",
        "profile_version": "mixed_scene_map_profiles_v2",
        "profile_hash": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
        "map_count": len(outputs),
        "maximum_maps": args.maximum_maps,
        "profiles_per_maze_type": 3,
        "seeds_per_profile": 2,
        "seed_namespaces": ["profile_debug", "profile_holdout"],
        "maps": outputs,
        "elapsed_seconds": time.perf_counter() - started,
        "development_only": True,
        "formal_eligible": False,
        "annex_used": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    target = ROOT / "reports/phase8jqv2_4n1_profile_candidates.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "maps": len(outputs),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
