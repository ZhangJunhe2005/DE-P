#!/usr/bin/env python3
"""Proposer-side audit that selects original development maps without annex."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_ROOT = ROOT / "data/phase8_authoritative_v3_map_sweep"
OUTPUT = ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json"
NATURAL_TYPES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    rows = []
    failures = []
    for state_path in sorted((MAP_ROOT / "generation_state/maps").glob("*.json")):
        state = json.loads(state_path.read_text())
        maze_type = int(state["maze_type"])
        if maze_type not in NATURAL_TYPES:
            continue
        authority_root = Path(state["authority_root"])
        provenance_path = authority_root / "mixed_scene_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        parameters = provenance["resolved_parameters"]
        annex_keys = sorted(
            key for key in parameters if "annex" in key.lower()
        )
        annex_enabled = any(bool(parameters[key]) for key in annex_keys)
        valid = (
            state["development_only"]
            and state["split"] == "development"
            and not annex_enabled
            and state["authority_manifest_hash"]
            == provenance["authority_manifest_hash"]
        )
        if not valid:
            failures.append(state["map_uuid"])
        rows.append({
            "map_uuid": state["map_uuid"],
            "maze_type": maze_type,
            "natural_type": NATURAL_TYPES[maze_type],
            "seed": int(state["seed"]),
            "authority_root": str(authority_root),
            "authority_manifest_hash": state["authority_manifest_hash"],
            "occupancy_hash": state["occupancy_hash"],
            "annex_parameter_keys": annex_keys,
            "annex_enabled": annex_enabled,
            "eligible": valid,
        })
    counts = {
        name: sum(
            row["natural_type"] == name and row["eligible"] for row in rows
        )
        for name in NATURAL_TYPES.values()
    }
    report = {
        "status": "PASS" if (
            not failures and len(rows) == 15
            and all(value == 3 for value in counts.values())
        ) else "FAIL",
        "purpose": "proposer-side map selection only",
        "validator_may_read_this_provenance": False,
        "map_root": str(MAP_ROOT),
        "natural_map_count": len(rows),
        "counts_by_natural_type": counts,
        "maps": rows,
        "failures": failures,
        "map_set_hash": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
