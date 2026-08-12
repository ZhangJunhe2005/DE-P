#!/usr/bin/env python3
"""Fail-closed validation for the immutable Route-A V4.7 dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.validate_route_a_v4_6_dataset as validator
from tools.prepare_route_a_v4_7_maps import (
    PILLAR_DISTRIBUTION_CONTRACT,
    load_profiles,
)


RAW = ROOT / "data/route_a_v4_7_raw_static"
DERIVED = ROOT / "data/route_a_v4_7_static_yopo"
REPORT = ROOT / "reports/route_a_v4_7_dataset_validation.json"
CONFIG = ROOT / "configs/route_a_v4_7_raw_static_resolved.yaml"
RAW_VERSION = "route_a_v4_7_raw_static_v1"
DERIVED_VERSION = "route_a_v4_7_static_yopo"


def validate_profile_contract():
    profiles = {(row["map_type"], row["size_class"]): row
                for row in load_profiles()}
    for size, count in (("large", 100), ("narrow", 25)):
        pillar = profiles[("pillar", size)]
        if (int(pillar["obstacle_number"]) != count
                or float(pillar["width_min"]) != 0.6
                or float(pillar["width_max"]) != 1.5):
            raise RuntimeError(f"V4.7 {size} pillar profile mismatch")
        wall = profiles[("wall", size)]
        if (int(wall["wall_number"]) != count
                or float(wall["wall_width_min"]) != 0.5
                or float(wall["wall_width_max"]) != 6.0
                or float(wall["wall_thick"]) != 0.5
                or int(wall["wall_ceiling"]) != 1):
            raise RuntimeError(f"V4.7 {size} wall profile mismatch")


def validate_pillar_distribution_contract():
    expected = PILLAR_DISTRIBUTION_CONTRACT["version"]
    map_root = ROOT / "data/route_a_v4_7_mixed_maps"
    for split in ("train", "valid"):
        path = map_root / f"manifests/{split}_maps.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        pillars = [
            row for row in document.get("maps", [])
            if row.get("map_type") == "pillar"
        ]
        if not pillars:
            raise RuntimeError(f"V4.7 {split} has no pillar maps")
        for row in pillars:
            contract = row.get("pillar_distribution_contract", {})
            metrics = row.get("pillar_distribution_metrics", {})
            if contract.get("version") != expected:
                raise RuntimeError(
                    "V4.7 pillar distribution contract missing: "
                    f"{row.get('map_uuid')}"
                )
            if metrics.get("contract_version") != expected \
                    or metrics.get("passed") is not True:
                raise RuntimeError(
                    f"V4.7 pillar distribution failed: {row.get('map_uuid')}"
                )


def main():
    validate_profile_contract()
    validate_pillar_distribution_contract()
    # Reuse the complete V4.6 integrity validator with versioned roots and
    # identities.  No V4.6 file or dataset is modified.
    validator.RAW = RAW
    validator.DERIVED = DERIVED
    validator.REPORT = REPORT
    validator.CONFIG = CONFIG
    validator.RAW_VERSION = RAW_VERSION
    validator.DERIVED_VERSION = DERIVED_VERSION
    validator.load_profiles = load_profiles
    validator.main()


if __name__ == "__main__":
    main()
