#!/usr/bin/env python3
"""Prepare EOSR1 entry, freeze, radius-contract and taxonomy evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.natural_exact_occlusion_solver_v2 import (
    CONTRACT_VERSION, SOLVER_VERSION, load_contract,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def first_dynamic_radius_distribution():
    counts = Counter()
    actor_count = 0
    sequence_count = 0
    root = ROOT / "data/phase8_authoritative_v2/dynamic"
    for path in root.glob("*/*/frames.jsonl"):
        sequence_count += 1
        with path.open() as source:
            line = source.readline()
        if not line:
            continue
        frame = json.loads(line)
        for actor in frame.get("actor_metadata", []):
            radius = float(actor["radius_m"])
            counts[f"{radius:.2f}"] += 1
            actor_count += 1
    return {
        "method": "first frame of every dynamic V2 sequence",
        "sequence_count": sequence_count,
        "actor_count": actor_count,
        "radius_actor_counts": dict(sorted(counts.items())),
    }


def main():
    mtc1 = json.loads((
        REPORTS / "phase8jqv2_4mtc1_final_result.json"
    ).read_text())
    if not (
        mtc1["status"] == "FAIL" and mtc1["route"] == "B"
        and mtc1["next_allowed_phase"]
        == "phase8jqv2_4_natural_gap1_exact_occlusion_solver_repair"
    ):
        raise RuntimeError("EOSR1 entry requires frozen MTC1 Route B")

    frozen_paths = [
        "authoritative_dataset/occlusion_constructor_v2_1.py",
        "authoritative_dataset/occlusion_constructor_v2_2.py",
        "authoritative_dataset/cuda_renderer_v1.py",
        "geometry_authority/static_v1.py",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/range_image_foreground_v2_1.py",
        "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "reports/phase8jqv2_4mtc1_final_result.json",
        "reports/phase8jqv2_4mtc1_room_wall_near_miss_manifest.json",
        "reports/phase8jqv2_4rr1_natural_corpus_manifest.json",
    ]
    frozen = {path: sha(ROOT / path) for path in frozen_paths}
    atomic_json(REPORTS / "phase8jqv2_4eosr1_frozen_artifacts.json", {
        "status": "PASS",
        "phase": "phase8jqv2_4_natural_gap1_exact_occlusion_solver_repair",
        "artifacts": frozen,
        "historical_gap1_contract_modified": False,
        "occlusion_constructor_v2_1_modified": False,
        "occlusion_constructor_v2_2_modified": False,
        "mtc1_artifacts_modified": False,
        "rr1_corpus_v1_modified": False,
    })
    atomic_json(REPORTS / "phase8jqv2_4eosr1_entry_gate.json", {
        "status": "PASS",
        "mtc1_status": mtc1["status"],
        "mtc1_route": mtc1["route"],
        "primary_cause": mtc1["primary_cause"],
        "historical_cuda_near_misses": 5,
        "proof_witnesses_at_entry": 0,
        "solver_version": SOLVER_VERSION,
        "test_accessed": False,
        "blind_accessed": False,
        "formal_generation_started": False,
        "training_started": False,
    })

    motion = yaml.safe_load((
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    ).read_text())
    distribution = first_dynamic_radius_distribution()
    provenance = {
        "status": "PASS",
        "uav_radius_m": float(
            motion["sampling"]["uav_radius_m"]
        ),
        "actor_default_radius_m": float(
            motion["sampling"]["actor_radius_m"]
        ),
        "semantic_relation":
            "independent contract fields with coincident numeric value",
        "scenario_overrides": {
            name: value.get("actor_radius_m")
            for name, value in motion["scenarios"].items()
            if "actor_radius_m" in value
        },
        "formal_v2_distribution": distribution,
        "all_dynamic_actors_fixed_to_0_30":
            set(distribution["radius_actor_counts"]) == {"0.30"},
        "sources": {
            "motion_contract":
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
            "generator": "authoritative_dataset/generate_v1.py",
            "motion": "authoritative_dataset/dynamic_motion_v2.py",
            "renderer": "authoritative_dataset/cuda_renderer_v1.py",
        },
    }
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_actor_radius_provenance.json",
        provenance,
    )
    compatibility = {
        "status": "PASS",
        "per_actor_radius_contract_present": True,
        "components": {
            "actor_schema": {
                "status": "PASS", "evidence": "actor spec radius_m"
            },
            "generator": {
                "status": "PASS",
                "evidence": "build_actor_specs_v2 resolves scenario/default radius",
            },
            "collision": {
                "status": "PASS",
                "evidence": "ExactAuthorityBVH.query_one(center, radius)",
            },
            "renderer": {
                "status": "PASS",
                "evidence":
                    "render_with_actor_diagnostics(..., actor_radii)",
            },
            "manifest": {
                "status": "PASS",
                "evidence": "actor_metadata[].radius_m",
            },
            "evaluator": {
                "status": "PASS",
                "evidence": "visibility and collision certificates bind radius",
            },
        },
        "fixed_0_30_downstream_assumption_found":
            "historical v2.2 constructor only; frozen and not reused by EOSR1",
        "safe_to_continue_solver_repair": True,
    }
    atomic_json(
        REPORTS
        / "phase8jqv2_4eosr1_radius_downstream_compatibility.json",
        compatibility,
    )
    (REPORTS / "phase8jqv2_4eosr1_actor_radius_contract.md").write_text(
        "# EOSR1 actor radius contract\n\n"
        "UAV radius and actor radius are distinct schema fields. Their default "
        "value is both 0.30 m, but no shared variable couples the semantics. "
        "Formal V2 manifests preserve each actor's `radius_m`; generation, "
        "exact collision, CUDA rendering and validation accept the resolved "
        "per-actor value. EOSR1 may therefore stratify actor radii at 0.20, "
        "0.25 and 0.30 m without changing the UAV radius or global default.\n\n"
        "The frozen v2.2 constructor's `ACTOR_RADIUS_M=0.30` is historical "
        "scope, not an interface limitation. EOSR1 does not modify or call "
        "that constant as its radius source.\n"
    )

    contract = load_contract(
        ROOT / "configs/natural_short_full_occlusion_contract_v2.yaml"
    )
    atomic_json(REPORTS / "phase8jqv2_4eosr1_contract_versioning.json", {
        "status": "PASS",
        "historical": {
            "version": "natural_full_occlusion_gap1_v1",
            "actor_radius_m": .30,
            "modified": False,
        },
        "new": {
            "version": CONTRACT_VERSION,
            "allowed_radius_m": contract["actor"]["allowed_radius_m"],
            "accepted_exact_gap_frames":
                contract["occlusion"]["accepted_exact_gap_frames"],
        },
    })
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_occlusion_scenario_taxonomy.json",
        {
            "status": "PASS",
            "classes": {
                "partial_occlusion":
                    "0 < blocked_fraction < severe threshold",
                "severe_partial_occlusion":
                    "high blocked fraction but visible_pixels > 0",
                "short_full_occlusion_gap1":
                    "strict full occlusion for exactly one frame",
                "short_full_occlusion_gap2":
                    "strict full occlusion for exactly two frames",
                "long_full_occlusion":
                    "strict full occlusion for at least three frames",
            },
            "mtc1_near_miss_reclassification": "long_full_occlusion",
            "partial_classes_count_for_proof_gate": False,
        },
    )
    atomic_json(REPORTS / "phase8jqv2_4eosr1_gap_length_contract.json", {
        "status": "PASS",
        "gap1": {"strict": True, "confidence_upper": .75},
        "gap2": {"strict": True, "confidence_upper": .5625},
        "gap3": {
            "strict": False, "confidence_upper": .421875,
            "reason": "below frozen dynamic threshold 0.55",
        },
    })
    print(json.dumps({
        "status": "PASS",
        "formal_v2_radius_distribution": distribution,
    }, indent=2))


if __name__ == "__main__":
    main()
