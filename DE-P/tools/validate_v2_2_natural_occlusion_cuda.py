#!/usr/bin/env python3
"""Independent CUDA sweep over canonical no-annex development map authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer  # noqa
from authoritative_dataset.dynamic_motion_v2 import (  # noqa
    actor_position,
    load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH  # noqa
from authoritative_dataset.occlusion_constructor_v2_2 import (  # noqa
    CONSTRUCTOR_VERSION,
    build_occluded_actor_specs_v2_2,
    sample_occlusion_uav_sequence_v2_2,
)
from authoritative_dataset.perception_probe_v2 import (  # noqa
    run_frozen_perception_probe,
)


NATURAL_TYPES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite historical report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--types", default="1,2,5,6,7")
    parser.add_argument("--gaps", default="1,2,3")
    parser.add_argument("--map-index-start", type=int, default=0)
    parser.add_argument("--max-maps-per-type", type=int, default=3)
    parser.add_argument("--maximum-candidates", type=int, default=128)
    return parser.parse_args()


def identity_contract(probe):
    value = probe["occlusion"]
    return {
        "pre_gap_confirmed_dynamic": bool(
            value.get("pre_gap_confirmed_dynamic", False)
        ),
        "prediction_only_same_track": bool(
            value.get("identity_continuous", False)
        ),
        "post_gap_same_id": bool(value.get("identity_continuous", False)),
        "track_deleted": bool(value.get("track_deleted", True)),
        "replacement_track_created": bool(
            value.get("replacement_track_created", True)
        ),
        "pass": bool(
            value.get("identity_continuous", False)
            and not value.get("track_deleted", True)
            and not value.get("replacement_track_created", True)
        ),
    }


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for independent visibility")
    selected_types = {int(value) for value in args.types.split(",")}
    gaps = tuple(int(value) for value in args.gaps.split(","))
    if not selected_types <= NATURAL_TYPES.keys() or not set(gaps) <= {1, 2, 3}:
        raise ValueError("unsupported natural type or gap")
    map_set_path = (
        ROOT / "reports/occlusion_constructor_v2_2_natural_map_set.json"
    )
    map_set = json.loads(map_set_path.read_text())
    if map_set["status"] != "PASS":
        raise RuntimeError("no-annex proposer map-set audit is not PASS")
    # The proposer selected roots. From here onward, acceptance uses only
    # canonical occupancy, camera/actor states, and CUDA visibility.
    rows = [
        row for row in map_set["maps"]
        if row["eligible"] and row["maze_type"] in selected_types
    ]
    selected = []
    for maze_type in sorted(selected_types):
        group = sorted(
            (row for row in rows if row["maze_type"] == maze_type),
            key=lambda row: (row["seed"], row["map_uuid"]),
        )
        start = args.map_index_start
        selected.extend(group[start:start + args.max_maps_per_type])
    sensor = {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }
    renderer = CudaAuthorityRenderer(sensor, "cuda:0")
    contract = load_motion_contract(
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    results = []
    for map_index, map_row in enumerate(selected):
        backend = ExactAuthorityBVH(map_row["authority_root"])
        for gap in gaps:
            started = time.perf_counter()
            seed = (
                2_200_000_000 + int(map_row["seed"]) * 10 + gap
            ) % (2**32)
            rng = np.random.default_rng(seed)
            result = {
                "map_uuid": map_row["map_uuid"],
                "maze_type": map_row["maze_type"],
                "natural_type": NATURAL_TYPES[map_row["maze_type"]],
                "seed": map_row["seed"],
                "requested_gap_frames": gap,
                "geometry_status": "FAIL",
                "identity_status": "NOT_RUN",
            }
            try:
                uav = sample_occlusion_uav_sequence_v2_2(
                    backend, rng, frame_times, gap
                )
                construction = build_occluded_actor_specs_v2_2(
                    backend, renderer, rng, uav.position_world, uav.yaw,
                    frame_times, contract, gap,
                    maximum_candidates=args.maximum_candidates,
                )
                result.update({
                    "geometry_status": "PASS",
                    "constructed_gap": [
                        construction.gap_start, construction.gap_end
                    ],
                    "constructor_attempts": construction.attempts,
                    "authority_sightline_hash":
                        construction.evidence["authority_hash"],
                })
                actors = np.empty(
                    (len(frame_times), len(construction.actors), 3),
                    dtype=np.float64,
                )
                for actor_index, actor in enumerate(construction.actors):
                    actors[:, actor_index] = actor_position(actor, frame_times)
                visible = (
                    construction.diagnostics[
                        "per_actor_visible_pixel_count"
                    ][:, 0] > 0
                )
                minimum_frames = int(np.ceil(
                    contract["sampling"][
                        "minimum_sustained_dynamic_duration_s"
                    ] / .1
                ))
                probe = run_frozen_perception_probe(
                    construction.diagnostics["composed_depth"],
                    uav.position_world, uav.yaw, actors, frame_times,
                    sensor, minimum_sustained_frames=minimum_frames,
                    visibility_mask=visible,
                    require_occlusion_identity=True,
                    prediction_position_error_max_m=float(
                        contract["validation"][
                            "prediction_position_error_max_m"
                        ]
                    ),
                )
                identity = identity_contract(probe)
                result.update({
                    "identity_status":
                        "PASS" if identity["pass"] else "FAIL",
                    "identity_contract": identity,
                    "frozen_perception": probe,
                })
            except Exception as error:
                result["error"] = repr(error)
            result["elapsed_seconds"] = time.perf_counter() - started
            results.append(result)
            print(json.dumps({
                "map": result["map_uuid"], "type": result["natural_type"],
                "gap": gap, "geometry": result["geometry_status"],
                "identity": result["identity_status"],
                "seconds": result["elapsed_seconds"],
                "error": result.get("error"),
            }), flush=True)
    geometry_passes = [
        row for row in results if row["geometry_status"] == "PASS"
    ]
    identity_passes = [
        row for row in results if row["identity_status"] == "PASS"
    ]
    strict_map_keys = {
        (row["map_uuid"], row["seed"]) for row in identity_passes
    }
    strict_gaps = sorted({
        row["requested_gap_frames"] for row in identity_passes
    })
    strict_gate = (
        len(strict_map_keys) >= 3
        and strict_gaps == [1, 2, 3]
        and all(row["identity_contract"]["pass"] for row in identity_passes)
    )
    report = {
        "status": "PASS" if strict_gate else "FAIL",
        "constructor_version": CONSTRUCTOR_VERSION,
        "constructor_implementation_hash": sha256(
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py"
        ),
        "map_set_hash": map_set["map_set_hash"],
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "validator_inputs": [
            "canonical occupancy", "camera state", "actor state",
            "CUDA per-actor visibility",
        ],
        "validator_annex_provenance_read": False,
        "natural_occlusion_capable": bool(strict_gate),
        "annex_fixture_occlusion_capable": None,
        "geometry_pass_count": len(geometry_passes),
        "strict_identity_pass_count": len(identity_passes),
        "independent_map_seed_count": len(strict_map_keys),
        "strict_gaps_passed": strict_gaps,
        "required": {
            "independent_map_seed_count": 3,
            "gaps": [1, 2, 3],
            "pre_gap_confirmed_dynamic": True,
            "prediction_only_same_track": True,
            "post_gap_same_id": True,
            "no_track_birth_deletion_replacement": True,
        },
        "results": results,
        "formal_preflight": "FAIL" if not strict_gate else
            "ELIGIBLE_FOR_NEXT_REVIEW_NOT_FORMAL_GENERATION",
        "formal_generation_started": False,
        "training_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_json(ROOT / args.output, report)
    print(json.dumps({
        key: report[key] for key in (
            "status", "natural_occlusion_capable",
            "geometry_pass_count", "strict_identity_pass_count",
            "independent_map_seed_count", "strict_gaps_passed",
            "formal_preflight",
        )
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
