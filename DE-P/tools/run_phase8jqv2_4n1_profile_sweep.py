#!/usr/bin/env python3
"""Bounded P0/P1/P2 natural observability and frozen identity sweep."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.dynamic_motion_v2 import (
    actor_position, load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.foreground_observability_v1 import (
    certify_observability,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    build_occluded_actor_specs_v2_2,
    canonical_surface_patches,
)
from authoritative_dataset.natural_observable_occlusion_proposer_v1 import (
    propose_natural_cameras, propose_observable_actor,
)
from tools.run_phase8jqv2_4i1_retained_audit import (
    canonical_hash, trace_case,
)
from tools.run_phase8jqv2_4i1_schedule_stage import strict_identity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", choices=("profile_debug", "profile_holdout"),
                        required=True)
    parser.add_argument("--gap", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--maximum-proposals-per-map", type=int, default=8)
    parser.add_argument("--maximum-runtime-per-map", type=float, default=60.0)
    parser.add_argument("--witnesses-per-map", type=int, default=2)
    parser.add_argument("--maximum-maps", type=int, default=15)
    parser.add_argument("--maze-types", default="1,2,5,6,7")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def atomic_new(path: Path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    import torch
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is mandatory")
    manifest = json.loads(
        (ROOT / "reports/phase8jqv2_4n1_profile_candidates.json").read_text()
    )
    maps = [
        row for row in manifest["maps"]
        if (
            row["seed_namespace"] == args.namespace
            and int(row["maze_type"]) in {
                int(value) for value in args.maze_types.split(",")
            }
        )
    ]
    maps.sort(key=lambda row: (
        int(row["maze_type"]), row["profile_name"], row["map_uuid"]
    ))
    maps = maps[:args.maximum_maps]
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
    timestamps = np.arange(60, dtype=np.float64) * .1
    rows = []
    rejection_counter = Counter()
    total_started = time.perf_counter()
    for map_index, map_row in enumerate(maps):
        backend = ExactAuthorityBVH(map_row["authority_root"])
        map_started = time.perf_counter()
        witnesses = []
        proposals = []
        camera_proposals = propose_natural_cameras(
            backend, timestamps, args.gap,
            maximum_proposals=args.maximum_proposals_per_map,
        )
        for proposal_index in range(args.maximum_proposals_per_map):
            if time.perf_counter() - map_started > args.maximum_runtime_per_map:
                rejection_counter["maximum_runtime_per_map"] += 1
                break
            seed = (
                86000000 + args.gap * 1000000
                + map_index * 1000 + proposal_index
                + (0 if args.namespace == "profile_debug" else 500000)
            )
            rng = np.random.default_rng(seed)
            candidate_started = time.perf_counter()
            try:
                if proposal_index >= len(camera_proposals):
                    raise RuntimeError("bounded natural camera proposals exhausted")
                camera_proposal = camera_proposals[proposal_index]
                uav = SimpleNamespace(
                    position_world=camera_proposal.position_world,
                    yaw=camera_proposal.yaw,
                )
                construction = propose_observable_actor(
                    backend, renderer, camera_proposal, timestamps,
                    contract, args.gap, maximum_raster_candidates=768,
                )
            except Exception as error:
                reason = f"p0:{type(error).__name__}"
                rejection_counter[reason] += 1
                proposals.append({
                    "proposal_index": proposal_index, "seed": seed,
                    "p0": "FAIL", "reason": reason,
                    "error": str(error)[:800],
                })
                continue
            actor_frames = actor_position(
                construction.actors[0], timestamps
            )[:, None, :]
            diagnostics = renderer.render_with_actor_diagnostics(
                backend, uav.position_world, uav.yaw, actor_frames,
                [float(construction.actors[0]["radius_m"])],
                return_owner_map=True, return_actor_near_depth=True,
            )
            visible = diagnostics["nearest_actor_owner"] == 0
            certificate = certify_observability(
                diagnostics["composed_depth"],
                diagnostics["static_depth"],
                diagnostics["actor_near_depth"][:, 0],
                visible,
                actor_frames[:, 0],
                uav.position_world,
                uav.yaw,
                timestamps,
                construction.gap_start,
                construction.gap_end,
            )
            proposal = {
                "proposal_index": proposal_index,
                "seed": seed,
                "p0": "PASS",
                "gap_start": construction.gap_start,
                "gap_end": construction.gap_end,
                "actor_start": construction.actors[0]["start"].tolist(),
                "actor_velocity": construction.actors[0]["velocity"].tolist(),
                "actor_speed_mps": float(np.linalg.norm(
                    construction.actors[0]["velocity"]
                )),
                "camera_position": uav.position_world[0].tolist(),
                "camera_yaw": float(uav.yaw[0]),
                "camera_standoff_m": camera_proposal.camera_standoff_m,
                "camera_proposal_rank": camera_proposal.proposal_rank,
                "camera_proposal_certificate": camera_proposal.certificate,
                "observability": {
                    key: value for key, value in certificate.items()
                    if key != "frames"
                },
                "p1": "PASS" if certificate["observable"] else "FAIL",
                "elapsed_seconds": time.perf_counter() - candidate_started,
            }
            if not certificate["observable"]:
                rejection_counter["p1:observability_proxy"] += 1
                proposals.append(proposal)
                continue
            runtime_construction = SimpleNamespace(
                actors=construction.actors, diagnostics=diagnostics
            )
            case = {
                "case_id": (
                    f"n1_{args.namespace}_{map_row['map_uuid'][:8]}"
                    f"_gap{args.gap}_p{proposal_index}"
                )
            }
            trace = trace_case(
                case, runtime_construction, uav,
                actor_frames, timestamps, sensor,
            )
            identity = strict_identity(
                trace, construction.gap_start, construction.gap_end
            )
            proposal["p2"] = "PASS" if identity["pass"] else "FAIL"
            proposal["identity"] = identity
            proposal["trace_hash"] = canonical_hash(trace)
            proposal["trace_summary"] = {
                "measurement_frames": [
                    row["frame_index"] for row in trace
                    if row["detection"]["measurement_valid"]
                ],
                "dynamic_frames": [
                    row["frame_index"] for row in trace
                    if (
                        next((
                            item for item in row["track"]["tracks"]
                            if item["track_id"] == row["track"]["matched_track_id"]
                        ), {}).get("is_dynamic", False)
                    )
                ],
                "deleted_frames": [
                    row["frame_index"] for row in trace
                    if row["track"]["deleted_track_ids"]
                ],
                "replacement_frames": [
                    row["frame_index"] for row in trace
                    if row["track"]["replaced_this_frame"]
                ],
            }
            proposals.append(proposal)
            if identity["pass"]:
                witnesses.append({
                    **proposal,
                    "case_id": case["case_id"],
                    "trace": trace,
                })
                if len(witnesses) >= args.witnesses_per_map:
                    break
            else:
                rejection_counter["p2:frozen_identity"] += 1
        rows.append({
            "map_uuid": map_row["map_uuid"],
            "maze_type": map_row["maze_type"],
            "semantic_name": map_row["semantic_name"],
            "profile_name": map_row["profile_name"],
            "seed": map_row["seed"],
            "seed_namespace": args.namespace,
            "ordinary_navigation_capable":
                map_row["ordinary_navigation_capable"],
            "p0_authority_capable":
                map_row["occlusion_geometry_capable"],
            "proposal_count": len(proposals),
            "witness_count": len(witnesses),
            "status": (
                "PASS" if len(witnesses) >= args.witnesses_per_map else "FAIL"
            ),
            "proposals": proposals,
            "witnesses": witnesses,
            "elapsed_seconds": time.perf_counter() - map_started,
        })
        print(json.dumps({
            "map": map_row["map_uuid"], "type": map_row["maze_type"],
            "profile": map_row["profile_name"],
            "witnesses": len(witnesses),
            "proposals": len(proposals),
        }))
    passing = [row for row in rows if row["status"] == "PASS"]
    report = {
        "status": "PASS" if passing else "FAIL",
        "namespace": args.namespace,
        "gap": args.gap,
        "map_count": len(rows),
        "passing_map_count": len(passing),
        "passing_map_types": sorted(set(
            int(row["maze_type"]) for row in passing
        )),
        "witness_count": sum(row["witness_count"] for row in rows),
        "maximum_proposals_per_map": args.maximum_proposals_per_map,
        "maximum_runtime_per_map_seconds": args.maximum_runtime_per_map,
        "witnesses_required_per_map": args.witnesses_per_map,
        "deterministic_ordering":
            "maze_type_profile_map_uuid_then_seeded_proposal_index",
        "rejection_reasons": dict(sorted(rejection_counter.items())),
        "maps": rows,
        "elapsed_seconds": time.perf_counter() - total_started,
        "annex_used": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "training_started": False,
        "test_accessed": False,
        "blind_accessed": False,
    }
    atomic_new(ROOT / args.output, report)
    print(json.dumps({
        "status": report["status"],
        "passing_maps": report["passing_map_count"],
        "types": report["passing_map_types"],
        "witnesses": report["witness_count"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
