#!/usr/bin/env python3
"""Generate a frozen, annex-free RR1 gap-1 representation audit corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4rr1"
DEFAULT_OUTPUT = ROOT / "data/phase8_natural_representation_audit_v1"
sys.path.insert(0, str(ROOT))

from authoritative_dataset.continuous_v1 import (
    ContinuousState, certify_curve,
)
from authoritative_dataset.cuda_renderer_v1 import (
    CudaAuthorityRenderer, RENDERER_VERSION,
)
from authoritative_dataset.dynamic_motion_v2 import (
    _dense_times, actor_position, load_motion_contract,
)
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_constructor_v2_1 import (
    natural_occlusion_pattern,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    ACTOR_RADIUS_M, CONSTRUCTOR_VERSION,
    build_occluded_actor_specs_v2_2,
    sample_occlusion_uav_sequence_v2_2,
)


CORPUS_VERSION = "natural_depth_representation_audit_corpus_v1"
GENERATOR_VERSION = "phase8jqv2_4rr1_corpus_generator_v1"
SENSOR = {
    "height": 96, "width": 160,
    "intrinsics": [80.0, 80.0, 80.0, 45.0],
    "min_depth_m": .1, "max_depth_m": 20.0,
    "ray_step_m": .1, "frame_period_ns": 100_000_000,
}
NATURAL_TYPES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-maps", type=int, default=30)
    parser.add_argument("--target-cases", type=int, default=8)
    parser.add_argument("--trials-per-map", type=int, default=3)
    parser.add_argument("--maximum-camera-candidates", type=int, default=4096)
    parser.add_argument("--maximum-actor-candidates", type=int, default=512)
    parser.add_argument("--trial-timeout-seconds", type=int, default=45)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class TrialTimeout(TimeoutError):
    pass


class bounded_trial:
    def __init__(self, seconds):
        self.seconds = int(seconds)
        self.previous = None

    def __enter__(self):
        if self.seconds > 0:
            self.previous = signal.signal(
                signal.SIGALRM,
                lambda signum, frame: (_ for _ in ()).throw(
                    TrialTimeout(
                        f"trajectory trial exceeded {self.seconds}s"
                    )
                ),
            )
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.previous is not None:
            signal.signal(signal.SIGALRM, self.previous)
        return False


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=json_default
    ).encode()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default)
        + "\n"
    )
    os.replace(temporary, path)


def camera_variant(uav, backend, variant, frame_times):
    positions = np.asarray(uav.position_world, dtype=np.float64).copy()
    yaws = np.asarray(uav.yaw, dtype=np.float64).copy()
    if variant == "static":
        pass
    else:
        forward = np.asarray(
            [np.cos(yaws[0]), np.sin(yaws[0]), 0.0], dtype=np.float64
        )
        lateral = np.asarray([-forward[1], forward[0], 0.0])
        if variant == "lateral_linear":
            offsets = np.linspace(-.03, .03, len(frame_times))
            positions += offsets[:, None] * lateral[None, :]
        elif variant == "retreat_linear":
            offsets = np.linspace(0.0, .06, len(frame_times))
            positions -= offsets[:, None] * forward[None, :]
        else:
            raise ValueError(variant)
    duration = float(frame_times[-1])
    start, end = positions[0].copy(), positions[-1].copy()
    position_at = lambda t: start + (end-start) * (
        np.clip(float(t) / max(duration, 1e-9), 0.0, 1.0)
    )
    certificate = certify_curve(position_at, backend, duration, max_depth=14)
    if certificate.state is not ContinuousState.CERTIFIED_SAFE:
        raise RuntimeError(
            f"camera motion not continuously safe: {certificate.state}"
        )
    return positions, yaws, {
        "variant": variant,
        "continuous_certificate": asdict(certificate),
    }


def validate_case(map_row, backend, construction, positions, yaws,
                  actor_frames, frame_times, contract, camera_certificate):
    diagnostics = construction.diagnostics
    gap_start, gap_end = construction.gap_start, construction.gap_end
    if gap_start != gap_end:
        raise RuntimeError("not an exact one-frame gap")
    pattern = natural_occlusion_pattern(diagnostics, 0, contract)
    if [gap_start, gap_end] not in [
        list(value) for value in pattern["accepted_gaps"]
    ]:
        raise RuntimeError("gap is not accepted by frozen natural pattern")
    projected = diagnostics["per_actor_projected_pixel_count"][:, 0]
    visible = diagnostics["per_actor_visible_pixel_count"][:, 0]
    blocked = diagnostics["per_actor_static_blocked_pixel_count"][:, 0]
    pre_min = int(contract["occlusion"]["pre_gap_visible_min"])
    post_min = int(contract["occlusion"]["post_gap_visible_min"])
    pre = np.arange(gap_start-pre_min, gap_start)
    post = np.arange(gap_end+1, gap_end+1+post_min)
    if pre.min() < 0 or post.max() >= len(frame_times):
        raise RuntimeError("pre/post visibility guard outside sequence")
    if not np.all(visible[pre] > 0) or not np.all(visible[post] > 0):
        raise RuntimeError("pre/post direct visibility contract failed")
    if int(visible[gap_start]) != 0:
        raise RuntimeError("gap frame remains directly visible")
    if (
        int(projected[gap_start]) <= 0
        or int(blocked[gap_start]) < int(projected[gap_start])
    ):
        raise RuntimeError("gap is not full natural static occlusion")
    legal_window = np.arange(pre.min(), post.max()+1)
    for key in (
        "per_actor_outside_fov", "per_actor_behind_camera",
        "per_actor_beyond_max_depth",
    ):
        if bool(np.asarray(diagnostics[key])[legal_window, 0].any()):
            raise RuntimeError(f"illegal visibility transition: {key}")
    actor_centers = actor_frames[:, 0]
    center_distance = np.linalg.norm(actor_centers-positions, axis=1)
    immediate = np.concatenate((pre, post))
    if float(center_distance[immediate].max()) > (
        float(contract["sampling"]["maximum_detection_center_distance_m"])
        + 1e-6
    ):
        raise RuntimeError("pre/post actor exceeds detection-distance contract")
    if int(projected[immediate].min()) <= 0:
        raise RuntimeError("pre/post projected support is empty")
    actor = construction.actors[0]
    speed = float(np.linalg.norm(actor["velocity"]))
    low, high = contract["scenarios"]["occluded_but_tracked"][
        "speed_range_mps"
    ]
    if not (float(low)-1e-6 <= speed <= float(high)+1e-6):
        raise RuntimeError("actor speed outside frozen motion contract")
    if float(actor["radius_m"]) != ACTOR_RADIUS_M:
        raise RuntimeError("actor radius outside frozen motion contract")
    dense_times = np.asarray(actor["validation_times"], dtype=np.float64)
    dense_actor = actor_position(actor, dense_times)
    actor_queries = [backend.query_one(point, ACTOR_RADIUS_M) for point in dense_actor]
    if any(row["collision"] for row in actor_queries):
        raise RuntimeError("actor continuous path collision/OOB")
    return {
        "status": "PASS",
        "exact_gap_frames": 1,
        "gap_start": int(gap_start), "gap_end": int(gap_end),
        "pre_visible_frames": pre.tolist(),
        "post_visible_frames": post.tolist(),
        "projected_pixels_min_pre_post": int(projected[immediate].min()),
        "visible_pixels_min_pre_post": int(visible[immediate].min()),
        "gap_projected_pixels": int(projected[gap_start]),
        "gap_static_blocked_pixels": int(blocked[gap_start]),
        "actor_center_distance_max_pre_post_m":
            float(center_distance[immediate].max()),
        "actor_radius_m": float(actor["radius_m"]),
        "actor_speed_mps": speed,
        "actor_dense_validation_samples": len(dense_times),
        "actor_minimum_static_gap_m": float(min(
            row["minimum_gap_m"] for row in actor_queries
        )),
        "camera_certificate": camera_certificate,
        "fov_front_max_depth_legal": True,
        "natural_static_occlusion": True,
        "artificial_hiding": False,
        "annex_used": False,
        "detector_output_used_for_selection": False,
        "authority_runtime_input": False,
    }


def save_case(root, case_id, map_row, trajectory_seed, construction,
              positions, yaws, actor_frames, frame_times, validation,
              camera_motion):
    case_root = root / "cases" / case_id
    case_root.mkdir(parents=True)
    diagnostics = construction.diagnostics
    arrays = {
        "depth.npy": diagnostics["composed_depth"],
        "static_depth.npy": diagnostics["static_depth"],
        "nearest_actor_owner.npy":
            diagnostics["nearest_actor_owner"],
        "actor_near_depth.npy": diagnostics["actor_near_depth"],
        "camera_positions.npy": positions,
        "camera_yaws.npy": yaws,
        "actor_positions.npy": actor_frames[:, 0],
        "timestamps.npy": frame_times,
    }
    hashes = {}
    for name, value in arrays.items():
        np.save(case_root / name, value, allow_pickle=False)
        hashes[name] = sha(case_root / name)
    actor = construction.actors[0]
    metadata = {
        "status": "PASS", "corpus_version": CORPUS_VERSION,
        "role": "representation_audit_only",
        "case_id": case_id, "map_uuid": map_row["map_uuid"],
        "maze_type": map_row["maze_type"],
        "natural_type": NATURAL_TYPES[map_row["maze_type"]],
        "map_seed": map_row["seed"],
        "trajectory_seed": trajectory_seed,
        "authority_hash": map_row["authority_hash"],
        "occupancy_hash": map_row["occupancy_hash"],
        "authority_root": map_row["authority_root"],
        "source_map_registry": map_row["source"],
        "profile_name": map_row["profile_name"],
        "requested_gap": 1,
        "observed_gap": [
            construction.gap_start, construction.gap_end
        ],
        "camera_motion": camera_motion,
        "actor": {
            "start": np.asarray(actor["start"]).tolist(),
            "velocity": np.asarray(actor["velocity"]).tolist(),
            "radius_m": actor["radius_m"],
            "motion_profile": actor["motion_profile"],
            "motion_contract_version": actor["motion_contract_version"],
        },
        "constructor": {
            "version": CONSTRUCTOR_VERSION,
            "method": construction.method,
            "attempts": construction.attempts,
            "evidence": construction.evidence,
        },
        "renderer_version": RENDERER_VERSION,
        "geometry_certificate": validation,
        "files": hashes,
        "runtime_inputs": {
            "depth": "depth.npy",
            "camera_positions": "camera_positions.npy",
            "camera_yaws": "camera_yaws.npy",
            "timestamps": "timestamps.npy",
        },
        "offline_scoring_only": {
            "actor_positions": "actor_positions.npy",
            "actor_mask": "nearest_actor_owner.npy",
            "gap": [construction.gap_start, construction.gap_end],
        },
        "annex_used": False, "artificial_hiding": False,
        "manual_depth_overwrite": False,
        "detector_output_used_for_selection": False,
        "test_accessed": False, "blind_accessed": False,
        "formal_eligible": False,
    }
    metadata["case_hash"] = canonical_hash(metadata)
    atomic_json(case_root / "case.json", metadata)
    hashes["case.json"] = sha(case_root / "case.json")
    return {
        "case_id": case_id, "map_uuid": map_row["map_uuid"],
        "maze_type": map_row["maze_type"],
        "natural_type": NATURAL_TYPES[map_row["maze_type"]],
        "map_seed": map_row["seed"], "trajectory_seed": trajectory_seed,
        "authority_hash": map_row["authority_hash"],
        "occupancy_hash": map_row["occupancy_hash"],
        "camera_motion": camera_motion["variant"],
        "gap": [construction.gap_start, construction.gap_end],
        "artifact_hashes": hashes,
        "case_hash": metadata["case_hash"],
    }


def select_split(cases):
    if len(cases) < 6:
        return None
    ordered = sorted(cases, key=lambda row: hashlib.sha256(
        row["case_id"].encode()
    ).hexdigest())
    for selected in itertools.combinations(ordered, 6):
        if len({row["map_uuid"] for row in selected}) < 6:
            continue
        if len({row["maze_type"] for row in selected}) < 3:
            continue
        for holdout in itertools.combinations(selected, 2):
            development = [row for row in selected if row not in holdout]
            if (
                len({row["maze_type"] for row in holdout}) >= 2
                and len({row["maze_type"] for row in development}) >= 2
                and not (
                    {row["authority_hash"] for row in holdout}
                    & {row["authority_hash"] for row in development}
                )
                and not (
                    {row["trajectory_seed"] for row in holdout}
                    & {row["trajectory_seed"] for row in development}
                )
            ):
                return list(development), list(holdout), list(selected)
    return None


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for RR1 corpus generation")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite corpus: {args.output}")
    inventory = load(
        DIAGNOSTICS / "corpus_generation/allowed_map_inventory.json"
    )
    if inventory["status"] != "PASS":
        raise RuntimeError("allowed map inventory is not PASS")
    grouped = {}
    for row in inventory["maps"]:
        grouped.setdefault(int(row["maze_type"]), []).append(row)
    # Round-robin types so every bounded smoke/production prefix audits map
    # diversity instead of exhausting cave then pillar maps first.
    maps = []
    for offset in range(max(map(len, grouped.values()))):
        for maze_type in sorted(grouped):
            if offset < len(grouped[maze_type]):
                maps.append(grouped[maze_type][offset])
    maps = maps[:args.maximum_maps]
    staging = args.output.with_name(
        f".{args.output.name}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    contract = load_motion_contract(
        ROOT /
        "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    )
    frame_times = np.arange(60, dtype=np.float64) * .1
    cases, attempts = [], []
    started = time.perf_counter()
    variants = ("static", "lateral_linear", "retreat_linear")
    try:
        for map_index, map_row in enumerate(maps):
            if len(cases) >= args.target_cases and (
                args.smoke or select_split(cases)
            ):
                break
            backend = ExactAuthorityBVH(map_row["authority_root"])
            accepted = False
            for trial in range(args.trials_per_map):
                trajectory_seed = (
                    910_000_000 + map_index * 1009 + trial * 17
                    + int(map_row["seed"])
                ) % (2**32)
                rng = np.random.default_rng(trajectory_seed)
                variant = variants[(map_index + trial) % len(variants)]
                row = {
                    "map_uuid": map_row["map_uuid"],
                    "maze_type": map_row["maze_type"],
                    "trajectory_seed": trajectory_seed,
                    "camera_motion": variant,
                    "status": "FAIL",
                }
                trial_started = time.perf_counter()
                try:
                    with bounded_trial(args.trial_timeout_seconds):
                        uav = sample_occlusion_uav_sequence_v2_2(
                            backend, rng, frame_times, 1,
                            maximum_candidates=args.maximum_camera_candidates,
                        )
                        positions, yaws, camera_certificate = camera_variant(
                            uav, backend, variant, frame_times
                        )
                        construction = build_occluded_actor_specs_v2_2(
                            backend, renderer, rng, positions, yaws,
                            frame_times, contract, 1, method="random_seeded",
                            maximum_candidates=args.maximum_actor_candidates,
                        )
                        # Request the offline owner map only after constructor
                        # selection. It is never passed to a runtime candidate.
                        actor_frames = actor_position(
                            construction.actors[0], frame_times
                        )[:, None, :]
                        construction.diagnostics.update(
                            renderer.render_with_actor_diagnostics(
                                backend, positions, yaws, actor_frames,
                                [ACTOR_RADIUS_M], return_owner_map=True,
                                return_actor_near_depth=True,
                            )
                        )
                        validation = validate_case(
                            map_row, backend, construction, positions, yaws,
                            actor_frames, frame_times, contract,
                            camera_certificate,
                        )
                        case_id = (
                            f"rr1_gap1_{NATURAL_TYPES[map_row['maze_type']]}_"
                            f"{map_row['map_uuid'][:8]}_{trajectory_seed}"
                        )
                        summary = save_case(
                            staging, case_id, map_row, trajectory_seed,
                            construction, positions, yaws, actor_frames,
                            frame_times, validation, camera_certificate,
                        )
                    cases.append(summary)
                    row.update({
                        "status": "PASS", "case_id": case_id,
                        "gap": summary["gap"],
                    })
                    accepted = True
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                row["elapsed_seconds"] = time.perf_counter() - trial_started
                attempts.append(row)
                print(json.dumps(row), flush=True)
                if accepted:
                    break
        split = select_split(cases)
        status = "PASS" if split is not None else "FAIL"
        development, holdout, selected = split or ([], [], [])
        selected_ids = {row["case_id"] for row in selected}
        extra = [row for row in cases if row["case_id"] not in selected_ids]
        manifest = {
            "status": status, "corpus_version": CORPUS_VERSION,
            "generator_version": GENERATOR_VERSION,
            "generator_hash": sha(Path(__file__)),
            "constructor_version": CONSTRUCTOR_VERSION,
            "constructor_hash": sha(
                ROOT /
                "authoritative_dataset/occlusion_constructor_v2_2.py"
            ),
            "renderer_version": RENDERER_VERSION,
            "motion_contract_hash": sha(
                ROOT /
                "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
            ),
            "sensor": SENSOR, "role": "representation_audit_only",
            "accepted_case_count": len(cases),
            "selected_case_count": len(selected),
            "cases": selected,
            "unused_valid_cases": extra,
            "development_metadata": [{
                key: row[key] for key in (
                    "case_id", "map_uuid", "maze_type", "natural_type",
                    "map_seed", "trajectory_seed", "authority_hash",
                    "occupancy_hash", "camera_motion", "artifact_hashes",
                )
            } for row in development],
            "sealed_holdout_metadata": [{
                key: row[key] for key in (
                    "case_id", "map_uuid", "maze_type", "natural_type",
                    "map_seed", "trajectory_seed", "authority_hash",
                    "occupancy_hash", "camera_motion", "artifact_hashes",
                )
            } for row in holdout],
            "attempt_count": len(attempts),
            "map_count": len({row["map_uuid"] for row in selected}),
            "maze_type_count": len({
                row["maze_type"] for row in selected
            }),
            "seed_count": len({
                row["trajectory_seed"] for row in selected
            }),
            "camera_motion_variants": sorted({
                row["camera_motion"] for row in selected
            }),
            "annex_used": False, "new_maps_generated": False,
            "historical_cases_modified": False,
            "detector_output_used_for_selection": False,
            "tf1_sealed_holdout_accessed": False,
            "production_test_accessed": False, "blind_accessed": False,
            "formal_generation_started": False,
            "training_started": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        manifest["root_manifest_hash"] = canonical_hash(manifest)
        atomic_json(staging / "manifest.json", manifest)
        atomic_json(staging / "generation_attempts.json", attempts)
        os.replace(staging, args.output)
    except Exception:
        if staging.exists():
            failed = args.output.with_name(
                f"{args.output.name}_failed_{uuid.uuid4().hex[:8]}"
            )
            os.replace(staging, failed)
            print(json.dumps({"failed_artifacts": str(failed)}))
        raise
    if not args.smoke:
        generation = {
            "status": manifest["status"],
            "corpus_root": str(args.output),
            "root_manifest_hash": manifest["root_manifest_hash"],
            "accepted_case_count": manifest["accepted_case_count"],
            "selected_case_count": manifest["selected_case_count"],
            "map_count": manifest["map_count"],
            "maze_type_count": manifest["maze_type_count"],
            "seed_count": manifest["seed_count"],
            "camera_motion_variants": manifest["camera_motion_variants"],
            "attempt_count": manifest["attempt_count"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "annex_used": False, "new_maps_generated": False,
            "detector_output_used_for_selection": False,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
        atomic_json(
            REPORTS / "phase8jqv2_4rr1_natural_corpus_generation.json",
            generation,
        )
        atomic_json(
            DIAGNOSTICS /
            "corpus_generation/generation_attempts.json", attempts
        )
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output), "accepted": len(cases),
        "selected": len(selected), "maps": manifest["map_count"],
        "types": manifest["maze_type_count"],
        "camera_motion_variants": manifest["camera_motion_variants"],
    }, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
