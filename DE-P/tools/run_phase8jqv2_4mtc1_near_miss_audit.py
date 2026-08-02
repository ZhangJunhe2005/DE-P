#!/usr/bin/env python3
"""Replay only CE1 room/wall candidates that historically reached CUDA."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.dynamic_motion_v2 import load_motion_contract
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_gap1_corpus_expander_v1 import (
    camera_motion, propose_at_gap_origins, propose_one,
)
from authoritative_dataset.occlusion_constructor_v2_2 import (
    ACTOR_RADIUS_M, canonical_surface_patches,
)
from generate_phase8jqv2_4rr1_corpus import SENSOR
from run_phase8jqv2_4ce1_new_map_search import ranked_candidates

REPORTS = ROOT / "reports"
OUT = ROOT / "diagnostics/phase8jqv2_4mtc1/room_wall_leaks"
TEMPORAL = ROOT / "diagnostics/phase8jqv2_4mtc1/temporal_windows"
CONTRACT_PATH = (
    ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
)


def atomic_json(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


class RecordingRenderer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.calls = []

    def render_with_actor_diagnostics(
        self, backend, positions, yaws, actor_positions, actor_radii, **kwargs
    ):
        result = self.renderer.render_with_actor_diagnostics(
            backend, positions, yaws, actor_positions, actor_radii,
            return_owner_map=True, return_actor_near_depth=True,
        )
        self.calls.append({
            "positions": np.asarray(positions).copy(),
            "yaws": np.asarray(yaws).copy(),
            "actor_positions": np.asarray(actor_positions).copy(),
            "actor_radii": list(map(float, actor_radii)),
            "diagnostics": result,
        })
        return result


def load_target_maps():
    e2 = json.loads(
        (REPORTS / "phase8jqv2_4ce1_new_map_search.json").read_text()
    )
    e3 = json.loads(
        (REPORTS / "phase8jqv2_4ce1_e3_search.json").read_text()
    )
    e2_manifest = json.loads(
        (REPORTS / "phase8jqv2_4ce1_new_map_manifest.json").read_text()
    )
    e3_manifest = json.loads(
        (REPORTS / "phase8jqv2_4ce1_e3_map_manifest.json").read_text()
    )
    manifests = {
        row["map_uuid"]: row
        for row in e2_manifest["maps"] + e3_manifest["maps"]
    }
    targets = []
    for stage, report in (("E2", e2), ("E3", e3)):
        for row in report["maps"]:
            cuda = row.get("cuda_candidates_used", row.get("cuda_used", 0))
            if int(cuda) and row["natural_type"] in ("room", "wall"):
                targets.append({
                    "stage": stage, "historical_cuda_count": int(cuda),
                    **manifests[row["map_uuid"]],
                })
    return targets


def masks(diagnostics, frame):
    near = diagnostics["actor_near_depth"][frame, 0]
    projected = np.isfinite(near)
    static_blocked = projected & (
        diagnostics["static_depth"][frame] <= near
    )
    visible = diagnostics["nearest_actor_owner"][frame] == 0
    leak = projected & ~static_blocked
    return projected, visible, static_blocked, leak


def leak_classification(kind, leak, projected):
    if not leak.any():
        return "no_pixel_leak_full_coverage"
    ys, xs = np.where(leak)
    py, px = np.where(projected)
    if not len(px):
        return "other_with_pixel_evidence"
    distances = {
        "left": float(xs.mean() - px.min()),
        "right": float(px.max() - xs.mean()),
        "top": float(ys.mean() - py.min()),
        "bottom": float(py.max() - ys.mean()),
    }
    edge = min(distances, key=distances.get)
    if kind == "room" and (
        min(distances.values()) > .18 * max(
            px.max() - px.min() + 1, py.max() - py.min() + 1
        )
    ):
        return "room_window_leak"
    return {
        "left": "wall_left_edge_leak",
        "right": "wall_right_edge_leak",
        "top": "wall_top_edge_leak",
        "bottom": "wall_bottom_edge_leak",
    }[edge]


def save_mask(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), value.astype(bool), allow_pickle=False)
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(
        path.with_suffix(".png")
    )


def summarize_call(call, row, proposal, call_index, renderer):
    diagnostics = call["diagnostics"]
    projected_counts = diagnostics[
        "per_actor_projected_pixel_count"
    ][:, 0].astype(int)
    visible_counts = diagnostics[
        "per_actor_visible_pixel_count"
    ][:, 0].astype(int)
    blocked_counts = diagnostics[
        "per_actor_static_blocked_pixel_count"
    ][:, 0].astype(int)
    candidates = np.where(projected_counts > 0)[0]
    frame = int(candidates[np.argmin(visible_counts[candidates])])
    projected, visible, blocked, leak = masks(diagnostics, frame)
    prefix = OUT / f"{row['map_uuid']}_call{call_index:02d}_frame{frame:02d}"
    save_mask(prefix.with_name(prefix.name + "_projected"), projected)
    save_mask(prefix.with_name(prefix.name + "_visible"), visible)
    save_mask(prefix.with_name(prefix.name + "_static_blocked"), blocked)
    save_mask(prefix.with_name(prefix.name + "_leak"), leak)
    np.save(
        prefix.with_name(prefix.name + "_leak_coordinates.npy"),
        np.argwhere(leak).astype(np.int32), allow_pickle=False,
    )
    depth_path = OUT / f"{row['map_uuid']}_call{call_index:02d}_depth.npy"
    owner_path = OUT / f"{row['map_uuid']}_call{call_index:02d}_owner.npy"
    np.save(
        depth_path, diagnostics["composed_depth"].astype("<f4"),
        allow_pickle=False,
    )
    np.save(
        owner_path, diagnostics["nearest_actor_owner"].astype("<i4"),
        allow_pickle=False,
    )
    partial = np.where(
        (projected_counts > 0) & (visible_counts > 0)
        & (visible_counts < projected_counts)
    )[0]
    transition_evidence = None
    if len(partial):
        transition_frame = int(
            partial[np.argmin(visible_counts[partial])]
        )
        tp, tv, tb, tl = masks(diagnostics, transition_frame)
        transition_prefix = (
            OUT / f"{row['map_uuid']}_call{call_index:02d}"
            f"_transition_frame{transition_frame:02d}"
        )
        for name, value in (
            ("projected", tp), ("visible", tv),
            ("static_blocked", tb), ("leak", tl),
        ):
            save_mask(
                transition_prefix.with_name(
                    transition_prefix.name + f"_{name}"
                ), value,
            )
        np.save(
            transition_prefix.with_name(
                transition_prefix.name + "_leak_coordinates.npy"
            ),
            np.argwhere(tl).astype(np.int32), allow_pickle=False,
        )
        transition_evidence = {
            "frame": transition_frame,
            "projected_pixels": int(tp.sum()),
            "visible_leak_pixels": int(tl.sum()),
            "static_blocked_pixels": int(tb.sum()),
            "classification": leak_classification(
                row["natural_type"], tl, tp
            ),
            "artifact_prefix": str(transition_prefix),
        }
    # Dense time replay over the complete 60-frame horizon. This separates a
    # local pixel event from all other full-coverage intervals in the sequence.
    frame_times = np.arange(60, dtype=np.float64) * .1
    dense_times = np.arange(0.0, frame_times[-1] + .00001, .005)
    actor = call["actor_positions"][:, 0]
    velocity = (actor[1] - actor[0]) / .1
    dense_actor = (
        actor[0][None, :] + dense_times[:, None] * velocity
    )[:, None, :]
    positions = call["positions"]
    yaws = call["yaws"]
    dense_positions = np.column_stack([
        np.interp(dense_times, frame_times, positions[:, axis])
        for axis in range(3)
    ])
    dense_yaws = np.interp(dense_times, frame_times, yaws)
    backend = ExactAuthorityBVH(row["authority_root"])
    dense = renderer.render_with_actor_diagnostics(
        backend, dense_positions, dense_yaws, dense_actor, [ACTOR_RADIUS_M],
        return_owner_map=True, return_actor_near_depth=True,
    )
    dense_projected = dense["per_actor_projected_pixel_count"][:, 0]
    dense_visible = dense["per_actor_visible_pixel_count"][:, 0]
    dense_blocked = dense["per_actor_static_blocked_pixel_count"][:, 0]
    full = (
        (dense_projected > 0) & (dense_visible == 0)
        & (dense_blocked == dense_projected)
    )
    intervals = []
    indices = np.where(full)[0]
    if len(indices):
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        intervals = [{
            "enter_s": float(dense_times[group[0]]),
            "exit_s": float(dense_times[group[-1]] + .005),
            "duration_s": float(
                dense_times[group[-1]] - dense_times[group[0]] + .005
            ),
        } for group in groups]
    temporal_path = TEMPORAL / f"{row['map_uuid']}_call{call_index:02d}.npz"
    temporal_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporal_path, times=dense_times, projected=dense_projected,
        visible=dense_visible, static_blocked=dense_blocked,
        full_coverage=full,
    )
    patch = next(
        value for value in canonical_surface_patches(backend)
        if value.voxel_offset == proposal.patch_voxel_offset
    )
    gaps = []
    phase_categories = Counter()
    for phase in np.linspace(0.0, .1, 101, endpoint=False):
        samples = phase + np.arange(60) * .1
        samples = samples[samples <= frame_times[-1] + 1e-12]
        sample_indices = np.clip(
            np.rint(samples / .005).astype(np.int64), 0, len(full)-1
        )
        sampled_full = full[sample_indices]
        sampled_visible = dense_visible[sample_indices]
        run_lengths = []
        full_indices = np.where(sampled_full)[0]
        if len(full_indices):
            groups = np.split(
                full_indices, np.where(np.diff(full_indices) > 1)[0] + 1
            )
            run_lengths = [len(group) for group in groups]
        maximum_gap = max(run_lengths, default=0)
        gaps.append(int(maximum_gap))
        valid_gap1 = False
        insufficient_gap1 = False
        for index in full_indices:
            if (
                (index == 0 or not sampled_full[index-1])
                and (index + 1 == len(sampled_full)
                     or not sampled_full[index+1])
            ):
                if (
                    index >= 4 and index + 3 < len(sampled_visible)
                    and np.all(sampled_visible[index-4:index] > 0)
                    and np.all(sampled_visible[index+1:index+4] > 0)
                ):
                    valid_gap1 = True
                else:
                    insufficient_gap1 = True
        if valid_gap1:
            phase_categories["gap1_valid_pre4_post3"] += 1
        elif maximum_gap >= 2:
            phase_categories["gap2plus"] += 1
        elif insufficient_gap1:
            phase_categories["gap1_insufficient_pre_post"] += 1
        else:
            phase_categories["gap0"] += 1
    return {
        "stage": row["stage"],
        "map_uuid": row["map_uuid"],
        "maze_type": int(row["maze_type"]),
        "natural_type": row["natural_type"],
        "map_seed": int(row["seed"]),
        "profile": row["profile_name"],
        "resolved_parameters": row["resolved_parameters"],
        "authority_hash": row["authority_manifest_hash"],
        "patch_id": int(proposal.patch_voxel_offset),
        "patch_normal": list(map(float, proposal.patch_normal)),
        "patch_horizontal_extent_m": float(patch.horizontal_extent_m),
        "patch_vertical_extent_m": float(patch.vertical_extent_m),
        "camera_positions": call["positions"].astype(float).tolist(),
        "camera_yaw": call["yaws"].astype(float).tolist(),
        "actor_positions": actor.astype(float).tolist(),
        "actor_velocity_mps": velocity.astype(float).tolist(),
        "actor_speed_mps": float(np.linalg.norm(velocity)),
        "requested_gap_frames": 1,
        "per_frame_projected_pixels": projected_counts.tolist(),
        "per_frame_visible_pixels": visible_counts.tolist(),
        "per_frame_static_blocked_pixels": blocked_counts.tolist(),
        "least_visible_frame": frame,
        "least_visible_leak_pixels": int(leak.sum()),
        "leak_classification": leak_classification(
            row["natural_type"], leak, projected
        ),
        "leak_coordinates_file": str(
            prefix.with_name(prefix.name + "_leak_coordinates.npy")
        ),
        "transition_leak_evidence": transition_evidence,
        "discrete_depth_file": str(depth_path),
        "discrete_owner_file": str(owner_path),
        "discrete_depth_sha256": hashlib.sha256(
            depth_path.read_bytes()
        ).hexdigest(),
        "discrete_owner_sha256": hashlib.sha256(
            owner_path.read_bytes()
        ).hexdigest(),
        "dense_temporal_file": str(temporal_path),
        "continuous_full_occlusion_intervals": intervals,
        "sample_phase_gap_length_counts": dict(Counter(gaps)),
        "sample_phase_categories": dict(phase_categories),
        "full_coverage": bool(full.any()),
        "exact_one_frame_phase_possible":
            bool(phase_categories["gap1_valid_pre4_post3"]),
        "cuda_rejection_reason": (
            "raster_gap_or_pre_post_window"
            if not (1 in gaps) else "historical_discrete_phase_mismatch"
        ),
        "continuous_collision_result":
            "PASS_BEFORE_CUDA_BY_FROZEN_PROPOSER",
        "annex_used": False,
    }


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    targets = load_target_maps()
    if len(targets) != 4 or sum(
        row["historical_cuda_count"] for row in targets
    ) != 5:
        raise RuntimeError("historical CE1 CUDA near-miss count changed")
    contract = load_motion_contract(CONTRACT_PATH)
    frame_times = np.arange(60, dtype=np.float64) * .1
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    rows = []
    started = time.perf_counter()
    for target in targets:
        target["natural_type"] = (
            "room" if int(target["maze_type"]) == 6 else "wall"
        )
        backend = ExactAuthorityBVH(target["authority_root"])
        recorder = RecordingRenderer(renderer)
        proposals, candidates, scored = ranked_candidates(
            backend, frame_times
        )
        if target["stage"] == "E2":
            selected = candidates[:2]
            for proposal in selected:
                before = len(recorder.calls)
                try:
                    propose_one(
                        backend, recorder, proposal, frame_times, contract,
                        actor_candidate_budget=10,
                    )
                except Exception:
                    pass
                for index, call in enumerate(recorder.calls[before:], before):
                    rows.append(summarize_call(
                        call, target, proposal, index, renderer
                    ))
        else:
            selected = [value for _, value in scored[:4]]
            for attempt, (proposal, origin) in enumerate(zip(
                selected, (8, 12, 16, 20)
            )):
                try:
                    expanded = camera_motion(
                        proposal, "static", frame_times, backend
                    )
                except RuntimeError:
                    continue
                before = len(recorder.calls)
                try:
                    propose_at_gap_origins(
                        backend, recorder, expanded, frame_times, contract,
                        gap_origins=(origin,), maximum_candidates=5,
                    )
                except Exception:
                    pass
                for index, call in enumerate(recorder.calls[before:], before):
                    rows.append(summarize_call(
                        call, target, expanded, index, renderer
                    ))
        if len(recorder.calls) != target["historical_cuda_count"]:
            raise RuntimeError(
                f"CUDA replay mismatch for {target['map_uuid']}: "
                f"{len(recorder.calls)} != {target['historical_cuda_count']}"
            )
    manifest = {
        "status": "PASS",
        "historical_cuda_candidates_expected": 5,
        "replayed_cuda_candidates": len(rows),
        "candidate_count": len(rows),
        "candidates": rows,
        "detector_executed": False,
        "representation_executed": False,
        "tracker_executed": False,
        "test_accessed": False,
        "blind_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_room_wall_near_miss_manifest.json",
        manifest,
    )
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_room_wall_temporal_window.json", {
            "status": "PASS",
            "dense_time_step_s": .005,
            "sample_period_s": .1,
            "candidate_count": len(rows),
            "never_full_coverage": sum(
                not row["full_coverage"] for row in rows
            ),
            "exact_one_frame_possible_with_time_phase": sum(
                row["exact_one_frame_phase_possible"] for row in rows
            ),
            "candidates": [{
                key: row[key] for key in (
                    "map_uuid", "natural_type",
                    "continuous_full_occlusion_intervals",
                    "sample_phase_gap_length_counts",
                    "full_coverage", "exact_one_frame_phase_possible",
                )
            } for row in rows],
        },
    )
    print(json.dumps({
        "status": "PASS", "candidates": len(rows),
        "exact_one_frame_phase_possible": sum(
            row["exact_one_frame_phase_possible"] for row in rows
        ),
        "peak_gpu_memory_bytes": manifest["peak_gpu_memory_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
