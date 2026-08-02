#!/usr/bin/env python3
"""Independently rerender and validate accepted RR1 corpus cases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/phase8_natural_representation_audit_v1"
REPORT = ROOT / "reports/phase8jqv2_4rr1_natural_corpus_integrity.json"
DIAGNOSTIC = (
    ROOT /
    "diagnostics/phase8jqv2_4rr1/corpus_generation/independent_validation.json"
)
sys.path.insert(0, str(ROOT))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH


SENSOR = {
    "height": 96, "width": 160,
    "intrinsics": [80.0, 80.0, 80.0, 45.0],
    "min_depth_m": .1, "max_depth_m": 20.0,
    "ray_step_m": .1, "frame_period_ns": 100_000_000,
}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite RR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for RR1 independent rerender")
    manifest = load(CORPUS / "manifest.json")
    raw = dict(manifest)
    recorded_root = raw.pop("root_manifest_hash")
    errors, cases = [], []
    if canonical_hash(raw) != recorded_root:
        errors.append("root_manifest_hash mismatch")
    accepted = (
        list(manifest["cases"])
        + list(manifest["unused_valid_cases"])
    )
    historical = {
        row["case_id"] for row in load(
            ROOT / "reports/phase8jqv2_4i1_retained_case_manifest.json"
        )["cases"]
    }
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    for summary in accepted:
        case_id = summary["case_id"]
        case_root = CORPUS / "cases" / case_id
        metadata = load(case_root / "case.json")
        case_errors = []
        for name, expected in metadata["files"].items():
            if sha(case_root / name) != expected:
                case_errors.append(f"hash mismatch: {name}")
        if case_id in historical:
            case_errors.append("historical case ID collision")
        if not case_id.startswith("rr1_gap1_"):
            case_errors.append("case ID outside RR1 namespace")
        for key in (
            "annex_used", "artificial_hiding", "manual_depth_overwrite",
            "detector_output_used_for_selection", "test_accessed",
            "blind_accessed",
        ):
            if metadata[key]:
                case_errors.append(f"forbidden flag true: {key}")
        if metadata["role"] != "representation_audit_only":
            case_errors.append("incorrect role")
        if metadata["observed_gap"][0] != metadata["observed_gap"][1]:
            case_errors.append("gap is not exactly one frame")
        depth = np.load(case_root / "depth.npy", allow_pickle=False)
        positions = np.load(
            case_root / "camera_positions.npy", allow_pickle=False
        )
        yaws = np.load(case_root / "camera_yaws.npy", allow_pickle=False)
        actors = np.load(
            case_root / "actor_positions.npy", allow_pickle=False
        )[:, None, :]
        backend = ExactAuthorityBVH(metadata["authority_root"])
        rerender = renderer.render_with_actor_diagnostics(
            backend, positions, yaws, actors,
            [metadata["actor"]["radius_m"]],
            return_owner_map=True, return_actor_near_depth=True,
        )
        depth_difference = float(np.max(np.abs(
            depth - rerender["composed_depth"]
        )))
        owner_equal = bool(np.array_equal(
            np.load(
                case_root / "nearest_actor_owner.npy", allow_pickle=False
            ),
            rerender["nearest_actor_owner"],
        ))
        gap = metadata["observed_gap"][0]
        visible = rerender["per_actor_visible_pixel_count"][:, 0]
        projected = rerender["per_actor_projected_pixel_count"][:, 0]
        blocked = rerender["per_actor_static_blocked_pixel_count"][:, 0]
        if depth_difference != 0:
            case_errors.append("CUDA depth rerender differs")
        if not owner_equal:
            case_errors.append("CUDA owner rerender differs")
        if not (
            visible[gap] == 0 and projected[gap] > 0
            and blocked[gap] >= projected[gap]
        ):
            case_errors.append("gap is not full static occlusion")
        if metadata["authority_hash"] != (
            backend.map.metadata["artifact_manifest_hash"]
        ):
            case_errors.append("authority hash mismatch")
        cases.append({
            "case_id": case_id,
            "status": "PASS" if not case_errors else "FAIL",
            "errors": case_errors,
            "depth_rerender_max_abs_difference": depth_difference,
            "owner_map_byte_exact": owner_equal,
            "gap_projected_pixels": int(projected[gap]),
            "gap_static_blocked_pixels": int(blocked[gap]),
            "authority_hash": metadata["authority_hash"],
            "maze_type": metadata["maze_type"],
            "trajectory_seed": metadata["trajectory_seed"],
            "camera_motion": metadata["camera_motion"]["variant"],
        })
        errors.extend(f"{case_id}: {error}" for error in case_errors)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "corpus_gate_status": manifest["status"],
        "root_manifest_hash": recorded_root,
        "root_manifest_hash_valid": canonical_hash(raw) == recorded_root,
        "accepted_case_count": len(accepted),
        "independent_map_count": len({
            row["map_uuid"] for row in accepted
        }),
        "maze_type_count": len({row["maze_type"] for row in accepted}),
        "trajectory_seed_count": len({
            row["trajectory_seed"] for row in accepted
        }),
        "authority_hash_unique": len({
            row["authority_hash"] for row in accepted
        }) == len(accepted),
        "cases": cases, "errors": errors,
        "renderer_device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "validator_runtime_inputs": [
            "canonical occupancy", "camera state", "actor state"
        ],
        "detector_output_read": False,
        "annex_provenance_read": False,
        "test_accessed": False, "blind_accessed": False,
    }
    write_new(REPORT, report)
    write_new(DIAGNOSTIC, report)
    print(json.dumps({
        "status": report["status"],
        "corpus_gate_status": report["corpus_gate_status"],
        "accepted": report["accepted_case_count"],
        "maps": report["independent_map_count"],
        "types": report["maze_type_count"],
        "errors": errors,
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
