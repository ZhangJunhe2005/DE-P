#!/usr/bin/env python3
"""Independent EOSR1 CUDA and continuous-safety validator.

Only physical inputs are consumed from witness artifacts. Proposer status,
expected gaps, margins and labels are deliberately ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.natural_exact_occlusion_solver_v2 import (
    strict_full_mask,
)
from generate_phase8jqv2_4rr1_corpus import SENSOR
from run_phase8jqv2_4eosr1_solver import collision_safe, full_runs


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def array_sha(value, dtype):
    value = np.asarray(value).astype(dtype)
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".npy") as target:
        np.save(target, value, allow_pickle=False)
        target.flush()
        target.seek(0)
        return hashlib.sha256(target.read()).hexdigest()


def render(renderer, backend, artifact):
    actor = np.asarray(artifact["actor_positions"], dtype=np.float64)
    camera = np.asarray(artifact["camera_positions"], dtype=np.float64)
    yaws = np.asarray(artifact["camera_yaws"], dtype=np.float64)
    radius = float(artifact["radius_m"])
    result = renderer.render_with_actor_diagnostics(
        backend, camera, yaws, actor[:, None, :], [radius],
        return_owner_map=True, return_actor_near_depth=True,
    )
    return result, actor, camera, radius


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("independent EOSR1 validation requires host CUDA")
    manifest = json.loads((
        REPORTS / "phase8jqv2_4eosr1_proof_witness_manifest.json"
    ).read_text())
    # The manifest is used only as an inventory of physical artifact paths.
    # Its status, counts and witness labels are not consumed.
    paths = [Path(row["artifact"]) for row in manifest["witnesses"][:12]]
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    rows = []
    for path in paths[:12]:
        physical = json.loads(path.read_text())
        # Do not inspect accepted_run, strict_validation, proposer status,
        # phase result, expected gap or coverage-margin fields.
        backend = ExactAuthorityBVH(physical["authority_root"])
        first, actor, camera, radius = render(
            renderer, backend, physical
        )
        second, _, _, _ = render(renderer, backend, physical)
        runs = full_runs(strict_full_mask(first))
        strict = [
            run for run in runs
            if len(run) in (1, 2)
            and run[0] >= 4 and run[-1]+3 < len(actor)
            and np.all(np.asarray(
                first["per_actor_visible_pixel_count"]
            )[run[0]-4:run[0], 0] > 0)
            and np.all(np.asarray(
                first["per_actor_visible_pixel_count"]
            )[run[-1]+1:run[-1]+4, 0] > 0)
        ]
        illegal = any(bool(np.asarray(first[key])[:, 0].any())
                      for key in (
                          "per_actor_outside_fov",
                          "per_actor_behind_camera",
                          "per_actor_beyond_max_depth",
                      ))
        safe, certificate = collision_safe(
            backend, actor, radius, camera
        )
        depth_first = array_sha(first["composed_depth"], "<f4")
        depth_second = array_sha(second["composed_depth"], "<f4")
        owner_first = array_sha(first["nearest_actor_owner"], "<i4")
        owner_second = array_sha(second["nearest_actor_owner"], "<i4")
        radius_manifest = float(physical["radius_m"])
        passed = bool(
            strict and not illegal and safe
            and depth_first == depth_second
            and owner_first == owner_second
            and radius == radius_manifest
            and not physical.get("annex_used", False)
        )
        rows.append({
            "witness_id": physical["witness_id"],
            "status": "PASS" if passed else "FAIL",
            "map_uuid": physical["map_uuid"],
            "natural_type": physical["natural_type"],
            "radius_m": radius,
            "derived_strict_runs": strict,
            "illegal_fov_front_depth_exit": illegal,
            "continuous_safe": safe,
            "continuous_certificate": certificate,
            "depth_sha256": depth_first,
            "depth_byte_exact": depth_first == depth_second,
            "owner_sha256": owner_first,
            "owner_map_byte_exact": owner_first == owner_second,
            "annex_used": bool(physical.get("annex_used", False)),
        })
    status = "PASS" if rows and all(
        row["status"] == "PASS" for row in rows
    ) else "FAIL"
    result = {
        "status": status,
        "independent_process": True,
        "proposer_labels_read": False,
        "expected_gap_read": False,
        "coverage_margin_read": False,
        "witness_count": len(rows),
        "rows": rows,
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_independent_rerender.json",
        result,
    )
    atomic_json(
        REPORTS / "phase8jqv2_4eosr1_independent_validator.json",
        result,
    )
    print(json.dumps({
        "status": status, "witnesses": len(rows)
    }, indent=2))
    if status != "PASS":
        raise RuntimeError("EOSR1 independent validation failed")


if __name__ == "__main__":
    main()
