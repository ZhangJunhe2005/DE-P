#!/usr/bin/env python3
"""Independent-process byte-exact CUDA rerender of MTC1 near misses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from authoritative_dataset.cuda_renderer_v1 import CudaAuthorityRenderer
from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_constructor_v2_2 import ACTOR_RADIUS_M
from generate_phase8jqv2_4rr1_corpus import SENSOR

REPORT = ROOT / "reports/phase8jqv2_4mtc1_independent_rerender.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite MTC1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required for independent rerender")
    manifest = json.loads((
        ROOT / "reports/phase8jqv2_4mtc1_room_wall_near_miss_manifest.json"
    ).read_text())
    authority_roots = {}
    for name in (
        "phase8jqv2_4ce1_new_map_manifest.json",
        "phase8jqv2_4ce1_e3_map_manifest.json",
    ):
        source = json.loads((ROOT / "reports" / name).read_text())
        authority_roots.update({
            item["map_uuid"]: item["authority_root"]
            for item in source["maps"]
        })
    renderer = CudaAuthorityRenderer(SENSOR, "cuda:0")
    rows = []
    for index, row in enumerate(manifest["candidates"]):
        backend = ExactAuthorityBVH(authority_roots[row["map_uuid"]])
        actors = np.asarray(row["actor_positions"], dtype=np.float64)[:, None, :]
        result = renderer.render_with_actor_diagnostics(
            backend,
            np.asarray(row["camera_positions"], dtype=np.float64),
            np.asarray(row["camera_yaw"], dtype=np.float64),
            actors, [ACTOR_RADIUS_M],
            return_owner_map=True, return_actor_near_depth=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="phase8-mtc1-rerender-", dir="/tmp"
        ) as temporary:
            depth = Path(temporary) / "depth.npy"
            owner = Path(temporary) / "owner.npy"
            np.save(depth, result["composed_depth"].astype("<f4"),
                    allow_pickle=False)
            np.save(owner, result["nearest_actor_owner"].astype("<i4"),
                    allow_pickle=False)
            depth_hash = sha(depth)
            owner_hash = sha(owner)
        rows.append({
            "candidate_index": index,
            "map_uuid": row["map_uuid"],
            "depth_expected_sha256": row["discrete_depth_sha256"],
            "depth_rerender_sha256": depth_hash,
            "depth_byte_exact":
                depth_hash == row["discrete_depth_sha256"],
            "owner_expected_sha256": row["discrete_owner_sha256"],
            "owner_rerender_sha256": owner_hash,
            "owner_map_byte_exact":
                owner_hash == row["discrete_owner_sha256"],
        })
    status = "PASS" if all(
        row["depth_byte_exact"] and row["owner_map_byte_exact"]
        for row in rows
    ) else "FAIL"
    atomic_new(REPORT, {
        "status": status,
        "independent_process": True,
        "renderer_device": str(renderer.device),
        "candidate_count": len(rows),
        "rows": rows,
        "proof_witness_count": 0,
        "proof_rerender_status":
            "NOT_APPLICABLE_NO_VALID_EXACT_GAP1_WITNESS",
    })
    print(json.dumps({"status": status, "candidates": len(rows)}, indent=2))
    if status != "PASS":
        raise RuntimeError("independent rerender mismatch")


if __name__ == "__main__":
    main()
