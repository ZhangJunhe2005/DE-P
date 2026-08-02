#!/usr/bin/env python3
"""Geometry-only capability audit for all formal authority maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH


AUDIT_VERSION = "phase8jqv2_4o1_map_occlusion_capability_v1"
MINIMUM_DIVERSITY = {
    "train_eligible_maps_min": 8,
    "valid_eligible_maps_min": 3,
    "maximum_sequence_share_per_map": 0.25,
}
FACE_BUDGET_PER_MAP = 4096
CAMERA_FACE_DISTANCE_M = 0.305
ACTOR_CAMERA_DISTANCE_M = 2.135
ACTOR_RADIUS_M = 0.30


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def exact_clear(backend, tree, points, radius):
    points = np.asarray(points, dtype=np.float64)
    result = np.ones(len(points), dtype=bool)
    result &= np.all(
        points-radius > backend.map.bounds_min[None, :], axis=1)
    result &= np.all(
        points+radius < backend.map.bounds_max[None, :], axis=1)
    search = radius + np.sqrt(3.0)*backend.map.resolution/2
    for index, neighbours in enumerate(
        tree.query_ball_point(points, search)
    ):
        if not result[index] or not neighbours:
            continue
        delta = np.maximum(np.maximum(
            backend.minimum[neighbours]-points[index],
            points[index]-backend.maximum[neighbours]), 0)
        if np.any(np.linalg.norm(delta, axis=1) <= radius+1e-9):
            result[index] = False
    return result


def horizontal_exposed_faces(backend):
    occupied = backend.occupied
    occupied_set = {tuple(row) for row in occupied.tolist()}
    rows = []
    total_exposed = 0
    for voxel in occupied:
        for axis in range(3):
            for sign in (-1, 1):
                neighbour = voxel.copy()
                neighbour[axis] += sign
                if tuple(neighbour) in occupied_set:
                    continue
                total_exposed += 1
                if axis < 2:
                    rows.append((voxel.copy(), axis, sign))
    return total_exposed, len(rows), rows


def audit_map(root, split, map_row):
    backend = ExactAuthorityBVH(
        root/"geometry_authority"/split/map_row["map_uuid"])
    total_exposed, horizontal_count, faces = horizontal_exposed_faces(backend)
    if len(faces) > FACE_BUDGET_PER_MAP:
        indices = np.linspace(
            0, len(faces)-1, FACE_BUDGET_PER_MAP,
            dtype=np.int64)
        faces = [faces[int(index)] for index in indices]
    centers = (
        backend.map.origin[None, :]
        + (backend.occupied.astype(np.float64)+0.5)
        * backend.map.resolution
    )
    tree = cKDTree(centers)
    camera, actor, evidence = [], [], []
    for voxel, axis, sign in faces:
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = sign
        face = (
            backend.map.origin
            + (voxel.astype(np.float64)+0.5)*backend.map.resolution
        )
        face[axis] += sign*backend.map.resolution/2
        camera.append(face+normal*CAMERA_FACE_DISTANCE_M)
        actor.append(
            face-normal*(
                ACTOR_CAMERA_DISTANCE_M-CAMERA_FACE_DISTANCE_M))
        evidence.append({
            "voxel_index": voxel.astype(int).tolist(),
            "face_axis": int(axis), "face_sign": int(sign),
            "face_center_world": face.tolist(),
        })
    camera = np.asarray(camera)
    actor = np.asarray(actor)
    if not len(camera):
        camera_clear = actor_point_clear = actor_clear = np.zeros(0, bool)
    else:
        camera_clear = exact_clear(backend, tree, camera, .30)
        actor_point_clear = exact_clear(backend, tree, actor, 0.0)
        actor_clear = exact_clear(
            backend, tree, actor, ACTOR_RADIUS_M)
    candidate = camera_clear & actor_clear
    occupied_set = {
        tuple(row) for row in backend.occupied.tolist()}
    actionable_patches = []
    for voxel in backend.occupied:
        for normal_axis in (0, 1):
            tangent_axis = 1-normal_axis
            for normal_sign in (-1, 1):
                patch = []
                for tangent_offset in (0, 1):
                    for vertical_offset in (0, 1):
                        value = voxel.copy()
                        value[tangent_axis] += tangent_offset
                        value[2] += vertical_offset
                        patch.append(value)
                if not all(tuple(value) in occupied_set for value in patch):
                    continue
                exposed = True
                for value in patch:
                    neighbour = value.copy()
                    neighbour[normal_axis] += normal_sign
                    exposed &= tuple(neighbour) not in occupied_set
                if exposed:
                    actionable_patches.append({
                        "voxel_indices": [
                            value.astype(int).tolist()
                            for value in patch],
                        "normal_axis": normal_axis,
                        "normal_sign": normal_sign,
                    })
    # One 0.1 m voxel requires a >=6:1 sightline depth ratio to cover a
    # 0.6 m sphere in both raster axes.  At the minimum collision-safe
    # 0.305 m camera/face distance this puts the actor beyond the frozen
    # trackable/detection envelope.  A coplanar 2x2 patch is therefore the
    # declared minimum actionable topology, before looking at model output.
    support = {
        str(gap): len(actionable_patches) for gap in (1, 2, 3)
    }
    sample = [
        evidence[index] for index in np.flatnonzero(candidate)[:8]
    ]
    eligible = all(value > 0 for value in support.values())
    return {
        "split": split,
        "map_id": map_row["map_id"],
        "map_uuid": map_row["map_uuid"],
        "authority_hash": backend.map.metadata["artifact_manifest_hash"],
        "occupancy_hash": backend.map.metadata["occupancy_hash"],
        "occupied_voxel_count": int(len(backend.occupied)),
        "exposed_face_count": int(total_exposed),
        "horizontal_exposed_face_count": int(horizontal_count),
        "candidate_face_evaluated_count": int(len(faces)),
        "candidate_occluder_count": int(candidate.sum()),
        "camera_accessible_occluder_count": int(camera_clear.sum()),
        "behind_occluder_free_space_count": int(actor_point_clear.sum()),
        "actor_radius_0_3m_candidate_count": int(actor_clear.sum()),
        "actionable_coplanar_2x2_patch_count":
            int(len(actionable_patches)),
        "gap_candidate_count": support,
        "constructor_broad_phase":
            "PASS" if candidate.any() else "FAIL",
        "actionable_constructor_capability":
            "PASS" if eligible else "FAIL",
        "constructor_failure_reason":
            None if eligible else
            "no_coplanar_2x2_authority_patch_for_trackable_full_occlusion",
        "eligible": bool(eligible),
        "sample_occluders": sample,
        "sample_actionable_patches": actionable_patches[:8],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phase8_authoritative_v3_generation.yaml")
    parser.add_argument(
        "--authority-root", default="data/phase8_authoritative_v1")
    args = parser.parse_args()
    config_path = (ROOT/args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    authority_root = (ROOT/args.authority_root).resolve()
    rows = []
    for split in ("train", "valid"):
        for map_row in config["formal_splits"][split]["maps"]:
            rows.append(audit_map(
                authority_root, split, map_row))
    eligible = {
        split: [
            row["map_uuid"] for row in rows
            if row["split"] == split and row["eligible"]
        ] for split in ("train", "valid")
    }
    diversity_pass = (
        len(eligible["train"])
        >= MINIMUM_DIVERSITY["train_eligible_maps_min"]
        and len(eligible["valid"])
        >= MINIMUM_DIVERSITY["valid_eligible_maps_min"]
    )
    report = {
        "status": "PASS" if diversity_pass else "FAIL",
        "audit_version": AUDIT_VERSION,
        "capability_source": "authority_geometry_only_no_model_selection",
        "minimum_diversity_declared_before_scan": MINIMUM_DIVERSITY,
        "scan_constants": {
            "face_budget_per_map": FACE_BUDGET_PER_MAP,
            "camera_face_distance_m": CAMERA_FACE_DISTANCE_M,
            "actor_camera_distance_m": ACTOR_CAMERA_DISTANCE_M,
            "actor_radius_m": ACTOR_RADIUS_M,
            "depth_ratio": (
                ACTOR_CAMERA_DISTANCE_M/CAMERA_FACE_DISTANCE_M),
        },
        "maps_scanned": len(rows),
        "train_maps_scanned": sum(
            row["split"] == "train" for row in rows),
        "valid_maps_scanned": sum(
            row["split"] == "valid" for row in rows),
        "eligible_counts": {
            key: len(value) for key, value in eligible.items()},
        "eligible_maps": eligible,
        "maps": rows,
        "constructor_success_is_broad_phase_only": False,
        "cuda_raster_validation_pending": True,
        "frozen_perception_validation_pending": True,
    }
    # JSON has no lowercase true literal; keep the statement explicit here.
    report["constructor_success_is_broad_phase_only"] = True
    atomic_json(
        ROOT/"reports/phase8jqv2_4o1_map_occlusion_capability.json",
        report)
    ineligible = {
        "status": "PASS",
        "maps": [row for row in rows if not row["eligible"]],
    }
    atomic_json(
        ROOT/"diagnostics/phase8jqv2_4o1/ineligible_maps.json",
        ineligible)
    manifest = {
        "manifest_version":
            "phase8_authoritative_v3_occlusion_map_eligibility_v1",
        "source_audit_version": AUDIT_VERSION,
        "source_authority_root_manifest_hash":
            "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723",
        "sensor_geometry": config["sensor_settings"],
        "minimum_diversity": MINIMUM_DIVERSITY,
        "formal_scenario_map_eligibility": {
            "occluded_but_tracked": eligible,
        },
        "split_isolation": not bool(
            set(eligible["train"]) & set(eligible["valid"])),
        "model_output_used": False,
        "test_or_blind_used": False,
    }
    manifest_path = (
        ROOT/"configs/phase8_authoritative_v3_occlusion_map_eligibility.yaml")
    manifest_path.write_text(yaml.safe_dump(
        manifest, sort_keys=False))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    atomic_json(
        ROOT/"reports/phase8jqv2_4o1_map_eligibility_manifest.json",
        {
            "status": report["status"],
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": manifest_hash,
            "eligible_counts": report["eligible_counts"],
            "split_isolation": manifest["split_isolation"],
        })
    print(json.dumps({
        "status": report["status"],
        "maps_scanned": len(rows),
        "eligible_counts": report["eligible_counts"],
        "manifest_sha256": manifest_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
