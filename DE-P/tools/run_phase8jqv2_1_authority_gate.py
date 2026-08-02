#!/usr/bin/env python3
"""Phase 8J-Q2.1 entry, fixture lock, and static authority Gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = Path("/home/zjh/YOPO/Simulator")
DATASET = Path("/home/zjh/YOPO/dataset")
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_1"
FIXTURE = ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1.npz"
FIXTURE_MANIFEST = (
    ROOT / "artifacts/phase8jv2s0/static_valid_fixture_v1_manifest.json"
)
REPORT_PROVENANCE = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    if (
        Path(path).parent == REPORTS
        and Path(path).name.startswith("phase8jqv2_1_")
    ):
        value = {**REPORT_PROVENANCE, **value}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def unavailable(reason, stage):
    return {
        "status": "NOT_EXECUTED_GEOMETRY_AUTHORITY_GATE_FAIL",
        "stage": stage,
        "reason": reason,
        "production_test_used": False,
        "blind_used": False,
        "training_executed": False,
    }


def main():
    v2 = json.loads((REPORTS / "phase8jqv2_final_result.json").read_text())
    provenance_keys = (
        "config_hash", "geometry_hash", "timeline_hash",
        "uncertainty_policy_hash", "Simulator_geometry_hash",
        "dataset_manifest_hash", "cache_index_hash", "checkpoint_hash",
    )
    REPORT_PROVENANCE.update({
        name: v2[name] for name in provenance_keys
    })
    REPORT_PROVENANCE["evaluator_version"] = (
        "NOT_CREATED_PHASE8JQV2_1_AUTHORITY_GATE_FAIL"
    )
    s0 = json.loads((REPORTS / "phase8jv2s0_final_result.json").read_text())
    conformance = json.loads(
        (REPORTS / "phase8jv2s0_static_continuous_conformance.json").read_text()
    )
    envelope = json.loads(
        (REPORTS / "phase8jv2s0_initial_state_envelope.json").read_text()
    )
    fixture_report = json.loads(
        (REPORTS / "phase8jv2s0_static_fixture_manifest.json").read_text()
    )
    fixture_determinism = json.loads(
        (REPORTS / "phase8jv2s0_static_fixture_determinism.json").read_text()
    )
    independent = conformance["independent_static_geometry_summary"]
    entry_checks = {
        "phase8jv2s0_status_fail": s0["status"] == "FAIL",
        "primary_cause_static_certificate": s0["primary_cause"]
        == "static_continuous_certificate_implementation",
        "static_fixture_deterministic":
            s0["static_fixture_deterministic"] is True,
        "static_pairing_repaired": s0["static_pairing_repaired"] is True,
        "static_checker_untrustworthy":
            s0["static_numerical_checker_trustworthy"] is False,
        "authorized_phase": s0["next_allowed_phase"]
        == "phase8jqv2_1_static_numerical_rebaseline",
        "network_weights_unmodified":
            s0["network_weights_modified"] is False,
        "training_unexecuted": s0["training_executed"] is False,
        "production_test_unused": s0["production_test_used"] is False,
        "blind_unused": s0["blind_used"] is False,
        "fixture_count_10000": fixture_report["window_count"] == 10000,
        "paired_c0_3683": envelope[
            "legacy_current_paired_coverage_failure"
        ]["numerator"] == 3683,
        "dense_recovered_739": conformance["comparison"][
            "current_unsafe_dense_128_safe_windows"
        ] == 739,
        "recursive_recovered_734": conformance["comparison"][
            "current_unsafe_recursive_certified_safe_windows"
        ] == 734,
        "recursive_unknown_17": conformance["comparison"][
            "recursive_unknown_trajectories"
        ] == 17,
        "pointcloud_disagreement_16_of_50": (
            independent["representative_trajectory_count"] == 50
            and independent[
                "dense_esdf_safe_but_pointcloud_collision_count"
            ] == 16
        ),
        "initial_outside_3043": envelope["classification"][
            "initially_outside_nominal_envelope"
        ]["numerator"] == 3043,
    }
    entry = {
        "status": "PASS" if all(entry_checks.values()) else "FAIL",
        "phase": "8J-Q2.1",
        "checks": entry_checks,
        "frozen_s0_hashes": {
            name: sha256(REPORTS / name) for name in (
                "phase8jv2s0_entry_gate.json",
                "phase8jv2s0_legacy_a0_validity.json",
                "phase8jv2s0_static_fixture_manifest.json",
                "phase8jv2s0_static_fixture_determinism.json",
                "phase8jv2s0_initial_state_envelope.json",
                "phase8jv2s0_static_continuous_conformance.json",
                "phase8jv2s0_final_result.json",
                "phase8jv2s0_final_recommendation.md",
                "phase8jv2s0_final_readiness.md",
            )
        },
        "network_weights_modified": False,
        "training_executed": False,
        "production_test_used": False,
        "blind_used": False,
    }
    atomic_json(REPORTS / "phase8jqv2_1_entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("Q2.1 entry Gate failed")

    fixture = np.load(FIXTURE, allow_pickle=False)
    current_image_hashes = np.asarray(
        [sha256(path) for path in fixture["image_path"]], dtype="U64"
    )
    images_match = bool(np.array_equal(
        current_image_hashes, fixture["image_sha256"]
    ))
    manifest = json.loads(FIXTURE_MANIFEST.read_text())
    catalog = YAML(typ="safe").load(ROOT / "configs/static_map_catalog.yaml")
    current_map_hashes = {
        str(row["map_id"]): sha256(row["static_ply"])
        for row in catalog["maps"]
    }
    current_pose_hashes = {
        str(map_id): sha256(DATASET / f"pose-{map_id}.csv")
        for map_id in range(10)
    }
    fixture_checks = {
        "fixture_sha256_match": sha256(FIXTURE)
        == fixture_report["fixture_sha256"],
        "fixture_manifest_sha256_match": sha256(FIXTURE_MANIFEST)
        == envelope["analysis_key_fields"][
            "static_fixture_manifest_hash"
        ],
        "semantic_hash_match": str(fixture["semantic_hash"])
        == fixture_report["semantic_hash"],
        "window_count_10000": len(fixture["fixture_id"]) == 10000,
        "all_image_hashes_match": images_match,
        "all_pose_csv_hashes_match":
            current_pose_hashes == manifest["pose_csv_hashes"],
        "all_static_map_hashes_match":
            current_map_hashes == manifest["static_map_hashes"],
        "worker_batch_determinism_pass":
            fixture_determinism["status"] == "PASS",
        "state_goal_explicit": all(
            name in fixture.files for name in (
                "velocity_body", "acceleration_body", "goal_body",
                "velocity_world", "acceleration_world", "goal_world",
            )
        ),
    }
    fixture_lock = {
        "status": "PASS" if all(fixture_checks.values()) else "FAIL",
        "checks": fixture_checks,
        "fixture": str(FIXTURE.resolve()),
        "fixture_sha256": sha256(FIXTURE),
        "fixture_manifest": str(FIXTURE_MANIFEST.resolve()),
        "fixture_manifest_sha256": sha256(FIXTURE_MANIFEST),
        "fixture_semantic_hash": str(fixture["semantic_hash"]),
        "fixture_resampled": False,
    }
    atomic_json(REPORTS / "phase8jqv2_1_fixture_lock.json", fixture_lock)
    if fixture_lock["status"] != "PASS":
        raise RuntimeError("Q2.1 fixture lock failed")

    generator = SIMULATOR / "src/src/dataset_generator.cpp"
    runtime = SIMULATOR / "src/src/test_simulator_cuda.cpp"
    grid = SIMULATOR / "src/src/sensor_simulator.cu"
    grid_header = SIMULATOR / "src/include/sensor_simulator.cuh"
    simulator_config = SIMULATOR / "src/config/config.yaml"
    generator_text = generator.read_text()
    runtime_text = runtime.read_text()
    grid_text = grid.read_text()
    static_loss = (ROOT / "loss/safety_loss.py").read_text()
    facts = {
        "dataset_raycast_grid_uses_unfiltered_generation_cloud":
            "GridMap grid_map(cloud, resolution, occupy_threshold);"
            in generator_text,
        "saved_ply_is_voxel_filtered_copy":
            "sor.setInputCloud(cloud);" in generator_text
            and "sor.filter(*filtered_cloud);" in generator_text
            and "savePointCloudAsPLY(filtered_cloud" in generator_text,
        "dataset_pose_clearance_uses_filtered_ply_nearest_point":
            "kdtree.setInputCloud(filtered_cloud);" in generator_text
            and "nearestKSearch" in generator_text,
        "simulator_runtime_static_grid_only_used_for_sensor_raycast":
            "grid_map.mapQuery(point)" in (
                SIMULATOR / "src/src/sensor_simulator.cu"
            ).read_text(),
        "simulator_has_no_static_uav_collision_query": (
            "static_collision" not in runtime_text
            and "actorCollidesWithUav" in runtime_text
            and runtime_text.count("message.collision =") == 1
        ),
        "configured_0_3_radius_is_dynamic_actor_only":
            "actorCollidesWithUav(actor, pos, uav_collision_radius)"
            in runtime_text,
        "offline_esdf_revoxelizes_saved_ply_at_0_2m":
            "self.voxel_size = 0.2" in static_loss
            and "distance_transform_edt" in static_loss,
        "simulator_grid_resolution_0_1m":
            "resolution: 0.1" in simulator_config.read_text(),
        "grid_origin_is_cloud_min_corner":
            "origin(min_pt(0), min_pt(1), min_pt(2))" in grid_text,
        "grid_voxel_center_is_half_cell":
            "(vox.x + 0.5f) * resolution_ + origin_x_" in grid_text,
        "grid_xy_out_of_bounds_is_mirrored":
            "symmetricIndex(vox.x, grid_size_x_)" in grid_text,
        "offline_oob_policy_is_high_cost_not_distance_authority":
            "out_of_bounds_cost" in static_loss,
        "legacy_generation_metadata_absent":
            not (DATASET / "generation_metadata.yaml").is_file(),
        "unfiltered_source_cloud_not_persisted":
            not any(DATASET.glob("unfiltered-pointcloud-*.ply")),
    }
    source_hashes = {
        str(path): sha256(path) for path in (
            generator, runtime, grid, grid_header, simulator_config,
            ROOT / "loss/safety_loss.py",
            ROOT / "configs/static_map_catalog.yaml",
        )
    }
    provenance = {
        "status": "PASS_WITH_AUTHORITY_GAP",
        "facts": facts,
        "source_hashes": source_hashes,
        "geometry_graph": [
            {
                "node": "procedural unfiltered cloud",
                "persistence": "not retained in legacy dataset",
                "transform": "mocka::Maps map_seed+map_id, current source only",
                "provenance_complete": False,
            },
            {
                "edge": "unfiltered cloud -> CUDA GridMap",
                "resolution_m": 0.1,
                "origin": "component-wise unfiltered cloud minimum",
                "occupancy": "point count > occupy_threshold(0)",
                "xy_oob": "symmetric mirror",
                "z_oob": "above free, at/below bottom occupied",
                "use": "depth/lidar raycast only",
            },
            {
                "edge": "unfiltered cloud -> saved raw PLY",
                "preprocessing": "PCL VoxelGrid leaf 0.1 m",
                "output_hashes": current_map_hashes,
                "information_loss": True,
            },
            {
                "edge": "saved PLY -> offline occupancy/ESDF",
                "resolution_m": 0.2,
                "origin": "filtered PLY min - map_expand_min",
                "occupancy": "floor point-to-cell; any point occupied",
                "distance": "SciPy Euclidean EDT at voxel samples",
                "interpolation": "torch grid_sample trilinear",
                "oob": "high cost, raw distance buffer still zero-padded",
            },
            {
                "node": "Simulator static UAV collision",
                "implementation": None,
                "contract": "absent",
                "authority_ready": False,
            },
            {
                "node": "Simulator dynamic UAV collision",
                "shape": "sphere radius 0.3 m versus actor shape",
                "not_applicable_to_static": True,
            },
        ],
        "same_hash_provenance": {
            "saved_ply_catalog_locked": True,
            "unfiltered_grid_hash_available": False,
            "offline_esdf_derived_artifact_persisted": False,
            "simulator_static_collision_geometry_hash_available": False,
        },
        "static_geometry_authority_ready": False,
    }
    atomic_json(
        REPORTS / "phase8jqv2_1_static_geometry_provenance.json",
        provenance,
    )

    contract = """# Phase 8J-Q2.1 Simulator static collision contract

Status: **ABSENT — no authoritative static UAV collision API exists.**

The CUDA Simulator constructs `GridMap` from a point cloud and uses
`mapQuery()` only inside camera/lidar ray-casting kernels. `odomCallback()`
updates pose and triggers sensors; it does not query static collision.

The only 0.3 m UAV collision path is
`actorCollidesWithUav(actor, pos, uav_collision_radius)`, which covers dynamic
actors and publishes `/dynamic_objects/uav_collision`. It does not query the
static `GridMap`.

For the legacy dataset, depth was rendered from a 0.1 m GridMap built from the
unfiltered procedural cloud. The persisted PLY was separately downsampled with
a 0.1 m PCL VoxelGrid. The unfiltered cloud and occupancy array were not
persisted, and the legacy dataset has no generation metadata tying it to the
current generator/config hashes.

Offline `SafetyLoss` loads the filtered PLY, creates a different 0.2 m
occupancy with a different origin convention, runs an EDT and trilinearly
interpolates it. PLY nearest-point distance is used only for pose rejection
and diagnostics. Neither is defined by Simulator as static collision truth.

Therefore no sphere-to-static contact semantics, authoritative out-of-bounds
policy, or Simulator/backend equality contract can currently be validated.
"""
    (REPORTS / "phase8jqv2_1_simulator_collision_contract.md").write_text(
        contract, encoding="utf-8"
    )

    decision = {
        "status": "FAIL",
        "selected_scheme": "D",
        "static_geometry_authority_ready": False,
        "reason": (
            "Simulator has no static UAV collision path; the raycast GridMap "
            "source occupancy was not persisted, while saved PLY and offline "
            "ESDF are lossy, differently discretized derivatives."
        ),
        "ply_nearest_promoted_to_authority": False,
        "dense_esdf_promoted_to_authority": False,
        "occupancy_promoted_to_authority": False,
        "derived_esdf_rebuilt": False,
        "next_allowed_phase": None,
    }
    atomic_json(
        REPORTS / "phase8jqv2_1_geometry_backend_decision.json", decision
    )

    reason = decision["reason"]
    blocked_reports = {
        "phase8jqv2_1_geometry_registration.json":
            "coordinate/map registration against authoritative collision",
        "phase8jqv2_1_esdf_error_envelope.json":
            "one-sided ESDF error against authoritative distance",
        "phase8jqv2_1_certificate_math.md":
            "formal V2.1 certificate selection",
        "phase8jqv2_1_synthetic_validation.json":
            "authority-backed synthetic checker validation",
        "phase8jqv2_1_real_map_validation.json":
            "authority-backed real-map checker validation",
        "phase8jqv2_1_unknown_analysis.json":
            "V2.1 checker UNKNOWN budget",
        "phase8jqv2_1_determinism_validation.json":
            "V2.1 evaluator determinism",
        "phase8jqv2_1_checkpoint_matrix.json":
            "V2.1 checkpoint rebaseline",
        "phase8jqv2_1_static_validation.json":
            "V2.1 full static validation",
        "phase8jqv2_1_dynamic_validation.json":
            "V2.1 full dynamic validation",
        "phase8jqv2_1_label_audit.json":
            "V2.1 label audit",
        "phase8jqv2_1_v2_delta.json":
            "V2 versus V2.1 delta",
    }
    for name, stage in blocked_reports.items():
        path = REPORTS / name
        if path.suffix == ".md":
            path.write_text(
                f"# {stage}\n\nStatus: **NOT EXECUTED**\n\n{reason}\n",
                encoding="utf-8",
            )
        else:
            atomic_json(path, unavailable(reason, stage))

    disagreements = conformance["independent_pointcloud_fixtures"]
    atomic_json(DIAGNOSTICS / "geometry_disagreements.json", {
        "status": "EVIDENCE_ONLY_NO_AUTHORITY",
        "count": len(disagreements),
        "dense_esdf_safe_pointcloud_collision_count": sum(
            bool(row["pointcloud_collision"]) for row in disagreements
        ),
        "records": disagreements,
        "interpretation": (
            "Neither side is promoted to truth because Simulator static "
            "collision authority is absent."
        ),
    })
    for name in (
        "esdf_false_safe.json", "esdf_false_unsafe.json",
        "certificate_unknown.json", "out_of_bounds.json",
        "static_window_flips.json",
    ):
        atomic_json(
            DIAGNOSTICS / name,
            unavailable(reason, name.removesuffix(".json")),
        )

    final = {
        "status": "FAIL",
        "primary_cause": "static_geometry_authority",
        "static_geometry_authority_ready": False,
        "static_numerical_rebaseline_ready": False,
        "immutable_fixture_preserved": True,
        "v1_preserved": True,
        "v2_preserved": True,
        "network_weights_modified": False,
        "training_executed": False,
        "score_training_executed": False,
        "candidate_generator_frozen": False,
        "production_test_used": False,
        "blind_used": False,
        "derived_geometry_rebuilt": False,
        "next_allowed_phase": None,
    }
    atomic_json(REPORTS / "phase8jqv2_1_final_result.json", final)
    recommendation = """# Phase 8J-Q2.1 final recommendation

**FAIL — geometry authority route C/D.**

The immutable fixture and S0 numbers are locked, but the current Simulator
does not implement static UAV collision. Its GridMap is a sensor raycast
backend, not a collision oracle.

The legacy source occupancy cannot be reconstructed exactly: depth rendering
used an unfiltered procedural cloud at 0.1 m, while only a separately
VoxelGrid-filtered PLY was persisted. Offline ESDF then re-voxelizes that PLY
at 0.2 m with a different origin and interpolation. The legacy dataset also
lacks generation metadata and the original unfiltered cloud/occupancy hash.

Consequently the 16/50 PLY-versus-ESDF disagreements cannot be labelled
false-safe or false-collision against Simulator truth. Selecting either side,
or manufacturing a safety margin, would violate the phase contract.

A future phase requires explicit user authorization and a product-level
static collision contract: choose and implement a versioned authoritative
geometry (for example persisted occupancy with sphere-to-voxel-AABB contact),
export its origin/resolution/hash, and make Simulator expose the same query.
Only then can registration, one-sided error, continuous certification and
V2.1 rebaseline resume.
"""
    (REPORTS / "phase8jqv2_1_final_recommendation.md").write_text(
        recommendation, encoding="utf-8"
    )
    (REPORTS / "phase8jqv2_1_final_readiness.md").write_text(
        """# Phase 8J-Q2.1 final readiness

**FAIL — no authoritative Simulator static collision geometry.**

V2.1 was not created. Full checkpoint/static/dynamic rebaseline, label audit,
gradient repair, candidate work and all training remain blocked.

`next_allowed_phase: null`
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "entry": entry["status"],
        "fixture_lock": fixture_lock["status"],
        "authority_ready": False,
        "final": final,
    }, indent=2))


if __name__ == "__main__":
    main()
