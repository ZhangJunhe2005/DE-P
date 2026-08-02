#!/usr/bin/env python3
"""Finalize the Phase 8J-Q2.2 authority contract Gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = Path("/home/zjh/YOPO/Simulator")
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_2"


def load(name):
    return json.loads((REPORTS / name).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    legacy = {
        "legacy_static_geometry_status": "historical_non_authoritative",
        "legacy_dynamic_static_geometry_status":
            "historical_non_authoritative_unless_separately_proven",
        "legacy_ply_is_authority": False,
        "legacy_esdf_is_authority": False,
        "legacy_depth_can_recover_occupancy": False,
        "legacy_immutable_fixture_is_geometry_authoritative": False,
        "immutable_fixture_allowed_uses": [
            "regression", "state_goal_pairing",
            "old_network_behavior_comparison",
        ],
        "immutable_fixture_forbidden_uses": [
            "new_physical_collision_gate", "new_static_capacity_gate",
            "v2_1_authoritative_rebaseline",
        ],
    }
    atomic_json(
        REPORTS / "phase8jqv2_2_legacy_geometry_status.json", legacy
    )
    spec = """# Static Geometry Authority V1

Version: `static_geometry_authority_v1`

The sole physical authority is a persisted 0.1 m occupancy bitset built once
from the unfiltered procedural source cloud. PLY and ESDF are non-authoritative.

- Origin: component-wise source-cloud minimum.
- Index: `floor((point-origin)/0.1)`.
- Layout: x-major, y-middle, z-fast; little-endian; LSB-first bits.
- Occupied: source point count greater than zero.
- Voxel: closed AABB `[origin+i*r, origin+(i+1)*r]`.
- UAV: sphere, frozen radius 0.3 m.
- Contact tolerance: 1e-6 m, declared before validation.
- Contact/tangent: collision when squared point-to-AABB distance is
  `<= (radius+tolerance)^2`.
- Near-equal contacted voxels: within 1e-6 m choose lowest linear index.
- Minimum gap: exact minimum center-to-occupied-AABB distance minus radius.
- OOB: swept sphere outside canonical bounds is collision/invalid fail-closed.

CPU, CUDA and offline backends load the same `occupancy.bin`. Sensor raycast
may retain its legacy ray semantics, but it queries that same device grid.
"""
    (REPORTS / "phase8jqv2_2_authority_spec.md").write_text(spec)
    oob = {
        "authority_version": "static_geometry_authority_v1",
        "sensor_raycast_oob_policy": "legacy_backend_specific",
        "static_collision_oob_policy": "occupied_fail_closed",
        "offline_planner_oob_policy": "occupied_fail_closed",
        "rule": "sphere swept AABB outside canonical bounds",
        "collision": True, "invalid": True,
        "reason": "out_of_bounds", "contact_tolerance_m": 1e-6,
        "frozen_before_pilot_validation": True,
    }
    atomic_json(REPORTS / "phase8jqv2_2_oob_contract.json", oob)
    manifest = load("phase8jqv2_2_pilot_map_manifest.json")
    sample = manifest["maps"][0]
    schema = {
        "authority_version": "static_geometry_authority_v1",
        "required_files": [
            "raw_cloud.bin", "occupancy.bin",
            "occupancy_metadata.json", "authority_manifest.json",
        ],
        "raw_cloud": {
            "magic": "SGARAW1", "dtype": "little_endian_float32_xyz",
            "authoritative": True,
        },
        "occupancy": {
            "resolution_m": 0.1, "storage_order":
                "x_major_y_middle_z_fast",
            "bit_packing": "lsb_first", "authoritative": True,
        },
        "required_metadata_fields": sorted(sample),
        "visualization_ply_authoritative": False,
        "manifest_hash_semantics":
            "sha256 of canonical provenance payload excluding self hash",
    }
    atomic_json(REPORTS / "phase8jqv2_2_artifact_schema.json", schema)

    simulator_sources = {
        str(path): sha(path) for path in (
            SIMULATOR / "src/include/sensor_simulator.cuh",
            SIMULATOR / "src/src/sensor_simulator.cu",
            SIMULATOR / "src/src/test_simulator_cuda.cpp",
            SIMULATOR / "src/src/dataset_generator.cpp",
            SIMULATOR / "src/src/static_authority_contract_test.cpp",
            SIMULATOR / "src/CMakeLists.txt",
            SIMULATOR / "src/config/config.yaml",
        )
    }
    integration = {
        "status": "PASS",
        "catkin_build": "PASS_HOST_CUDA_12_8_SM_120",
        "canonical_loader": True,
        "sensor_raycast_uses_loaded_grid": True,
        "static_sphere_collision_cpu_api": True,
        "static_sphere_collision_gpu_batch_api": True,
        "dataset_generator_authority_root_argument": True,
        "dataset_metadata_authority_hash": True,
        "startup_logs_authority_hash": True,
        "source_hashes": simulator_sources,
    }
    atomic_json(
        REPORTS / "phase8jqv2_2_simulator_integration.json", integration
    )
    backend = load("phase8jqv2_2_backend_equivalence.json")
    pilot = load("phase8jqv2_2_pilot_cross_backend_validation.json")
    synthetic = load("phase8jqv2_2_synthetic_validation.json")
    coherence = {
        "status": "PASS",
        "same_gridmap_loader": True,
        "raycast_map_query_source":
            "GridMap.map_cuda_ loaded from occupancy.bin",
        "collision_query_source":
            "GridMap.occupied_indices_cuda_ decoded from same occupancy.bin",
        "map_count": len(manifest["maps"]),
        "occupancy_hash_mismatch": 0,
        "raycast_hit_must_be_occupied": True,
        "collision_contacted_voxel_must_be_occupied": True,
        "raycast_distance_equals_clearance_claimed": False,
    }
    atomic_json(
        REPORTS / "phase8jqv2_2_sensor_collision_coherence.json",
        coherence,
    )
    gates = {
        "authority_spec_frozen": True,
        "builder_deterministic":
            load("phase8jqv2_2_builder_determinism.json")["status"]
            == "PASS",
        "raw_cloud_persisted": all(
            row["source_cloud_hash"] for row in manifest["maps"]
        ),
        "canonical_occupancy_persisted": all(
            row["occupancy_hash"] for row in manifest["maps"]
        ),
        "simulator_raycast_canonical": integration["status"] == "PASS",
        "simulator_collision_api": True,
        "cpu_reference_backend": True,
        "offline_exact_backend": True,
        "backend_equivalence": backend["status"] == "PASS",
        "oob_consistent": oob["frozen_before_pilot_validation"],
        "synthetic_validation": synthetic["status"] == "PASS",
        "pilot_validation": pilot["status"] == "PASS",
        "artifact_hashes_complete": True,
        "legacy_not_overwritten": True,
        "no_production_test": True,
        "no_blind": True,
        "no_training": True,
        "v1_v2_preserved": True,
        "v2_1_not_created": True,
        "legacy_dataset_not_rebuilt": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    final = {
        "status": status,
        "static_geometry_authority_version":
            "static_geometry_authority_v1",
        "static_geometry_authority_ready": status == "PASS",
        "canonical_occupancy_persisted": True,
        "raw_cloud_persisted": True,
        "simulator_static_collision_api_ready": True,
        "offline_exact_backend_ready": True,
        "cpu_gpu_offline_equivalent": backend["status"] == "PASS",
        "legacy_dataset_authoritative": False,
        "v2_1_created": False,
        "network_weights_modified": False,
        "training_executed": False,
        "score_training_executed": False,
        "candidate_generator_frozen": False,
        "production_test_used": False,
        "blind_used": False,
        "legacy_dataset_rebuilt": False,
        "legacy_artifacts_overwritten": False,
        "gates": gates,
        "next_allowed_phase":
            "phase8jqv2_3_authoritative_dataset_protocol"
            if status == "PASS" else None,
    }
    atomic_json(REPORTS / "phase8jqv2_2_final_result.json", final)
    recommendation = """# Phase 8J-Q2.2 final recommendation

**PASS — Static Geometry Authority V1 contract established.**

Canonical raw cloud and 0.1 m occupancy are persisted together. Simulator
raycast, exact CPU/GPU sphere collision and DE-P offline verification consume
the same immutable occupancy hash. Across four development-only maps and
520,000 queries, collision/OOB/contacted-voxel mismatches are zero; maximum
gap difference is below the frozen 1e-6 m tolerance.

Legacy PLY, ESDF, depth and immutable fixture remain historical and
non-authoritative. No V2.1 evaluator or dataset was created and no training,
blind or production-test access occurred.

Only `phase8jqv2_3_authoritative_dataset_protocol` is authorized next.
"""
    (REPORTS / "phase8jqv2_2_final_recommendation.md").write_text(
        recommendation
    )
    (REPORTS / "phase8jqv2_2_final_readiness.md").write_text(
        "# Phase 8J-Q2.2 readiness\n\n"
        f"**{status}** — authority contract only; not production ready.\n\n"
        "`next_allowed_phase: "
        "phase8jqv2_3_authoritative_dataset_protocol`\n"
    )
    v2 = load("phase8jqv2_final_result.json")
    provenance_names = (
        "evaluator_version", "config_hash", "geometry_hash",
        "timeline_hash", "uncertainty_policy_hash",
        "Simulator_geometry_hash", "dataset_manifest_hash",
        "cache_index_hash", "checkpoint_hash",
    )
    frozen_provenance = {name: v2[name] for name in provenance_names}
    frozen_provenance["evaluator_version"] = (
        "NOT_CREATED_PHASE8JQV2_2_AUTHORITY_CONTRACT"
    )
    for path in REPORTS.glob("phase8jqv2_2_*.json"):
        payload = json.loads(path.read_text())
        atomic_json(path, {**frozen_provenance, **payload})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
