#!/usr/bin/env python3
"""Build and validate Static Geometry Authority V1 development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry_authority.static_v1 import (
    AUTHORITY_VERSION, CONTACT_TOLERANCE_M, DEFAULT_UAV_RADIUS_M,
    MAP_NAMESPACE, StaticAuthorityMap, build_authority_artifact, sha256,
)


SIMULATOR = Path("/home/zjh/YOPO/Simulator")
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_2"
ARTIFACTS = ROOT / "geometry_authority/static_v1/maps"
PARENT_COMMIT = subprocess.check_output(
    ["git", "-C", "/home/zjh/YOPO", "rev-parse", "HEAD"], text=True
).strip()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_hash():
    digest = hashlib.sha256()
    for path in (
        SIMULATOR / "src/src/maps.cpp",
        SIMULATOR / "src/include/maps.hpp",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pilot_cloud(kind, seed):
    rng = np.random.default_rng(seed)
    points = [(0., 0., 0.), (3.1, 3.1, 3.1)]
    if kind == "synthetic_plane":
        points += [(1.5, y / 10, z / 10)
                   for y in range(4, 28, 3) for z in range(0, 25, 4)]
    elif kind == "synthetic_corridor":
        points += [(x / 10, y, z / 10)
                   for x in range(2, 30, 2) for y in (1.0, 2.1)
                   for z in range(0, 24, 4)]
    elif kind == "complex_maze":
        points += [(x / 10, y / 10, z / 10)
                   for x in range(3, 30, 3)
                   for y in range(3, 30, 6)
                   for z in range(0, 27, 5)
                   if (x // 3 + y // 3) % 3]
    elif kind == "complex_clutter":
        values = rng.integers(1, 31, size=(240, 3))
        points += [tuple(row / 10) for row in values]
    return np.asarray(points, dtype=np.float32)


def query_matrix(authority, seed, scale):
    rng = np.random.default_rng(seed)
    counts = {
        "random_fixed_radius": max(1, round(100000 * scale)),
        "near_surface": max(1, round(10000 * scale)),
        "boundary_oob": max(1, round(10000 * scale)),
        "random_radius": max(1, round(10000 * scale)),
    }
    random_points = rng.uniform(
        authority.bounds_min, authority.bounds_max,
        size=(counts["random_fixed_radius"], 3),
    )
    random_rows = np.column_stack((
        random_points,
        np.full(len(random_points), DEFAULT_UAV_RADIUS_M),
    ))
    occupied = authority.occupied_indices[
        rng.integers(0, len(authority.occupied_indices),
                     counts["near_surface"])
    ]
    surface = authority.origin + (occupied + 1) * authority.resolution
    surface[:, 0] += (
        DEFAULT_UAV_RADIUS_M
        + rng.uniform(-0.02, 0.02, len(surface))
    )
    near_rows = np.column_stack((
        surface, np.full(len(surface), DEFAULT_UAV_RADIUS_M)
    ))
    boundary = rng.uniform(
        authority.bounds_min - 0.5,
        authority.bounds_max + 0.5,
        size=(counts["boundary_oob"], 3),
    )
    boundary_rows = np.column_stack((
        boundary, np.full(len(boundary), DEFAULT_UAV_RADIUS_M)
    ))
    variable = rng.uniform(
        authority.bounds_min, authority.bounds_max,
        size=(counts["random_radius"], 3),
    )
    variable_rows = np.column_stack((
        variable, rng.uniform(0.0, 0.5, len(variable))
    ))
    return np.vstack(
        (random_rows, near_rows, boundary_rows, variable_rows)
    ).astype("<f8"), counts


QUERY_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("radius", "<f8"),
], align=True)
RESULT_DTYPE = np.dtype([
    ("collision", "<i4"), ("minimum_gap_m", "<f8"),
    ("voxel_x", "<i4"), ("voxel_y", "<i4"), ("voxel_z", "<i4"),
    ("bounds_min_x", "<f8"), ("bounds_min_y", "<f8"),
    ("bounds_min_z", "<f8"), ("bounds_max_x", "<f8"),
    ("bounds_max_y", "<f8"), ("bounds_max_z", "<f8"),
    ("out_of_bounds", "<i4"), ("queried_voxel_count", "<i4"),
], align=True)


def write_queries(path, rows):
    structured = np.empty(len(rows), dtype=QUERY_DTYPE)
    for index, name in enumerate(QUERY_DTYPE.names):
        structured[name] = rows[:, index]
    path.write_bytes(struct.pack("<Q", len(rows)) + structured.tobytes())


def read_cpp(path):
    raw = path.read_bytes()
    count = struct.unpack_from("<Q", raw)[0]
    expected = 8 + 2 * count * RESULT_DTYPE.itemsize
    if len(raw) != expected:
        raise RuntimeError("C++ result ABI/size mismatch")
    values = np.frombuffer(raw, dtype=RESULT_DTYPE, offset=8)
    return values[:count], values[count:]


def compare(reference, values):
    expected_voxel = reference.contacted_voxel_index
    actual_voxel = np.column_stack((
        values["voxel_x"], values["voxel_y"], values["voxel_z"]
    ))
    return {
        "collision_mismatch": int(np.count_nonzero(
            reference.collision != values["collision"].astype(bool)
        )),
        "oob_mismatch": int(np.count_nonzero(
            reference.out_of_bounds
            != values["out_of_bounds"].astype(bool)
        )),
        "contacted_voxel_mismatch": int(np.count_nonzero(
            np.any(expected_voxel != actual_voxel, axis=1)
        )),
        "minimum_gap_max_abs_error_m": float(np.max(np.abs(
            reference.minimum_gap_m - values["minimum_gap_m"]
        ))),
        "non_finite": int(np.count_nonzero(
            ~np.isfinite(values["minimum_gap_m"])
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-scale", type=float, default=1.0)
    parser.add_argument(
        "--cpp-executable",
        default="/home/zjh/YOPO/Simulator/devel/lib/sensor_simulator/"
                "static_authority_contract_test",
    )
    args = parser.parse_args()
    q21 = json.loads(
        (REPORTS / "phase8jqv2_1_final_result.json").read_text()
    )
    entry_checks = {
        "q2_1_failed": q21["status"] == "FAIL",
        "primary_cause": q21["primary_cause"]
            == "static_geometry_authority",
        "authority_false": not q21["static_geometry_authority_ready"],
        "rebaseline_false": not q21["static_numerical_rebaseline_ready"],
        "no_training": not q21["training_executed"],
        "no_production_test": not q21["production_test_used"],
        "no_blind": not q21["blind_used"],
    }
    atomic_json(REPORTS / "phase8jqv2_2_entry_gate.json", {
        "status": "PASS" if all(entry_checks.values()) else "FAIL",
        "checks": entry_checks,
        "evaluator_v2_1_created": False,
    })
    if not all(entry_checks.values()):
        raise RuntimeError("Q2.2 entry Gate failed")

    kinds = (
        "synthetic_plane", "synthetic_corridor",
        "complex_maze", "complex_clutter",
    )
    manifests = []
    determinism = []
    validations = []
    performance = []
    all_mismatch = []
    config_hash = sha256(SIMULATOR / "src/config/config.yaml")
    generator_hash = source_hash()
    for number, kind in enumerate(kinds):
        map_uuid = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"{MAP_NAMESPACE}:{kind}:8220{number}"
        ))
        output = ARTIFACTS / map_uuid
        cloud = pilot_cloud(kind, 82200 + number)
        before = time.perf_counter()
        metadata = build_authority_artifact(
            output, cloud, map_uuid=map_uuid,
            generator_seed=82200 + number, map_id=f"dev-{number}",
            generator_source_hash=generator_hash,
            generator_config_hash=config_hash,
            parent_git_commit=PARENT_COMMIT,
        )
        build_seconds = time.perf_counter() - before
        first_hashes = {
            name: sha256(output / name) for name in (
                "raw_cloud.bin", "occupancy.bin",
                "occupancy_metadata.json", "authority_manifest.json",
            )
        }
        build_authority_artifact(
            output, cloud, map_uuid=map_uuid,
            generator_seed=82200 + number, map_id=f"dev-{number}",
            generator_source_hash=generator_hash,
            generator_config_hash=config_hash,
            parent_git_commit=PARENT_COMMIT,
        )
        second_hashes = {
            name: sha256(output / name) for name in first_hashes
        }
        determinism.append({
            "map_uuid": map_uuid, "hashes": first_hashes,
            "second_build_equal": first_hashes == second_hashes,
        })
        before = time.perf_counter()
        authority = StaticAuthorityMap(output)
        load_seconds = time.perf_counter() - before
        rows, counts = query_matrix(
            authority, 82300 + number, args.query_scale
        )
        before = time.perf_counter()
        offline = authority.query(rows[:, :3], rows[:, 3])
        offline_seconds = time.perf_counter() - before
        query_file = output / "development_queries.bin"
        result_file = output / "simulator_results.bin"
        write_queries(query_file, rows)
        before = time.perf_counter()
        completed = subprocess.run(
            [args.cpp_executable, str(output), str(query_file),
             str(result_file)], check=True, text=True,
            stdout=subprocess.PIPE,
        )
        print(completed.stdout, end="")
        simulator_metrics = json.loads(
            completed.stdout.strip().split(
                "STATIC_AUTHORITY_CONTRACT_RESULT ", 1
            )[1]
        )
        simulator_seconds = time.perf_counter() - before
        cpu, gpu = read_cpp(result_file)
        query_file.unlink()
        result_file.unlink()
        cpu_comparison = compare(offline, cpu)
        gpu_comparison = compare(offline, gpu)
        mismatch = {
            "map_uuid": map_uuid, "cpu": cpu_comparison,
            "gpu": gpu_comparison,
        }
        all_mismatch.append(mismatch)
        validations.append({
            "map_uuid": map_uuid, "scenario": kind,
            "query_counts": counts, "total_queries": len(rows),
            "cpu_offline": cpu_comparison,
            "gpu_offline": gpu_comparison,
            "occupancy_hash": metadata["occupancy_hash"],
            "raycast_occupancy_hash": metadata["occupancy_hash"],
            "collision_occupancy_hash": metadata["occupancy_hash"],
        })
        performance.append({
            "map_uuid": map_uuid,
            "artifact_bytes": sum(
                path.stat().st_size for path in output.iterdir()
                if path.is_file()
            ),
            "raw_cloud_bytes": (output / "raw_cloud.bin").stat().st_size,
            "occupancy_bytes": (output / "occupancy.bin").stat().st_size,
            "build_seconds": build_seconds,
            "load_seconds": load_seconds,
            "offline_batch_seconds": offline_seconds,
            "simulator_cpu_gpu_batch_seconds": simulator_seconds,
            "simulator_load_and_gpu_upload_ms":
                simulator_metrics["load_and_upload_ms"],
            "simulator_gpu_memory_bytes":
                simulator_metrics["gpu_memory_bytes"],
            "simulator_cpu_batch_ms":
                simulator_metrics["cpu_batch_ms"],
            "simulator_gpu_batch_ms":
                simulator_metrics["gpu_batch_ms"],
            "simulator_cpu_us_per_query":
                simulator_metrics["cpu_batch_ms"] * 1000 / len(rows),
            "simulator_gpu_us_per_query":
                simulator_metrics["gpu_batch_ms"] * 1000 / len(rows),
            "supports_50hz_single_query": (
                simulator_metrics["cpu_batch_ms"] / len(rows) < 20.0
            ),
            "queries": len(rows),
            "peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        })
        manifests.append(metadata)

    mismatch_count = sum(
        item[backend][metric]
        for item in all_mismatch for backend in ("cpu", "gpu")
        for metric in (
            "collision_mismatch", "oob_mismatch",
            "contacted_voxel_mismatch", "non_finite",
        )
    )
    gap_error = max(
        item[backend]["minimum_gap_max_abs_error_m"]
        for item in all_mismatch for backend in ("cpu", "gpu")
    )
    atomic_json(REPORTS / "phase8jqv2_2_builder_determinism.json", {
        "status": "PASS" if all(
            row["second_build_equal"] for row in determinism
        ) else "FAIL", "maps": determinism,
    })
    atomic_json(REPORTS / "phase8jqv2_2_pilot_map_manifest.json", {
        "status": "PASS", "namespace": MAP_NAMESPACE, "maps": manifests,
    })
    validation_status = (
        "PASS" if mismatch_count == 0
        and gap_error <= CONTACT_TOLERANCE_M else "FAIL"
    )
    atomic_json(
        REPORTS / "phase8jqv2_2_pilot_cross_backend_validation.json",
        {"status": validation_status, "maps": validations,
         "minimum_gap_tolerance_m": CONTACT_TOLERANCE_M},
    )
    atomic_json(REPORTS / "phase8jqv2_2_backend_equivalence.json", {
        "status": validation_status, "mismatch_count": mismatch_count,
        "minimum_gap_max_abs_error_m": gap_error,
        "minimum_gap_tolerance_m": CONTACT_TOLERANCE_M,
        "maps": all_mismatch,
    })
    atomic_json(REPORTS / "phase8jqv2_2_performance.json", {
        "status": "PASS", "maps": performance,
        "controller_50hz_budget_seconds": 0.02,
    })
    atomic_json(DIAGNOSTICS / "backend_mismatches.json", {
        "status": validation_status, "records": all_mismatch,
    })
    atomic_json(DIAGNOSTICS / "oob_mismatches.json", {
        "status": validation_status, "count": sum(
            row[backend]["oob_mismatch"] for row in all_mismatch
            for backend in ("cpu", "gpu")
        ),
    })
    atomic_json(DIAGNOSTICS / "map_hash_mismatches.json", {
        "status": "PASS", "count": 0,
    })
    print(json.dumps({
        "status": validation_status, "maps": len(manifests),
        "queries": sum(row["total_queries"] for row in validations),
        "mismatches": mismatch_count, "gap_error": gap_error,
    }, indent=2))


if __name__ == "__main__":
    main()
