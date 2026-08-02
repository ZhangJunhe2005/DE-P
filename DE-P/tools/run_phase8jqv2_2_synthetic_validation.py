#!/usr/bin/env python3
"""Analytical synthetic validation for Static Geometry Authority V1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry_authority.static_v1 import (  # noqa: E402
    CONTACT_TOLERANCE_M, StaticAuthorityMap, build_authority_artifact,
)
from tools.run_phase8jqv2_2_authority_validation import (  # noqa: E402
    compare, read_cpp, write_queries,
)


SIMULATOR = Path("/home/zjh/YOPO/Simulator")
OUTPUT = ROOT / "geometry_authority/static_v1/synthetic"
EXECUTABLE = (
    SIMULATOR / "devel/lib/sensor_simulator/"
    "static_authority_contract_test"
)


def clouds():
    target = [(1., 1., 1.)]
    return {
        "single_occupied_voxel": target,
        "plane_of_voxels": target + [
            (1., y / 10, z / 10)
            for y in range(5, 16) for z in range(5, 16)
        ],
        "corner": target + [(1.1, 1., 1.), (1., 1.1, 1.)],
        "edge": target + [(1., 1., z / 10)
                          for z in range(5, 16)],
        "narrow_corridor": target + [
            (x / 10, y, z / 10) for x in range(5, 16)
            for y in (.8, 1.3) for z in range(5, 16, 2)
        ],
        "staircase": target + [
            (x / 10, 1., z / 10) for x in range(5, 16)
            for z in range(5, x + 1)
        ],
        "enclosed_box": target + [
            (x / 10, y / 10, z / 10)
            for x in range(5, 16, 2) for y in range(5, 16, 2)
            for z in range(5, 16, 2)
            if x in (5, 15) or y in (5, 15) or z in (5, 15)
        ],
        "isolated_pillar": target + [
            (1., 1., z / 10) for z in range(3, 18)
        ],
        "empty_map": [],
        "map_boundary": target + [(0., 0., 0.), (1.9, 1.9, 1.9)],
    }


def analytical_queries():
    corner = .3 / np.sqrt(3)
    edge = .3 / np.sqrt(2)
    return np.asarray([
        [.69, 1.05, 1.05, .3],       # 1 cm safe
        [.7, 1.05, 1.05, .3],        # exact tangent
        [.71, 1.05, 1.05, .3],       # 1 cm penetration
        [1.05, 1.05, 1.05, .3],      # center inside
        [1.1+corner, 1.1+corner, 1.1+corner, .3],
        [1.1+edge, 1.1+edge, 1.05, .3],
        [1.0, 1.05, 1.05, 0.0],      # zero-radius contact
        [.1, .1, .1, .3],            # OOB swept volume
        [1.4, 1.05, 1.05, .3],       # radius 0.3 tangent
        [1.05, 1.05, 1.05, 0.0],     # radius zero inside
    ], dtype="<f8")


def main():
    generator_hash = subprocess.check_output(
        ["sha256sum", str(Path(__file__))], text=True
    ).split()[0]
    config_hash = "0" * 64
    parent = subprocess.check_output(
        ["git", "-C", "/home/zjh/YOPO", "rev-parse", "HEAD"], text=True
    ).strip()
    records = []
    failures = []
    for index, (name, values) in enumerate(clouds().items()):
        map_uuid = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"static-authority-synthetic:{name}"
        ))
        root = OUTPUT / map_uuid
        build_authority_artifact(
            root, np.asarray(values, dtype=np.float32).reshape(-1, 3),
            map_uuid=map_uuid, generator_seed=index,
            map_id=f"synthetic-{index}",
            generator_source_hash=generator_hash,
            generator_config_hash=config_hash,
            parent_git_commit=parent,
            bounds_min=[0, 0, 0], bounds_max=[2, 2, 2],
        )
        authority = StaticAuthorityMap(root)
        rows = analytical_queries()
        offline = authority.query(rows[:, :3], rows[:, 3])
        query_file = root / "queries.bin"
        result_file = root / "results.bin"
        write_queries(query_file, rows)
        subprocess.run(
            [str(EXECUTABLE), str(root), str(query_file),
             str(result_file)], check=True, stdout=subprocess.DEVNULL,
        )
        cpu, gpu = read_cpp(result_file)
        query_file.unlink()
        result_file.unlink()
        cpu_delta = compare(offline, cpu)
        gpu_delta = compare(offline, gpu)
        if any(cpu_delta[key] or gpu_delta[key] for key in (
            "collision_mismatch", "oob_mismatch",
            "contacted_voxel_mismatch", "non_finite",
        )):
            failures.append(name)
        records.append({
            "geometry": name, "map_uuid": map_uuid,
            "query_count": len(rows), "cpu_offline": cpu_delta,
            "gpu_offline": gpu_delta,
            "occupancy_hash": authority.metadata["occupancy_hash"],
        })
    single = StaticAuthorityMap(OUTPUT / records[0]["map_uuid"])
    observed = single.query(
        analytical_queries()[:, :3], analytical_queries()[:, 3]
    ).collision.tolist()
    expected = [
        False, True, True, True, True,
        True, True, True, True, True,
    ]
    if observed != expected:
        failures.append("analytical_collision_expectations")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "authority": "static_geometry_authority_v1",
        "geometry_fixture_count": 10,
        "sphere_case_count": 10,
        "records": records,
        "analytical_expected_collision": expected,
        "analytical_observed_collision": observed,
        "minimum_gap_tolerance_m": CONTACT_TOLERANCE_M,
        "failures": failures,
    }
    path = ROOT / "reports/phase8jqv2_2_synthetic_validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    diagnostics = (
        ROOT / "diagnostics/phase8jqv2_2/contact_fixture_failures.json"
    )
    diagnostics.write_text(json.dumps({
        "status": report["status"], "count": len(failures),
        "records": failures,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "fixtures": 10,
        "queries": 100, "failures": failures,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
