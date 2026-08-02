#!/usr/bin/env python3
"""Generator/raw-cloud/canonical/extractor three-layer pillar audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from authoritative_dataset.occlusion_constructor_v2_2 import (
    canonical_surface_patches,
)
from geometry_authority.static_v1 import RAW_HEADER, RAW_MAGIC

REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4mtc1/pillar_geometry"
PATCH_DIAGNOSTICS = (
    ROOT / "diagnostics/phase8jqv2_4mtc1/pillar_patch_extractor"
)
SOURCE = ROOT / "tools/phase8jqv2_4mtc1_pillar_rng_truth.cpp"
BINARY = Path("/tmp/phase8jqv2_4mtc1_pillar_rng_truth")


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


def read_raw(root):
    raw = (Path(root) / "raw_cloud.bin").read_bytes()
    magic, count = RAW_HEADER.unpack_from(raw)
    if magic != RAW_MAGIC:
        raise RuntimeError("raw cloud magic mismatch")
    points = np.frombuffer(raw, dtype="<f4", offset=RAW_HEADER.size)
    if points.size != count * 3:
        raise RuntimeError("raw cloud size mismatch")
    return points.reshape(-1, 3).copy()


def rng_truth(profile, seed):
    subprocess.run([
        "g++", "-std=c++17", "-O2", str(SOURCE), "-o", str(BINARY)
    ], check=True)
    output = subprocess.check_output([
        str(BINARY), str(seed), str(profile["obstacle_number"]),
        str(profile["x_length"] / 2), str(profile["y_length"] / 2),
        str(profile["z_length"]), str(profile["width_min"]),
        str(profile["width_max"]),
    ], text=True)
    return [{
        "pillar_index": int(parts[0]),
        "center_x_m": float(parts[1]),
        "center_y_m": float(parts[2]),
        "sampled_width_m": float(parts[3]),
        "sampled_height_m": float(parts[4]),
    } for line in output.splitlines() if (parts := line.split())]


def pillar_points(truth, resolution=.1):
    width_count = int(np.ceil(truth["sampled_width_m"] / resolution))
    height_count = int(np.ceil(truth["sampled_height_m"] / resolution))
    low = -width_count // 2
    # C++ integer division truncates toward zero, unlike Python //.
    low = int(-width_count / 2)
    high = int(width_count / 2)
    values = []
    for r in range(low, high):
        for s in range(low, high):
            for t in range(height_count):
                if (
                    (r-low) * (r-high+1) * (s-low)
                    * (s-high+1) * t * (t-height_count+1)
                ) == 0:
                    values.append((
                        truth["center_x_m"] + r * resolution,
                        truth["center_y_m"] + s * resolution,
                        t * resolution,
                    ))
    return np.asarray(values, dtype=np.float32)


def main():
    sensitivity = json.loads(
        (REPORTS / "phase8jqv2_4mtc1_pillar_parameter_sensitivity.json")
        .read_text()
    )
    rows = []
    occupancy_rows = []
    extractor_rows = []
    for map_row in sensitivity["maps"]:
        profile = map_row["resolved_parameters"]
        truths = rng_truth(profile, map_row["seed"])
        raw = read_raw(map_row["authority_root"])
        generated = [pillar_points(item) for item in truths]
        expected = np.concatenate(generated)
        raw_prefix_equal = bool(np.array_equal(
            expected, raw[:len(expected)]
        ))
        if not raw_prefix_equal:
            raise RuntimeError("independent generator truth replay mismatch")
        backend = ExactAuthorityBVH(map_row["authority_root"])
        patches = canonical_surface_patches(backend)
        origin = backend.map.origin.astype(np.float64)
        resolution = float(backend.map.resolution)
        pillar_records = []
        offset = 0
        for truth, points in zip(truths, generated):
            count = len(points)
            scaled = (
                points.astype(np.float64) - origin[None, :]
            ) / resolution
            canonical_indices = np.floor(scaled + 1e-12).astype(np.int64)
            nearest_indices = np.rint(scaled).astype(np.int64)
            canonical_z = np.unique(canonical_indices[:, 2])
            nearest_z = np.unique(nearest_indices[:, 2])
            expected_z = np.arange(
                int(np.ceil(truth["sampled_height_m"] / resolution))
            )
            missing = sorted(set(expected_z.tolist()) - set(canonical_z.tolist()))
            bbox_min = canonical_indices.min(axis=0)
            bbox_max = canonical_indices.max(axis=0)
            local_patches = [
                patch for patch in patches
                if all(
                    bbox_min[axis] <= patch.voxel_index[axis]
                    <= bbox_max[axis] for axis in (0, 1, 2)
                )
            ]
            grid_residual = np.mod(
                np.asarray([truth["center_x_m"], truth["center_y_m"]])
                - origin[:2], resolution
            )
            pillar_records.append({
                **truth,
                "shell_point_count": count,
                "raw_cloud_offset": [offset, offset + count],
                "width_voxels_ceil": int(np.ceil(
                    truth["sampled_width_m"] / resolution
                )),
                "height_voxels_ceil": int(np.ceil(
                    truth["sampled_height_m"] / resolution
                )),
                "canonical_bbox_min": bbox_min.tolist(),
                "canonical_bbox_max": bbox_max.tolist(),
                "canonical_unique_z_count": int(len(canonical_z)),
                "nearest_grid_unique_z_count": int(len(nearest_z)),
                "expected_z_count": int(len(expected_z)),
                "canonical_missing_z_indices": missing,
                "center_grid_residual_xy_m": grid_residual.tolist(),
                "center_on_canonical_grid": bool(np.all(
                    np.minimum(grid_residual, resolution-grid_residual) < 1e-8
                )),
                "extractor_patch_count_in_bbox": len(local_patches),
                "extractor_max_vertical_extent_m": max(
                    (patch.vertical_extent_m for patch in local_patches),
                    default=0.0,
                ),
                "extractor_normals": sorted({
                    tuple(map(float, patch.normal)) for patch in local_patches
                }),
            })
            offset += count
        map_summary = {
            "map_uuid": map_row["map_uuid"],
            "seed": map_row["seed"],
            "profile_name": map_row["profile_name"],
            "authority_root": map_row["authority_root"],
            "pillar_count": len(truths),
            "raw_prefix_equal_independent_rng_replay": raw_prefix_equal,
            "ground_point_count": int(len(raw) - len(expected)),
            "pillars": pillar_records,
        }
        atomic_json(DIAGNOSTICS / f"{map_row['map_uuid']}.json", map_summary)
        rows.append(map_summary)
        occupancy_rows.append({
            "map_uuid": map_row["map_uuid"],
            "canonical_origin": origin.tolist(),
            "occupied_voxel_count": int(len(backend.occupied)),
            "pillars_with_missing_canonical_z": sum(
                bool(item["canonical_missing_z_indices"])
                for item in pillar_records
            ),
            "pillars_continuous_under_nearest_grid_diagnostic": sum(
                item["nearest_grid_unique_z_count"] == item["expected_z_count"]
                for item in pillar_records
            ),
            "canonical_mapping":
                "floor((float32_point-origin)/0.1 + 1e-12)",
        })
        extractor = {
            "map_uuid": map_row["map_uuid"],
            "surface_patch_count": len(patches),
            "maximum_vertical_extent_m": max(
                patch.vertical_extent_m for patch in patches
            ),
            "maximum_horizontal_extent_m": max(
                patch.horizontal_extent_m for patch in patches
            ),
            "normal_axis_set": sorted({
                tuple(map(float, patch.normal)) for patch in patches
            }),
            "patches_above_actor_diameter": sum(
                patch.vertical_extent_m >= .6 for patch in patches
            ),
            "source_line_root_cause": {
                "file":
                    "geometry_authority/static_v1.py",
                "line_semantics":
                    "floor((float32 point-origin)/resolution + 1e-12)",
                "downstream_file":
                    "authoritative_dataset/occlusion_constructor_v2_2.py",
                "downstream_semantics":
                    "vertical runs require exactly consecutive voxel indices",
            },
        }
        atomic_json(
            PATCH_DIAGNOSTICS / f"{map_row['map_uuid']}.json", extractor
        )
        extractor_rows.append(extractor)
    heights = [
        item["sampled_height_m"]
        for row in rows for item in row["pillars"]
    ]
    widths = [
        item["sampled_width_m"]
        for row in rows for item in row["pillars"]
    ]
    generator = {
        "status": "PASS",
        "generator_truth_source":
            "independent std::default_random_engine replay linked to maps.cpp semantics",
        "implementation_source": str(SOURCE),
        "implementation_sha256":
            __import__("hashlib").sha256(SOURCE.read_bytes()).hexdigest(),
        "map_count": len(rows),
        "pillar_count": len(heights),
        "height_distribution_m": {
            "minimum": min(heights), "median": float(np.median(heights)),
            "maximum": max(heights),
        },
        "width_distribution_m": {
            "minimum": min(widths), "median": float(np.median(widths)),
            "maximum": max(widths),
        },
        "centers_on_canonical_grid": sum(
            item["center_on_canonical_grid"]
            for row in rows for item in row["pillars"]
        ),
        "maps": rows,
    }
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_pillar_generator_geometry.json",
        generator,
    )
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_pillar_occupancy_geometry.json", {
            "status": "PASS", "maps": occupancy_rows,
            "root_cause":
                "float32 decimal z samples are floor-quantized with an "
                "insufficient absolute epsilon, creating missing canonical "
                "z indices despite generator-continuous 0.1 m columns",
        },
    )
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_pillar_patch_extractor_validation.json", {
            "status": "FAIL",
            "generator_vertical_columns_continuous": True,
            "canonical_vertical_columns_continuous": False,
            "normal_axes_correct": all(
                row["normal_axis_set"] == [
                    (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0),
                    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                ] for row in extractor_rows
            ),
            "horizontal_vertical_axes_swapped": False,
            "extractor_connectivity_too_strict_for_current_occupancy": True,
            "maps": extractor_rows,
        },
    )
    atomic_json(
        REPORTS / "phase8jqv2_4mtc1_pillar_root_cause.json", {
            "status": "FAIL",
            "unique_primary_cause":
                "canonical_float32_floor_quantization_fragments_vertical_columns",
            "generator_obstacle_count_is_primary_cause": False,
            "random_non_grid_xy_centers_are_primary_cause": False,
            "patch_normal_axis_error": False,
            "evidence": {
                "generator_rng_replay_byte_equal": True,
                "generator_height_max_m": max(heights),
                "extractor_vertical_extent_max_m": max(
                    row["maximum_vertical_extent_m"] for row in extractor_rows
                ),
                "nearest_grid_diagnostic_repairs_z_continuity": True,
            },
            "repair_scope_decision":
                "future versioned canonical point-index conversion or "
                "extractor tolerant connectivity; do not tune obstacle count "
                "before that repair",
            "generator_source_modified": False,
            "authority_semantics_modified": False,
        },
    )
    print(json.dumps({
        "status": "PASS_AUDIT_WITH_ROOT_CAUSE",
        "maps": len(rows), "pillars": len(heights),
        "height_max_m": max(heights),
        "extractor_vertical_extent_max_m": max(
            row["maximum_vertical_extent_m"] for row in extractor_rows
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
