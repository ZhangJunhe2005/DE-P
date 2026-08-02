#!/usr/bin/env python3
"""DPAR1 entry and control-contract audit.

This tool is intentionally an early gate.  It does not construct an
architecture candidate, read sealed holdout cases, or run a tracker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from tools.evaluate_phase8jqv2_4tf1_candidates import (  # noqa: E402
    negative_fixture, synthetic_fixture,
)


def read(name):
    return json.loads((REPORTS / name).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DPAR1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = (
        value.rstrip() + "\n" if isinstance(value, str)
        else json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    temporary.write_text(text)
    os.replace(temporary, path)


def frozen_paths():
    return {
        "temporal_foreground.py":
            ROOT / "policy/dynamic/temporal_foreground.py",
        "range_image_foreground.py":
            ROOT / "policy/dynamic/range_image_foreground.py",
        "track_manager.py": ROOT / "policy/dynamic/track_manager.py",
        "occlusion_constructor_v2_1.py":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "occlusion_constructor_v2_2.py":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "occlusion_identity_schedule_v1.py":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "motion_contract_v2_1.yaml":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "mixed_scene_map_profiles_v1.yaml":
            ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "mixed_scene_map_profiles_v2.yaml":
            ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "sensor_traj_opt.yaml": ROOT / "config/traj_opt.yaml",
    }


def projected_sphere(focal, radius, center_distance):
    # The silhouette half-angle of a sphere is asin(r / d).  The pinhole
    # radius is f*tan(angle) = f*r/sqrt(d^2-r^2).
    radius_pixels = (
        focal * radius
        / math.sqrt(center_distance**2 - radius**2)
    )
    return {
        "diameter_pixels": 2.0 * radius_pixels,
        "continuous_disc_area_pixels2": math.pi * radius_pixels**2,
    }


def main():
    tf1_final = read("phase8jqv2_4tf1_final_result.json")
    tf1_entry = read("phase8jqv2_4tf1_entry_gate.json")
    tf1_freeze = read("phase8jqv2_4tf1_holdout_freeze.json")
    tf1_split = read("phase8jqv2_4tf1_evaluation_split.json")
    required_final = {
        "status": "FAIL",
        "primary_cause": "temporal_foreground_contract_architecture_limit",
        "candidate_selected": False,
        "selected_candidate": None,
        "holdout_accessed": False,
        "tracker_modified": False,
        "legacy_modified": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "training_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_architecture_review",
    }
    mismatches = {
        key: {"expected": expected, "actual": tf1_final.get(key)}
        for key, expected in required_final.items()
        if tf1_final.get(key) != expected
    }
    current_hashes = {
        name: sha256(path) for name, path in frozen_paths().items()
    }
    for name, current in current_hashes.items():
        expected = tf1_entry["frozen_hashes"][name]
        if current != expected:
            mismatches[f"frozen_hash:{name}"] = {
                "expected": expected, "actual": current,
            }
    tf1_candidate_path = (
        ROOT / "policy/dynamic/range_image_foreground_v2_1.py"
    )
    tf1_registry_path = (
        ROOT / "policy/dynamic/foreground_contract_registry.py"
    )
    candidate_hashes = {
        "range_image_foreground_v2_1.py": sha256(tf1_candidate_path),
        "foreground_contract_registry.py": sha256(tf1_registry_path),
    }
    if candidate_hashes["range_image_foreground_v2_1.py"] != (
        tf1_freeze["source_sha256"]
    ):
        mismatches["tf1_candidate_hash"] = {
            "expected": tf1_freeze["source_sha256"],
            "actual":
                candidate_hashes["range_image_foreground_v2_1.py"],
        }
    if candidate_hashes["foreground_contract_registry.py"] != (
        tf1_freeze["registry_sha256"]
    ):
        mismatches["tf1_registry_hash"] = {
            "expected": tf1_freeze["registry_sha256"],
            "actual":
                candidate_hashes["foreground_contract_registry.py"],
        }
    entry = {
        "status": "PASS" if not mismatches else "FAIL",
        "phase": "phase8jqv2_4_dynamic_perception_architecture_review",
        "tf1_route": "D",
        "tf1_primary_cause": tf1_final["primary_cause"],
        "candidate_selected": tf1_final["candidate_selected"],
        "holdout_accessed": tf1_final["holdout_accessed"],
        "legacy_hashes": current_hashes,
        "tf1_candidate_hashes": candidate_hashes,
        "tracker_algorithm_hash": current_hashes["track_manager.py"],
        "official_dynamic_sensor_source": "depth",
        "pointcloud_sensor_enabled": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "annex_used": False,
        "new_maps_generated": False,
        "mismatches": mismatches,
    }
    write_new(REPORTS / "phase8jqv2_4dpar1_entry_gate.json", entry)
    if mismatches:
        raise RuntimeError(f"DPAR1 entry mismatch: {mismatches}")
    split = {
        "status": "FROZEN_REUSE",
        "source": "reports/phase8jqv2_4tf1_evaluation_split.json",
        "source_split_hash": tf1_split["split_hash"],
        "development_map_count": tf1_split["development_map_count"],
        "holdout_map_count": tf1_split["holdout_map_count"],
        "development_case_count":
            len(tf1_split["development_natural_cases"]),
        "sealed_holdout_case_count":
            len(tf1_split["sealed_holdout_natural_cases"]),
        "holdout_results_accessed": False,
        "holdout_contents_evaluated": False,
        "maps_reassigned": False,
        "seeds_added": False,
        "new_maps_generated": False,
        "formal_used": False,
        "test_used": False,
        "blind_used": False,
    }
    write_new(REPORTS / "phase8jqv2_4dpar1_evaluation_split.json", split)

    contract = yaml.safe_load((
        ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml"
    ).read_text())
    fx = fy = 80.0
    detection_distance = float(
        contract["sampling"]["maximum_detection_center_distance_m"]
    )
    radii = sorted({
        float(contract["sampling"]["actor_radius_m"]),
        float(contract["scenarios"]["multi_target"]["actor_radius_m"]),
    })
    formal_projection = {
        str(radius): projected_sphere(fx, radius, detection_distance)
        for radius in radii
    }
    depths, positions, yaws, actor_positions, masks = synthetic_fixture(
        "small_projection", "cpu"
    )
    visible_pixels = [int(mask.sum()) for mask in masks]
    patch_depth = float(np.median(depths[masks]))
    background_depth = 10.0
    size_pixels = int(round(math.sqrt(max(visible_pixels))))
    equivalent_radius = (
        patch_depth * (size_pixels / 2.0) / fx
    )
    pixel_centers = []
    for mask in masks:
        vv, uu = np.nonzero(mask)
        pixel_centers.append([
            float(np.mean(uu)) if len(uu) else None,
            float(np.mean(vv)) if len(vv) else None,
        ])
    image_velocity_px_s = (
        (pixel_centers[1][0] - pixel_centers[0][0]) / 0.1
    )
    inferred_lateral_velocity = (
        image_velocity_px_s * patch_depth / fx
    )
    actor_metadata_speed = float(np.linalg.norm(
        actor_positions[1] - actor_positions[0]
    ) / 0.1)
    actor_metadata_distance = [
        float(np.linalg.norm(row-position))
        for row, position in zip(actor_positions, positions)
    ]
    small = {
        "classification": "fixture_semantics_invalid",
        "historical_tf1_role": "hard_positive",
        "revised_role": "invalid_not_eligible_for_gate_or_diagnostic",
        "actor_physical_radius_m": None,
        "actor_radius_encoded_by_renderer": False,
        "depth_patch_shape": [size_pixels, size_pixels],
        "single_frame_projected_area_pixels": max(visible_pixels),
        "visible_pixels": visible_pixels,
        "patch_depth_m": patch_depth,
        "background_depth_m": background_depth,
        "depth_contrast_m": background_depth-patch_depth,
        "fx": fx, "fy": fy,
        "motion_direction": "image_tangential",
        "image_velocity_pixels_per_second": image_velocity_px_s,
        "inferred_lateral_velocity_at_patch_depth_mps":
            inferred_lateral_velocity,
        "metadata_actor_speed_mps": actor_metadata_speed,
        "metadata_actor_camera_distance_range_m": [
            min(actor_metadata_distance), max(actor_metadata_distance)
        ],
        "metadata_position_matches_depth_patch": False,
        "equivalent_radius_from_square_half_width_m":
            equivalent_radius,
        "formal_minimum_actor_radius_m": min(radii),
        "formal_default_actor_radius_m": max(radii),
        "formal_detection_center_distance_max_m": detection_distance,
        "inside_detection_envelope": False,
        "inside_formal_radius_range": False,
        "inside_motion_contract": False,
        "formal_projection_at_detection_boundary": formal_projection,
        "failure_reasons": [
            "fixture is a square depth overwrite, not a rendered actor",
            "no physical actor radius is defined",
            "4.0 m patch depth exceeds the 1.8 m detection envelope",
            "equivalent 0.10 m radius is below formal minimum 0.20 m",
            "pixel motion implies 3.0 m/s but metadata implies about 1.54 m/s",
            "metadata actor world position is not calibrated to patch depth/pixel",
        ],
    }

    fov_depths, fov_positions, fov_yaws, _, fov_masks = negative_fixture(
        "fov_boundary_change", "cpu"
    )
    valid = np.isfinite(fov_depths) & (fov_depths > 0.1) & (
        fov_depths < 20.0
    )
    changed = np.abs(fov_depths[5] - fov_depths[4]) > 0
    newly_valid = valid[5] & ~valid[4]
    newly_invalid = ~valid[5] & valid[4]
    fov = {
        "status": "FAIL",
        "classification": "fixture_semantics_invalid",
        "contains_dynamic_actor": bool(np.any(fov_masks)),
        "camera_translation_max_m": float(np.max(np.linalg.norm(
            np.diff(fov_positions, axis=0), axis=1
        ))),
        "camera_yaw_delta_max_rad": float(np.max(np.abs(
            np.diff(fov_yaws)
        ))),
        "depth_changed_pixels_at_transition": int(changed.sum()),
        "changed_pixel_bbox_uv": [0, 30, 1, 49],
        "newly_valid_pixels_from_depth_validity": int(newly_valid.sum()),
        "newly_invalid_pixels_from_depth_validity":
            int(newly_invalid.sum()),
        "all_depths_valid": bool(valid.all()),
        "static_geometry_renderer_used": False,
        "warp_fov_transition_caused_by_camera_motion": False,
        "manual_depth_overwrite": {
            "frames": "5:end", "rows": "30:50", "columns": "0:2",
            "old_depth_m": 10.0, "new_depth_m": 8.0,
        },
        "hard_negative_semantics_verified": False,
        "failure_reasons": [
            "camera position and yaw are constant",
            "no static authority geometry or camera-motion render is used",
            "all pixels remain valid, so newly-valid/newly-invalid count is zero",
            "the fixture is a manual left-border depth change, not an FOV event",
        ],
    }
    write_new(
        REPORTS
        / "phase8jqv2_4dpar1_fov_boundary_fixture_validation.json",
        fov,
    )
    control = {
        "status": "FAIL",
        "control_contract_version":
            "dynamic_perception_control_contract_audit_v1",
        "small_projection": small,
        "fov_boundary_change": fov,
        "small_projection_semantics_valid": False,
        "fov_boundary_semantics_valid": False,
        "architecture_design_allowed": False,
        "holdout_access_allowed": False,
        "primary_cause": "dynamic_perception_control_contract_invalid",
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_control_contract_repair",
    }
    write_new(
        REPORTS / "phase8jqv2_4dpar1_control_contract_audit.json",
        control,
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_small_projection_contract.md",
        f"""# DPAR1 small-projection contract audit

## Result

`fixture_semantics_invalid`.

The TF1 fixture writes a moving **4×4 square** at a constant depth of 4.0 m.
It does not render a sphere and defines no physical actor radius. Its 16
visible pixels therefore cannot be mapped authoritatively to the frozen actor
contract.

## Frozen physical bound

The camera uses `fx = fy = 80 px`. The smallest formal actor radius is
`r = 0.20 m`, and the maximum required detection center distance is
`d = 1.80 m`. For a sphere, the pinhole silhouette diameter is

`D = 2 f r / sqrt(d²-r²)`.

This gives `{formal_projection[str(min(radii))]["diameter_pixels"]:.6f} px`
and a continuous projected disc area of
`{formal_projection[str(min(radii))]["continuous_disc_area_pixels2"]:.6f} px²`.
The default `r = 0.30 m` actor gives
`{formal_projection[str(max(radii))]["diameter_pixels"]:.6f} px`.

By contrast, interpreting half of the 4 px square width as a pinhole angular
radius at 4.0 m gives only `{equivalent_radius:.6f} m`, below the formal
minimum. The patch is also beyond the 1.8 m detection envelope.

The fixture's 6 px/frame motion at 10 Hz implies
`{inferred_lateral_velocity:.6f} m/s` at 4 m, whereas its disconnected
metadata trajectory implies `{actor_metadata_speed:.6f} m/s`. The metadata
position is not calibrated to the patch depth and pixels.

The historical TF1 report is preserved. This control must be replaced by a
versioned, CUDA-rendered physical actor suite before architecture prototyping.
It must not simply be deleted or silently demoted.
""",
    )
    diagnostics = ROOT / "diagnostics/phase8jqv2_4dpar1/control_audit"
    write_new(diagnostics / "small_projection_observed.json", small)
    write_new(diagnostics / "fov_boundary_observed.json", fov)

    final = {
        "status": "FAIL",
        "architecture_review": "NOT_RUN_CONTROL_GATE_FAILED",
        "primary_cause": "dynamic_perception_control_contract_invalid",
        "selected_candidate": None,
        "candidate_selected": False,
        "legacy_default_changed": False,
        "tf1_candidates_modified": False,
        "tracker_modified": False,
        "holdout_accessed": False,
        "architecture_prototype_created": False,
        "formal_preflight_rerun": False,
        "formal_generation_started": False,
        "test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "gap_3": "OUT_OF_SCOPE_CONTRACT_REVIEW_REQUIRED",
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_control_contract_repair",
    }
    write_new(REPORTS / "phase8jqv2_4dpar1_final_result.json", final)
    write_new(
        REPORTS / "phase8jqv2_4dpar1_final_recommendation.md",
        """# DPAR1 final recommendation

Stop at the Stage-A control gate. Do not design or tune visibility-aware,
track-before-detect, or dual-path candidates against the current TF1 controls.

The next phase must create a versioned physical control suite:

1. render 0.20 m and 0.30 m actors with the canonical CUDA renderer at
   contract-valid distances, velocities, backgrounds, and image positions;
2. render paired static FOV-entry scenes from canonical geometry and actual
   camera pose changes;
3. independently verify warp newly-visible/newly-invalid provenance;
4. freeze the repaired controls before any architecture parameters exist.

TF1 reports remain valid as historical evidence of what those fixtures tested,
but their two control labels are not physically authoritative.
""",
    )
    write_new(
        REPORTS / "phase8jqv2_4dpar1_final_readiness.md",
        """# DPAR1 readiness

**FAIL — control contract repair required.**

- DPAR1 entry: PASS.
- Official runtime sensor: depth.
- TF1/N1/I1 and all frozen source hashes: preserved.
- Small-projection physical semantics: FAIL.
- FOV-boundary fixture semantics: FAIL.
- Architecture prototypes: correctly not created.
- Sealed holdout: not accessed.
- Tracker integration: not run.
- Formal/test/blind/training: not run.

Next allowed phase:
`phase8jqv2_4_dynamic_perception_control_contract_repair`.
""",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
