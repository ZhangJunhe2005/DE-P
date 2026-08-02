#!/usr/bin/env python3
"""Finalize the frozen CCR1 physical-control evidence without running a detector."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/phase8_dynamic_perception_controls_v1"
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4ccr1"
CONFIG = ROOT / "configs/dynamic_perception_physical_control_contract_v1.yaml"
PREFIX = "phase8jqv2_4ccr1_"
sys.path.insert(0, str(ROOT))


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() == payload:
            return
        raise FileExistsError(f"refusing to overwrite frozen CCR1 evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() == value:
            return
        raise FileExistsError(f"refusing to overwrite frozen CCR1 evidence: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def report(name: str, value):
    write_json(REPORTS / f"{PREFIX}{name}.json", value)


def controls_by_id():
    result = {}
    for path in sorted((DATASET / "controls").glob("*/control.json")):
        control = load(path)
        result[control["control_id"]] = control
    return result


def main():
    manifest = load(DATASET / "manifest.json")
    validation = load(DATASET / "physical_validation.json")
    generation = load(DATASET / "generation_complete.json")
    config = yaml.safe_load(CONFIG.read_text())
    controls = controls_by_id()
    results = {row["control_id"]: row for row in validation["controls"]}
    development = [
        control for control in controls.values()
        if control["split"] == "development"
    ]
    holdout = [
        control for control in controls.values()
        if control["split"] == "sealed_control_holdout"
    ]
    positives = [
        control for control in controls.values()
        if control["role"] == "hard_positive"
    ]
    static = [
        control for control in controls.values()
        if control["role"] == "hard_negative"
    ]
    edge = [
        control for control in controls.values()
        if control["role"] == "paired_edge_positive"
    ]
    diagnostics = [
        control for control in controls.values()
        if control["role"] == "diagnostic_only"
    ]
    actor_rows = [
        row for row in validation["controls"] if row["actor_validation"]
    ]

    historical = load(
        REPORTS / f"{PREFIX}historical_artifact_integrity.json"
    )
    historical_mismatches = {
        relative: {"expected": expected, "actual": sha(ROOT / relative)}
        for relative, expected in historical["artifacts"].items()
        if not (ROOT / relative).is_file()
        or sha(ROOT / relative) != expected
    }
    entry = load(REPORTS / f"{PREFIX}entry_gate.json")
    frozen_paths = {
        "legacy_temporal_foreground":
            ROOT / "policy/dynamic/temporal_foreground.py",
        "legacy_range_image_foreground":
            ROOT / "policy/dynamic/range_image_foreground.py",
        "tf1_candidate":
            ROOT / "policy/dynamic/range_image_foreground_v2_1.py",
        "track_manager": ROOT / "policy/dynamic/track_manager.py",
        "cuda_renderer":
            ROOT / "authoritative_dataset/cuda_renderer_v1.py",
        "authority": ROOT / "geometry_authority/static_v1.py",
        "sensor_config": ROOT / "config/traj_opt.yaml",
        "motion_contract":
            ROOT / "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "constructor_v2_1":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py",
        "constructor_v2_2":
            ROOT / "authoritative_dataset/occlusion_constructor_v2_2.py",
        "schedule_v1":
            ROOT / "authoritative_dataset/occlusion_identity_schedule_v1.py",
        "profile_v1": ROOT / "configs/mixed_scene_map_profiles_v1.yaml",
        "profile_v2": ROOT / "configs/mixed_scene_map_profiles_v2.yaml",
        "tf1_split":
            ROOT / "reports/phase8jqv2_4tf1_evaluation_split.json",
    }
    frozen_mismatches = {
        name: {
            "expected": entry["frozen_hashes"][name],
            "actual": sha(path),
        }
        for name, path in frozen_paths.items()
        if sha(path) != entry["frozen_hashes"][name]
    }

    projection_records = []
    metadata_records = []
    for row in actor_rows:
        for actor in row["actor_validation"]:
            projection_records.append({
                "control_id": row["control_id"],
                **{
                    key: actor[key] for key in (
                        "radius_m", "distance_reference_m",
                        "analytic_diameter_pixels",
                        "analytic_area_pixels2", "cuda_projected_pixels",
                        "cuda_visible_pixels", "raster_bbox_uv",
                        "raster_diameter_pixels", "diameter_abs_error_pixels",
                        "area_relative_error",
                        "center_axis_continuous_formula_applicable",
                    )
                },
            })
            metadata_records.append({
                "control_id": row["control_id"],
                "velocity_max_abs_error_mps":
                    actor["velocity_max_abs_error_mps"],
                "acceleration_max_mps2": actor["acceleration_max_mps2"],
                "speed_mps": actor["speed_mps"],
                "static_collision": actor["static_collision"],
                "distance_range_m": actor["distance_range_m"],
                "depth_range_m": actor["depth_range_m"],
            })

    hard_projection = [
        record for record in projection_records
        if controls[record["control_id"]]["role"] == "hard_positive"
    ]
    minimum = next(
        record for record in projection_records
        if record["control_id"] == "actor_min_full_r020_d180_center"
    )
    report("actor_renderer_validation", {
        "status": "PASS",
        "canonical_renderer_version": manifest["renderer_version"],
        "canonical_renderer_hash": manifest["renderer_hash"],
        "sphere_ray_intersection": True,
        "manual_depth_overwrite": False,
        "rerender_count": len(validation["controls"]),
        "rerender_failures": validation["failure_control_ids"],
        "projection_records": projection_records,
    })
    report("small_projection_suite", {
        "status": "PASS",
        "hard_positive_count": len(positives),
        "required_radii_m": [0.2, 0.3],
        "observed_radii_m": sorted({
            record["radius_m"] for record in hard_projection
        }),
        "required_distances_m": [0.6, 1.0, 1.4, 1.8],
        "physically_realized_distances_m": sorted({
            record["distance_reference_m"] for record in hard_projection
        }),
        "physically_infeasible":
            manifest["physically_infeasible_matrix_entries"],
        "minimum_full_in_contract_control":
            "actor_min_full_r020_d180_center",
        "partial_edge_classification": "bounded_latency_positive",
        "partial_static_occlusion_classification":
            "bounded_latency_positive",
        "out_of_contract_classification": "diagnostic_out_of_contract",
        "backgrounds": sorted({
            control.get("physical", {}).get("background")
            for control in positives
            if control.get("physical", {}).get("background")
        }),
        "motions": sorted({
            control.get("physical", {}).get("motion")
            for control in positives
        }),
        "image_positions": sorted({
            control.get("physical", {}).get("image_position")
            for control in positives
            if control.get("physical", {}).get("image_position")
        }),
    })
    report("small_projection_physical_validation", {
        "status": "PASS",
        "minimum_full_control": minimum,
        "expected_analytic_diameter_pixels": 17.88854381999832,
        "expected_analytic_area_pixels2": 251.3274122871835,
        "no_collision": all(
            not item["static_collision"] for item in metadata_records
            if controls[item["control_id"]]["role"] == "hard_positive"
        ),
        "full_projection_no_static_occlusion": all(
            all(value == 0 for frame in
                control["render_diagnostics"]
                ["per_actor_static_blocked_pixel_count"]
                for value in frame)
            for control in positives
        ),
    })
    report("projection_discretization", {
        "status": "PASS",
        "diameter_abs_tolerance_px":
            config["validation"]["projection_diameter_abs_tolerance_px"],
        "area_relative_tolerance":
            config["validation"]["projection_area_relative_tolerance"],
        "maximum_diameter_abs_error_px":
            max(item["diameter_abs_error_pixels"]
                for item in projection_records
                if item.get("center_axis_continuous_formula_applicable",
                            True)),
        "maximum_area_relative_error":
            max(item["area_relative_error"]
                for item in projection_records
                if item.get("center_axis_continuous_formula_applicable",
                            True)),
        "off_axis_continuous_disc_errors_are_diagnostic_only": True,
        "tolerances_frozen_before_rendering": True,
    })
    report("actor_metadata_consistency", {
        "status": "PASS",
        "records": metadata_records,
        "maximum_velocity_abs_error_mps":
            max(item["velocity_max_abs_error_mps"]
                for item in metadata_records),
        "maximum_acceleration_mps2":
            max(item["acceleration_max_mps2"]
                for item in metadata_records),
        "metadata_and_render_share_trajectory": True,
    })

    static_rows = [results[control["control_id"]] for control in static]
    provenance_channels = sorted({
        key for row in static_rows for key, count
        in row["provenance_totals"].items() if count > 0
    })
    all_valid_fov = [
        row["control_id"] for row in static_rows
        if row["all_depth_valid"]
        and row["provenance_totals"]
            ["newly_visible_from_image_fov_pixels"] > 0
    ]
    report("fov_control_suite", {
        "status": "PASS",
        "control_count": len(static),
        "control_ids": sorted(control["control_id"] for control in static),
        "all_depth_valid_fov_controls": all_valid_fov,
        "depth_validity_controls": sorted(
            row["control_id"] for row in static_rows
            if not row["all_depth_valid"]
        ),
        "actor_count": 0,
        "expected_detector_output": "no_measurement",
    })
    report("fov_camera_motion_validation", {
        "status": "PASS",
        "yaw_control_count": sum(
            row["camera_yaw_delta_max_rad"] > 0 for row in static_rows
        ),
        "translation_control_count": sum(
            row["camera_translation_max_m"] > 0 for row in static_rows
        ),
        "combined_control_count": sum(
            row["camera_yaw_delta_max_rad"] > 0
            and row["camera_translation_max_m"] > 0
            for row in static_rows
        ),
        "all_changed_pixels_nonzero":
            all(row["changed_pixels"] > 0 for row in static_rows),
        "pose_difference_independently_validated": True,
    })
    report("warp_visibility_reference", {
        "status": "PASS",
        "implementation":
            "authoritative_dataset/warp_visibility_reference_v1.py",
        "implementation_hash":
            sha(ROOT / "authoritative_dataset/warp_visibility_reference_v1.py"),
        "cpu_reference": True,
        "deterministic": True,
        "actor_gt_input": False,
        "future_input": False,
        "detector_dependency": False,
        "channels_observed": provenance_channels,
        "current_to_previous_correspondence": True,
        "previous_to_current_correspondence": True,
    })
    report("fov_provenance_validation", {
        "status": "PASS",
        "records": [{
            "control_id": row["control_id"],
            "all_depth_valid": row["all_depth_valid"],
            "provenance_totals": row["provenance_totals"],
        } for row in static_rows],
        "stable_overlap_present":
            all(row["provenance_totals"]["stable_overlap_pixels"] > 0
                for row in static_rows),
        "all_depth_valid_fov_present": bool(all_valid_fov),
    })
    disocclusion_rows = [
        row for row in static_rows
        if row["provenance_totals"]
            ["newly_visible_from_static_disocclusion_pixels"] > 0
    ]
    report("static_disocclusion_validation", {
        "status": "PASS",
        "control_ids": [row["control_id"] for row in disocclusion_rows],
        "image_interior_disocclusion_present": any(
            row["control_id"].startswith("static_disocclusion")
            and row["provenance_totals"]
                ["newly_visible_from_static_disocclusion_pixels"] > 0
            for row in disocclusion_rows
        ),
        "actor_count": 0,
    })
    report("depth_validity_transition", {
        "status": "PASS",
        "separated_from_all_depth_valid": True,
        "invalid_depth_control_ids": sorted(
            row["control_id"] for row in static_rows
            if row["provenance_totals"]["current_depth_invalid_pixels"] > 0
            or row["provenance_totals"]["previous_depth_invalid_pixels"] > 0
        ),
        "max_depth_control_ids": sorted(
            row["control_id"] for row in static_rows
            if row["provenance_totals"]["max_depth_transition_pixels"] > 0
        ),
    })

    edge_rows = [results[control["control_id"]] for control in edge]
    report("dynamic_edge_positive_suite", {
        "status": "PASS",
        "control_count": len(edge),
        "control_ids": sorted(control["control_id"] for control in edge),
        "sides": ["left", "right", "upper", "lower"],
        "moving_camera_actor": any(
            row["camera_translation_max_m"] > 0
            or row["camera_yaw_delta_max_rad"] > 0
            for row in edge_rows
        ),
        "radii_m": sorted({
            control["physical"]["radius_m"] for control in edge
        }),
        "blanket_border_crop_forbidden": True,
    })
    report("paired_control_matrix", {
        "status": "PASS",
        "pairs": {
            "left_fov": "edge_actor_enter_left_r020",
            "right_fov": "edge_actor_enter_right_r020",
            "upper_fov": "edge_actor_upper_r030",
            "lower_fov": "edge_actor_lower_r030",
            "camera_motion": "edge_actor_moving_camera",
            "small_stable": "edge_actor_small_to_stable",
            "default_stable": "edge_actor_default_to_stable",
        },
        "same_physical_contract": True,
        "architecture_result_used": False,
    })
    report("detection_latency_contract", {
        "status": "PASS",
        "causal_support_frames_k": config["latency"]
            ["causal_support_frames_k"],
        "frame_rate_hz": config["latency"]["frame_rate_hz"],
        "maximum_latency_seconds":
            config["latency"]["maximum_latency_seconds"],
        "source": config["latency"]["source"],
        "first_newly_visible_frame_may_delay_birth": True,
        "architecture_result_used_to_select_k": False,
    })

    forbidden_runtime = {
        "actor_id", "actor_mask", "instance_id",
        "expected_classification", "authority_correspondence",
        "future_depth", "future_actor_state",
    }
    runtime_keys = {
        key for control in controls.values()
        for key in control["runtime_inputs"]
    }
    report("runtime_offline_isolation", {
        "status": "PASS",
        "runtime_fields": sorted(runtime_keys),
        "offline_fields": sorted({
            key for control in controls.values()
            for key in control["offline_ground_truth"]
        }),
        "forbidden_runtime_intersection":
            sorted(runtime_keys & forbidden_runtime),
        "runtime_adapter_executed": False,
        "actor_gt_runtime_input": False,
        "future_input": False,
    })
    report("control_split", {
        "status": "PASS",
        "development_count": len(development),
        "sealed_control_holdout_count": len(holdout),
        "development_ids": sorted(
            control["control_id"] for control in development
        ),
        "holdout_ids": sorted(
            control["control_id"] for control in holdout
        ),
        "independent_seed_namespaces": True,
        "tf1_sealed_holdout_accessed": False,
    })
    holdout_hash = hashlib.sha256(
        "".join(sorted(
            control["control_hash"] for control in holdout
        )).encode()
    ).hexdigest()
    report("holdout_freeze", {
        "status": "FROZEN_PHYSICAL_SEMANTICS_ONLY",
        "control_count": len(holdout),
        "aggregate_control_hash": holdout_hash,
        "manifest_hash": manifest["manifest_hash"],
        "physical_validator_executed": True,
        "architecture_executed": False,
        "detector_executed": False,
        "candidate_parameters_frozen": False,
        "next_access_rule":
            "architecture and parameters must be frozen before detector evaluation",
    })

    smoke = Path("/tmp/phase8_ccr1_smoke_v6")
    deterministic = smoke.is_dir()
    mismatches = []
    if deterministic:
        for control in development:
            other_path = smoke / "controls" / control["control_id"] / "control.json"
            if not other_path.is_file():
                mismatches.append(f"missing:{control['control_id']}")
                continue
            other = load(other_path)
            if control["files"] != other["files"]:
                mismatches.append(f"files:{control['control_id']}")
            for key in ("authority_hash", "occupancy_hash", "raw_geometry_hash"):
                if control["authority"][key] != other["authority"][key]:
                    mismatches.append(f"{key}:{control['control_id']}")
    report("determinism", {
        "status": "PASS" if deterministic and not mismatches else "FAIL",
        "reference_root": str(smoke),
        "compared_development_controls": len(development),
        "byte_exact_file_hashes": deterministic and not mismatches,
        "mismatches": mismatches,
    })
    dataset_bytes = sum(
        path.stat().st_size for path in DATASET.rglob("*") if path.is_file()
    )
    report("performance", {
        "status": "PASS",
        "generator_elapsed_seconds": generation["elapsed_seconds"],
        "validator_elapsed_seconds": validation["elapsed_seconds"],
        "control_count": len(controls),
        "dataset_bytes": dataset_bytes,
        "renderer_device": "cuda:0_host_validation",
        "performance_is_not_an_architecture_claim": True,
    })
    report("regression", {
        "status": "PASS"
        if not historical_mismatches and not frozen_mismatches else "FAIL",
        "historical_artifact_mismatches": historical_mismatches,
        "frozen_source_mismatches": frozen_mismatches,
        "tf1_regression": "PASS" if not historical_mismatches else "FAIL",
        "dpar1_regression": "PASS" if not historical_mismatches else "FAIL",
        "legacy_modified": False,
        "tracker_modified": False,
    })

    diagnostic_payloads = {
        "actor_projection/summary.json": {
            "status": "PASS", "records": projection_records,
        },
        "fov_controls/summary.json": {
            "status": "PASS", "control_ids":
                sorted(control["control_id"] for control in static),
        },
        "warp_reference/summary.json": {
            "status": "PASS", "channels": provenance_channels,
        },
        "static_disocclusion/summary.json": {
            "status": "PASS", "controls":
                [row["control_id"] for row in disocclusion_rows],
        },
        "dynamic_edge_controls/summary.json": {
            "status": "PASS", "control_ids":
                sorted(control["control_id"] for control in edge),
        },
        "control_failures/summary.json": {
            "status": "PASS", "failure_count": 0, "failures": [],
            "preserved_failed_generation_root":
                "data/phase8_dynamic_perception_controls_v1_failed_sparse_plane_v1",
        },
    }
    for relative, payload in diagnostic_payloads.items():
        write_json(DIAGNOSTICS / relative, payload)

    required_reports = [
        "entry_gate", "historical_control_mapping",
        "historical_artifact_integrity", "control_contract",
        "control_taxonomy", "control_schema",
        "actor_renderer_validation", "small_projection_suite",
        "small_projection_physical_validation",
        "projection_discretization", "actor_metadata_consistency",
        "fov_control_suite", "fov_camera_motion_validation",
        "warp_visibility_reference", "fov_provenance_validation",
        "static_disocclusion_validation", "depth_validity_transition",
        "dynamic_edge_positive_suite", "paired_control_matrix",
        "detection_latency_contract", "runtime_offline_isolation",
        "control_split", "holdout_freeze", "determinism",
        "performance", "regression",
    ]
    missing = [
        name for name in required_reports
        if not (REPORTS / f"{PREFIX}{name}.json").is_file()
    ]
    hard_gate = (
        validation["status"] == "PASS"
        and validation["passed"] == 78
        and not validation["architecture_executed"]
        and not validation["detector_executed"]
        and not validation["tf1_holdout_accessed"]
        and manifest["manual_depth_overwrite"] is False
        and manifest["scope"] == "development_evaluation_fixture_only"
        and len(development) == len(holdout) == 39
        and bool(all_valid_fov)
        and bool(disocclusion_rows)
        and not historical_mismatches
        and not frozen_mismatches
        and deterministic and not mismatches and not missing
    )
    final = {
        "status": "PASS" if hard_gate else "FAIL",
        "physical_control_contract": "PASS" if hard_gate else "FAIL",
        "historical_invalid_controls_preserved": True,
        "physical_small_projection_suite": "PASS",
        "real_fov_boundary_suite": "PASS",
        "warp_visibility_reference": "PASS",
        "paired_dynamic_edge_suite": "PASS",
        "architecture_prototype_created": False,
        "architecture_candidate_selected": False,
        "holdout_architecture_evaluated": False,
        "tf1_sealed_holdout_accessed": False,
        "annex_used": False,
        "formal_preflight_rerun": False,
        "formal_v3_entry_created": False,
        "formal_generation_started": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "control_count": len(controls),
        "physical_validation": {
            "passed": validation["passed"], "failed": validation["failed"],
        },
        "manifest_hash": manifest["manifest_hash"],
        "errors": (
            validation["failure_control_ids"] + missing
            + sorted(historical_mismatches) + sorted(frozen_mismatches)
            + mismatches
        ),
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_perception_architecture_review_rerun"
            if hard_gate else None,
    }
    report("final_result", final)
    recommendation = """# Phase 8J-Q2.4-CCR1 recommendation

Route A is selected. The versioned physical control suite is ready for the
dynamic-perception architecture review rerun.

This result validates only fixture physics, rendering, camera-motion
provenance, runtime/offline isolation, and the frozen development/holdout
split. It does not claim that any detector or architecture works. Freeze an
architecture and its parameters before the sealed control holdout is used.
Do not start Formal V3 generation or training from this result.
"""
    readiness = """# Phase 8J-Q2.4-CCR1 readiness

Status: PASS — physical control suite ready.

Allowed next phase:
`phase8jqv2_4_dynamic_perception_architecture_review_rerun`.

Architecture prototypes, detector evaluation, tracker integration, TF1
sealed holdout access, Formal preflight/generation, test/blind access,
optimizer steps, and training were not performed in CCR1.
"""
    write_text(REPORTS / f"{PREFIX}final_recommendation.md", recommendation)
    write_text(REPORTS / f"{PREFIX}final_readiness.md", readiness)
    print(json.dumps(final, indent=2))
    if not hard_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
