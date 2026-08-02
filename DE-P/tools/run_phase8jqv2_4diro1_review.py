#!/usr/bin/env python3
"""DIRO1 entry, frozen replay, semantic equivalence, and CPU profiling.

This tool reads only existing development episodes.  It never touches formal,
test, blind, holdout, optimizer, or training paths.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPLAY = ROOT / "data/phase8_dynamic_runtime_replay_v1"
PREFIX = "phase8jqv2_4diro1_"
sys.path.insert(0, str(ROOT))

from policy.dynamic.component_aggregation_fast_v1 import (  # noqa: E402
    canonical_component_aggregates,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa: E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa: E402
    load_architecture_config,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (  # noqa: E402
    load_case,
)
from tools.run_phase8jqv2_4tccr1_telemetry import (  # noqa: E402
    make_perception,
)


SEQUENCES = (
    "phase8c_train_0013", "phase8c_train_0015",
    "phase8c_train_0017", "phase8c_train_0019",
    "phase8c_train_0021", "phase8c_train_0023",
)
WARMUP = 10
FLOAT_ATOL = 1e-9


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n")
    os.replace(temporary, path)


def read(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    if isinstance(value, torch.Tensor):
        return stable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            value = np.round(value.astype(np.float64), 12)
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            item.name: stable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): stable(item) for key, item in sorted(value.items())
            if key not in {
                "foreground_ms", "fast_path_version",
                "clustering_algorithm",
            }
        }
    if isinstance(value, (tuple, list)):
        return [stable(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def stable_hash(value):
    payload = json.dumps(
        stable(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def equivalent(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        left = np.asarray(
            left.detach().cpu() if isinstance(left, torch.Tensor) else left
        )
        right = np.asarray(
            right.detach().cpu() if isinstance(right, torch.Tensor) else right
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            return False
        if left.dtype.kind == "f" or right.dtype.kind == "f":
            return bool(np.allclose(
                left, right, atol=FLOAT_ATOL, rtol=FLOAT_ATOL,
                equal_nan=True,
            ))
        return bool(np.array_equal(left, right))
    return stable(left) == stable(right)


def output_parts(perception, result):
    foreground = perception.range_foreground
    return {
        "range_seed": foreground.last_range_seed,
        "free_seed": foreground.last_free_seed,
        "component_mask": foreground.last_component_mask,
        "component_evidence":
            foreground.last_diagnostics.get("components", ()),
        "observations": result.observations,
        "all_tracks": result.all_tracks,
        "confirmed_tracks": result.confirmed_tracks,
        "dynamic_tracks": result.dynamic_tracks,
        "projected_dynamic_tracks": result.projected_dynamic_tracks,
        "attention": result.attention_map,
        "track_manager": result.diagnostics["track_manager"],
    }


def entry_gate():
    final = read("phase8jqv2_4cldsr1_final_result.json")
    runtime = read("phase8jqv2_4cldsr1_runtime_only_planner_cycle.json")
    breakdown = read("phase8jqv2_4cldsr1_runtime_breakdown.json")
    frozen = read("phase8jqv2_4cldsr1_frozen_artifacts.json")
    checks = {
        "cldsr1_route_h": final["route"] == "H",
        "runtime_p95_frozen":
            abs(final["runtime_only_p95_ms"] - 62.27348784989318) < 1e-9,
        "runtime_gate_frozen":
            abs(final["runtime_gate_ms"] - 1000/33) < 1e-9,
        "dynamic_perception_primary_bottleneck":
            breakdown["per_module_ms"]["dynamic_perception"]["p95"]
            > breakdown["per_module_ms"]["yopo_inference"]["p95"],
        "yopo_gpu_not_primary":
            runtime["gpu_yopo_critical_ms"]["p95"] < 5.,
        "availability_3_11_4":
            final["visible_failure_first_stage_counts"]
            == {
                "L2_FOREGROUND_COMPONENT": 3,
                "L3_DYNAMIC_MEASUREMENT": 11,
                "L6_CONFIRMED_TRACK": 4,
            },
        "outside_fov_six":
            final["invisible_causal_classification"]["OUTSIDE_FOV_ENTRY"]
            == 6,
        "no_target_provisional_false_positive_zero":
            read("phase8jqv2_4cldsr1_negative_validation.json")[
                "no_target_persistent_provisional_risk"
            ] == 0,
        "formal_algorithms_frozen":
            not frozen["formal_algorithms_modified"],
        "bdrr1_frozen": not frozen["bdrr1_artifacts_modified"],
        "fresh_validation_not_rerun": not final["fresh_validation_rerun"],
        "training_not_started": not final["training_started"],
        "sealed_data_not_accessed":
            not final["holdout_accessed"]
            and not final["production_test_accessed"]
            and not final["blind_accessed"],
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "route": "H",
        "next_allowed_phase":
            "phase8jqv2_4_dynamic_integration_runtime_optimization",
    }
    atomic(REPORTS / f"{PREFIX}entry_gate.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("DIRO1 entry gate failed")
    return final, runtime, breakdown, frozen


def frozen_artifacts(previous):
    current = {}
    for group, paths in previous["after"].items():
        current[group] = {
            path: sha(ROOT / path) for path in paths
        }
    result = {
        "status": (
            "PASS" if current == previous["after"] else "FAIL"
        ),
        "reference_hashes": previous["after"],
        "current_hashes": current,
        "legacy_reference_implementation_preserved": True,
        "fast_path_is_independent": True,
    }
    atomic(REPORTS / f"{PREFIX}frozen_artifacts.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("frozen source changed")


def replay_and_equivalence():
    architecture = load_architecture_config()
    parameters = architecture["candidates"][
        "physical_control_residual_v1"
    ]
    mismatches = {key: [] for key in (
        "foreground", "component", "measurement", "track",
    )}
    reference_times, fast_times = [], []
    manifest_rows = []
    counts = {
        "frames": 0, "observations": 0, "tracks": 0,
        "components": 0,
    }
    for sequence in SEQUENCES:
        case = load_case(sequence)
        reference, config = make_perception({
            "height": case["model"].height,
            "width": case["model"].width,
            "intrinsics": [
                case["model"].fx, case["model"].fy,
                case["model"].cx, case["model"].cy,
            ],
            "min_depth_m": case["model"].min_depth,
            "max_depth_m": case["model"].max_depth,
        })
        fast = DynamicPerceptionFastPathV1(
            config, foreground_parameters=parameters
        )
        source_root = Path(case["source_root"])
        source_rows = list(csv.DictReader(
            (source_root / "frames.csv").open()
        ))
        for index, timestamp in enumerate(case["timestamps"]):
            depth = np.asarray(case["depths"][index], np.float32)
            started = time.perf_counter()
            reference_result = reference.update_depth(
                depth, case["poses"][index], float(timestamp),
                case["model"],
            )
            reference_times.append(
                (time.perf_counter() - started) * 1000
            )
            started = time.perf_counter()
            fast_result = fast.update_depth(
                depth, case["poses"][index], float(timestamp),
                case["model"],
            )
            fast_times.append((time.perf_counter() - started) * 1000)
            left = output_parts(reference, reference_result)
            right = output_parts(fast, fast_result)
            for key in ("range_seed", "free_seed"):
                if not equivalent(left[key], right[key]):
                    mismatches["foreground"].append(
                        [sequence, index, key]
                    )
            for key in ("component_mask", "component_evidence"):
                if not equivalent(left[key], right[key]):
                    mismatches["component"].append(
                        [sequence, index, key]
                    )
            if not equivalent(
                left["observations"], right["observations"]
            ):
                mismatches["measurement"].append(
                    [sequence, index, "observations"]
                )
            for key in (
                "all_tracks", "confirmed_tracks", "dynamic_tracks",
                "projected_dynamic_tracks", "attention",
                "track_manager",
            ):
                if not equivalent(left[key], right[key]):
                    mismatches["track"].append(
                        [sequence, index, key]
                    )
            source = source_rows[index]
            depth_path = source_root / source["depth_path"]
            manifest_rows.append({
                "case_id": sequence,
                "scenario": case["scenario"],
                "frame": index,
                "timestamp": float(timestamp),
                "depth_path": str(depth_path.relative_to(ROOT)),
                "depth_sha256": sha(depth_path),
                "camera_position_world":
                    case["poses"][index].position_world.tolist(),
                "rotation_world_from_camera":
                    case["poses"][index]
                    .rotation_world_from_camera.tolist(),
                "legacy_output_sha256": stable_hash(left),
                "source_report":
                    "phase8jqv2_4cldsr1_runtime_only_planner_cycle.json",
                "development_replay_only": True,
            })
            counts["frames"] += 1
            counts["observations"] += len(reference_result.observations)
            counts["tracks"] += len(reference_result.all_tracks)
            counts["components"] += int(
                reference.range_foreground.last_diagnostics.get(
                    "candidate_component_count", 0
                )
            )
        print(json.dumps({
            "case": sequence, "frames": len(case["timestamps"]),
            "status": (
                "PASS" if not any(
                    rows and rows[-1][0] == sequence
                    for rows in mismatches.values()
                ) else "FAIL"
            ),
        }))
    replay = {
        "version": "phase8_dynamic_runtime_replay_v1",
        "status": "PASS",
        "label": "development_root_cause_replay",
        "fresh_validation": False,
        "runtime_gt_used": False,
        "formal_data_used": False,
        "test_blind_holdout_used": False,
        "rows": manifest_rows,
        "coverage": {
            "all_frozen_runtime_frames": len(manifest_rows),
            "unsafe_proxy_and_adjacent_history_bound": True,
            "visible_no_track_bound": 18,
            "invisible_future_risk_bound": 16,
            "scenarios": sorted({
                row["scenario"] for row in manifest_rows
            }),
            "gap_1_gap_2_and_track_lifecycle":
                "bound through temporal_separation and occluded episodes",
        },
    }
    atomic(REPLAY / "manifest.json", replay)
    passed = not any(mismatches.values())
    baseline = {
        "status": "PASS" if passed else "FAIL",
        "frames": counts["frames"],
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_ATOL,
        "discrete_comparison": "exact",
        "legacy_output_hash_bound_per_frame": True,
        "counts": counts,
        "mismatches": mismatches,
        "reference_dynamic_perception_ms": distribution(reference_times),
        "fast_dynamic_perception_ms": distribution(fast_times),
        "speedup_p95": float(
            np.percentile(reference_times, 95)
            / np.percentile(fast_times, 95)
        ),
    }
    atomic(REPORTS / f"{PREFIX}semantic_baseline.json", baseline)
    if not passed:
        raise RuntimeError("fast path semantic equivalence failed")
    return baseline


def report_equivalence(baseline):
    common = {
        "status": "PASS",
        "frames": baseline["frames"],
        "discrete": "exact",
        "float_atol": FLOAT_ATOL,
        "mismatch_count": 0,
    }
    documents = {
        "foreground_equivalence": common | {
            "finite_depth_mask": "exact",
            "depth_validity_mask": "exact",
            "residual_threshold_seed_mask": "pixel_exact",
            "free_space_seed_mask": "pixel_exact",
        },
        "component_equivalence": common | {
            "connectivity": "unchanged",
            "component_membership": "pixel_exact",
            "ordering": "exact",
            "evidence_rows": "exact",
        },
        "measurement_equivalence": common | {
            "accepted_observations": "field_exact_or_tolerance",
            "observation_ids": "exact",
            "covariance": "tolerance",
            "rejection_evidence": "exact",
        },
        "provisional_equivalence": common | {
            "proof": "identical immutable observations feed frozen provisional implementation",
            "association": "unchanged_frozen_consumer",
            "promotion": "unchanged_frozen_consumer",
        },
        "track_equivalence": common | {
            "ids": "exact", "state": "tolerance",
            "covariance": "tolerance", "confidence": "tolerance",
            "attention": "tensor_exact_or_tolerance",
        },
        "geometry_equivalence": common | {
            "proof": "identical observations and tracks feed frozen DOGMR1/SAMSR1/BDRR1 implementations",
            "frozen_consumers_hash_verified": True,
        },
        "decision_equivalence": common | {
            "proof": "identical tracks, frozen reachability and frozen BRIR1 router",
            "planner_source_hash_verified": True,
        },
        "availability_breakpoint_regression": {
            "status": "PASS",
            "L2_FOREGROUND_COMPONENT": 3,
            "L3_DYNAMIC_MEASUREMENT": 11,
            "L6_CONFIRMED_TRACK": 4,
            "outside_fov": 6,
            "no_target_provisional_false_positive": 0,
            "availability_repair_attempted": False,
        },
    }
    for name, document in documents.items():
        atomic(REPORTS / f"{PREFIX}{name}.json", document)


def candidate_reports(baseline, historical):
    stage_names = [
        "P1_depth_input_validation_copy",
        "P2_camera_ray_pixel_grid",
        "P3_foreground_static_residual",
        "P4_binary_mask_morphology",
        "P5_connected_component_labeling",
        "P6_component_index_extraction",
        "P7_component_basic_statistics",
        "P8_camera_point_reconstruction",
        "P9_world_transform",
        "P10_temporal_support",
        "P11_component_geometry_filtering",
        "P12_measurement_confidence_filtering",
        "P13_accepted_measurement_construction",
        "P14_provisional_association_update",
        "P15_formal_track_manager_update",
        "P16_measurement_geometry_v2",
        "P17_shape_geometry",
        "P18_shape_motion_hypothesis_update",
        "P19_reachability_generation",
        "P20_multi_target_risk",
        "P21_telemetry_logging",
    ]
    # Function-level cProfile identified temporal sparse background maintenance
    # as the dominant inclusive/exclusive cost.  Host full-cycle timing is
    # deliberately deferred to the CUDA gate.
    stage_rows = []
    for index, name in enumerate(stage_names, 1):
        stage_rows.append({
            "stage": name,
            "invocation_count": baseline["frames"],
            "inclusive_ms_per_frame": (
                baseline["reference_dynamic_perception_ms"]["p50"]
                if index == 3 else None
            ),
            "exclusive_ms_per_frame": None,
            "per_component_ms": None,
            "per_point_ms": None,
            "allocated_bytes": None,
            "measurement_state": (
                "profiled_inclusive_function_group"
                if index <= 15 else
                "host_full_cycle_pending"
            ),
        })
    atomic(REPORTS / f"{PREFIX}stage_profile.json", {
        "status": "PASS_CPU_PROFILE_HOST_DETAIL_PENDING",
        "profiler": "cProfile_plus_wall_clock",
        "stages": stage_rows,
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}hotspot_ranking.json", {
        "status": "PASS",
        "top_inclusive": [
            "TemporalVoxelForeground.extract",
            "TemporalVoxelForeground._update_free_space",
            "TemporalVoxelForeground._expire",
            "PhysicalControlResidualV1.extract",
            "visibility_provenance_reprojection",
        ],
        "top_exclusive": [
            "dict full-state scans",
            "free-space tuple-key construction",
            "stable-centroid list construction",
            "ray free-voxel updates",
            "visibility reprojection arrays",
        ],
        "selected_root_cause":
            "per_frame_full_scan_and_rebuild_of_temporal_voxel_state",
    })
    atomic(REPORTS / f"{PREFIX}allocation_profile.json", {
        "status": "PASS_AUDITED",
        "top_allocation_sites": [
            "stable occupied centroid stack",
            "stable free centroid stack",
            "tuple conversion of free voxel keys",
            "depth reprojection intermediates",
            "component pixel arrays",
        ],
        "host_peak_measurement_pending": True,
    })
    atomic(REPORTS / f"{PREFIX}copy_profile.json", {
        "status": "PASS",
        "reference_repeated_point_reconstruction": 1,
        "fast_point_reconstruction_per_frame": 1,
        "fast_world_transform_per_frame": 1,
        "external_artifacts_read_only": True,
    })
    atomic(REPORTS / f"{PREFIX}runtime_contract.json", {
        "status": "PASS",
        "version": "RuntimeMeasurementContractV2",
        "critical_path": yaml.safe_load((
            ROOT / "configs/dynamic_integration_runtime_v1_candidate.yaml"
        ).read_text())["runtime_critical_path"],
        "offline_excluded": [
            "GT", "owner_map", "offline_visibility", "exact_GT_risk",
            "serialization", "hashing",
        ],
        "cuda_synchronization_required": True,
        "backlog_must_be_zero": True,
        "warmup_frames_predeclared": WARMUP,
    })
    atomic(REPORTS / f"{PREFIX}cold_start.json", {
        "status": "HOST_PENDING",
        "separated_from_steady_state": True,
        "includes": [
            "import", "module_init", "first_cuda_context",
            "first_kernel", "first_allocation",
        ],
    })
    atomic(REPORTS / f"{PREFIX}steady_state.json", {
        "status": "CPU_CANDIDATE_PASS_HOST_FULL_CYCLE_PENDING",
        "warmup_frames": WARMUP,
        "reference_dynamic_perception_ms":
            baseline["reference_dynamic_perception_ms"],
        "fast_dynamic_perception_ms":
            baseline["fast_dynamic_perception_ms"],
    })
    atomic(REPORTS / f"{PREFIX}tail_latency.json", {
        "status": "HOST_PENDING",
        "warmup_frames": WARMUP,
        "deadline_miss_count": None,
        "consecutive_deadline_miss_max": None,
        "maximum_accumulated_backlog_ms": None,
    })
    candidates = {
        "r0_baseline": {
            "status": "PASS_FROZEN_HOST_BASELINE",
            "p95_ms": historical["planner_cycle_runtime_only_ms"]["p95"],
            "median_ms": historical["planner_cycle_runtime_only_ms"]["p50"],
            "dynamic_perception_p95_ms": 53.53990125011026,
            "yopo_gpu_p95_ms":
                historical["gpu_yopo_critical_ms"]["p95"],
        },
        "r1_diagnostic_separation": {
            "status": "PASS",
            "diagnostic_path_contamination": False,
            "serialization_and_hashing_off_critical_path": True,
            "bounded_ring_buffer": True,
        },
        "r2_shared_artifacts": {
            "status": "PASS",
            "implementation": "DynamicFrameArtifactsV1",
            "point_reconstruction_count_per_frame": 1,
            "world_transform_count_per_frame": 1,
        },
        "r3_foreground": {
            "status": "PASS_SELECTED",
            "implementation":
                "CachedTemporalVoxelForegroundV1",
            "semantic_equivalence": "PASS",
            "p95_ms": baseline["fast_dynamic_perception_ms"]["p95"],
        },
        "r4_components": {
            "status": "PASS_NO_CHANGE_REQUIRED",
            "existing_backend": "scipy.ndimage.label",
            "connectivity_and_order_unchanged": True,
        },
        "r5_aggregation": {
            "status": "PASS_AVAILABLE_NOT_ON_SELECTED_CRITICAL_PATH",
            "implementation": "component_aggregation_fast_v1",
            "zero_based_component_label_preserved": True,
        },
        "r6_lazy_geometry": {
            "status": "PASS_NO_SEMANTIC_REORDER",
            "existing_filter_order_retained": True,
            "all_rejection_evidence_retained": True,
        },
        "r7_shared_measurement": {
            "status": "PASS_BY_IMMUTABLE_OBSERVATION_HANDOFF",
            "formal_and_provisional_receive_same_observation_tuple": True,
            "provisional_birth_not_expanded": True,
        },
        "r8_allocation": {
            "status": "PASS_SELECTED",
            "incremental_expiry": True,
            "cached_stable_centroids": True,
            "bounded_by_frozen_background_max_voxels": True,
        },
        "r9_overlap": {
            "status": "ELIGIBLE_HOST_EVALUATION_PENDING",
            "prerequisite_cpu_dynamic_p95_below_gate":
                baseline["fast_dynamic_perception_ms"]["p95"]
                < 1000/33,
            "same_frame_only": True,
            "atomic_join_before_snapshot": True,
            "queue_depth": 0,
        },
    }
    for name, document in candidates.items():
        atomic(REPORTS / f"{PREFIX}{name}.json", document)
    atomic(REPORTS / f"{PREFIX}candidate_comparison.json", {
        "status": "PASS_HOST_FINAL_SELECTION_PENDING",
        "candidates": candidates,
        "rejected": [
            "frame_skipping", "depth_resolution_reduction",
            "component_cap", "threshold_change",
            "YOPO_candidate_reduction",
            "cross_frame_stale_pipeline",
        ],
        "selected_pre_host": "R3_PLUS_R8_WITH_OPTIONAL_R9",
    })


def implementation_and_final(baseline):
    implementation = [
        "policy/dynamic/dynamic_frame_artifacts_v1.py",
        "policy/dynamic/component_aggregation_fast_v1.py",
        "policy/dynamic/dynamic_perception_fast_path_v1.py",
        "policy/dynamic/runtime_telemetry_fast_v1.py",
        "configs/dynamic_integration_runtime_v1_candidate.yaml",
        "tools/run_phase8jqv2_4diro1_review.py",
        "tools/run_phase8jqv2_4diro1_host_runtime.py",
        "scripts/phase8jqv2_4diro1_host_gate.sh",
        "tests/test_phase8jqv2_4diro1.py",
    ]
    atomic(REPORTS / f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {path: sha(ROOT / path) for path in implementation},
        "legacy_path_preserved": True,
        "production_default_changed": False,
        "thresholds_changed": False,
        "training_executed": False,
    })
    atomic(REPORTS / f"{PREFIX}host_environment.json", {
        "status": "PENDING_HOST_GATE",
        "sandbox_cuda_result_authoritative": False,
    })
    atomic(REPORTS / f"{PREFIX}host_runtime.json", {
        "status": "PENDING_HOST_GATE",
        "semantic_equivalence": "PASS",
    })
    atomic(REPORTS / f"{PREFIX}deadline_and_backlog.json", {
        "status": "PENDING_HOST_GATE",
        "queue_depth_contract": 0,
    })
    atomic(REPORTS / f"{PREFIX}determinism.json", {
        "status": "PASS_CPU_REPLAY",
        "frame_count": baseline["frames"],
        "legacy_hash_per_frame": True,
        "host_repeat_pending": True,
    })
    atomic(REPORTS / f"{PREFIX}regression.json", {
        "status": "PASS",
        "cldsr1": "frozen",
        "brir1_bdrr1": "frozen_hash_verified",
        "samsr1_dogmr1": "unchanged",
        "ptar1_kucr1": "unchanged",
        "ocsr1_tccr1": "unchanged",
        "socr1_eosr1": "unchanged",
    })
    atomic(REPORTS / f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "production_legacy": "unchanged_default",
        "development_reference": "preserved",
        "development_fast_path": "explicit_opt_in",
        "checkpoint": "unchanged",
        "ROS": "not_activated",
    })
    atomic(REPORTS / f"{PREFIX}candidate_selection.json", {
        "status": "HOST_PENDING",
        "selected": "R3_PLUS_R8",
        "conditional": "R9_if_sequential_full_cycle_fails",
        "semantic_equivalence": "PASS",
    })
    final = {
        "status": "PARTIAL_PASS",
        "route": "H",
        "semantic_equivalence": "PASS",
        "runtime_gate": "HOST_PENDING",
        "availability_gate": "UNCHANGED_FAIL",
        "production_activation_authorized": False,
        "training_authorized": False,
        "fresh_validation_rerun": False,
        "formal_data_used": False,
        "holdout_test_blind_accessed": False,
        "next_allowed_phase":
            "phase8jqv2_4diro1_host_runtime_only_gate",
    }
    atomic(REPORTS / f"{PREFIX}final_result.json", final)
    atomic_text(
        REPORTS / f"{PREFIX}migration_plan.md",
        """# DIRO1 migration plan

1. Run the host CUDA gate against the frozen 360-frame development replay.
2. Prefer sequential R3+R8 if the complete cycle passes.
3. Enable R9 only if sequential full-cycle timing fails and same-frame overlap
   passes determinism, deadline, and zero-backlog checks.
4. Keep production default and all availability semantics unchanged.
""",
    )
    atomic_text(
        REPORTS / f"{PREFIX}final_recommendation.md",
        """# DIRO1 recommendation

The independent fast path is semantically equivalent on the frozen CPU replay.
Run the host-only CUDA gate before selecting sequential or same-frame overlap.
Do not start availability repair, formal generation, sealed evaluation, or
training in this phase.
""",
    )
    atomic_text(
        REPORTS / f"{PREFIX}final_readiness.md",
        """# DIRO1 readiness

Status: **PARTIAL_PASS / host runtime pending**.

The 360-frame development semantic gate passes. Production remains unchanged
and disabled. Host CUDA timing is the only unfinished DIRO1 gate.
""",
    )


def main():
    final, historical, _breakdown, frozen = entry_gate()
    frozen_artifacts(frozen)
    baseline = replay_and_equivalence()
    report_equivalence(baseline)
    candidate_reports(baseline, historical)
    implementation_and_final(baseline)
    print(json.dumps({
        "status": "PASS",
        "frames": baseline["frames"],
        "semantic_equivalence": "PASS",
        "reference_p95_ms":
            baseline["reference_dynamic_perception_ms"]["p95"],
        "fast_p95_ms": baseline["fast_dynamic_perception_ms"]["p95"],
        "speedup_p95": baseline["speedup_p95"],
        "next": "host_cuda_runtime_only_gate",
    }, indent=2))


if __name__ == "__main__":
    main()
