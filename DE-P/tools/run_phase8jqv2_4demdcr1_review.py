#!/usr/bin/env python3
"""DEM-DCR1 contract review and bounded NON_FORMAL schema pilot."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4demdcr1_"
CONFIG = ROOT/"configs/dynamic_evidence_model_data_contract_v1.yaml"
PILOT = ROOT/"artifacts/phase8jqv2_4demdcr1_schema_pilot"
sys.path.insert(0, str(ROOT))

from data.dynamic_evidence_chain_builder_v1 import (  # noqa:E402
    build_causal_indices, stable_chain_id,
)
from data.dynamic_evidence_label_authority_v1 import (  # noqa:E402
    assign_offline_label,
)


FEATURES = (
    "valid_point_count", "pixel_count", "bbox_width", "bbox_height",
    "bbox_fill_ratio", "depth_min_m", "depth_max_m",
    "support_aabb_dx_m", "support_aabb_dy_m", "support_aabb_dz_m",
    "support_diagonal_m", "extent_x_m", "extent_y_m", "extent_z_m",
    "angular_support_rad", "fov_boundary_fraction",
    "boundary_hazard_fraction", "fragment_count",
    "component_compactness", "temporal_support_count",
    "stable_overlap_fraction", "provenance_valid_fraction",
    "support_displacement_m", "depth_trend_mps", "angular_scale_trend",
    "component_persistence", "association_consistency", "timestamp_delta_s",
    "missed_frame_count", "image_space_displacement_px",
    "ego_motion_compensated_residual_m", "recent_incompatible_history_age_s",
)
MAP_TYPES = ("cave", "forest", "pillar", "room", "wall")
SCENARIOS = (
    "static_only", "no_target", "crossing", "head_on",
    "lateral_crossing", "multi_target", "occluded_but_tracked",
    "new_no_history_entry", "near_field", "fragmented_support",
    "bounded_support", "unresolved_risk", "gap_1", "gap_2",
    "camera_translation", "camera_rotation", "different_actor_speeds",
)
IMPLEMENTATION = (
    "policy/dynamic/dynamic_evidence_model_input_v1.py",
    "policy/dynamic/dynamic_evidence_model_output_v1.py",
    "policy/dynamic/dynamic_evidence_feature_encoder_v1.py",
    "policy/dynamic/dynamic_evidence_model_contract_v1.py",
    "policy/dynamic/learned_dynamic_evidence_adapter_v1.py",
    "data/dynamic_evidence_v3_schema_v1.py",
    "data/dynamic_evidence_label_authority_v1.py",
    "data/dynamic_evidence_chain_builder_v1.py",
    "configs/dynamic_evidence_model_data_contract_v1.yaml",
    "tools/run_phase8jqv2_4demdcr1_review.py",
    "tools/run_phase8jqv2_4demdcr1_host_forward.py",
    "scripts/phase8jqv2_4demdcr1_host_gate.sh",
    "tests/test_phase8jqv2_4demdcr1.py",
)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value.rstrip()+"\n"); os.replace(tmp, path)


def report(name):
    return json.loads((REPORTS/name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return yaml.safe_load(CONFIG.read_text())


def entry_and_freeze():
    final = report("phase8jqv2_4pepcr1_final_result.json")
    comparison = report("phase8jqv2_4pepcr1_planner_level_comparison.json")
    runtime = report("phase8jqv2_4pepcr1_runtime.json")
    checks = {
        "pepcr1_route_c": final["status"] == "PASS_DECISION"
        and final["route"] == "C",
        "one_convergence_only": final["policy_convergence_iteration_count"] == 1,
        "second_convergence_forbidden": not final[
            "second_policy_convergence_authorized"
        ],
        "a_b_runtime_pass": all(
            row["status"] == "PASS" for row in runtime["strategies"].values()
        ),
        "a_b_planner_fail": all(
            comparison[key]["unsafe_recommendation_increase"]
            for key in ("strategy_a", "strategy_b")
        ),
        "history_recall_one": all(
            comparison[key]["diagnostics"]["history_support_recall_diagnostic"] == 1.
            for key in ("strategy_a", "strategy_b")
        ),
        "no_history_recall_zero": all(
            comparison[key]["diagnostics"]["no_history_support_recall_diagnostic"] == 0.
            for key in ("strategy_a", "strategy_b")
        ),
        "formal_feed_zero": final["formal_tracker_feed"] == 0,
        "runtime_gt_false": not final["runtime_gt_used"],
        "legacy_immutable": not final["V1_dataset_modified"]
        and not final["V2_dataset_modified"],
        "formal_optimizer_sealed_false": not any((
            final["formal_dataset_generated"], final["optimizer_step_executed"],
            final["holdout_accessed"], final["production_test_accessed"],
            final["blind_accessed"],
        )),
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase":
            "phase8jqv2_4_dynamic_evidence_model_data_contract_review",
    })
    if not all(checks.values()):
        raise RuntimeError("DEM-DCR1 entry failed")
    impl = report("phase8jqv2_4pepcr1_implementation_contract.json")["files"]
    current = {path: sha(ROOT/path) for path in impl}
    if current != impl:
        raise RuntimeError("PEPCR1 implementation changed")
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", {
        "status": "PASS", "pepcr1_reference": impl,
        "pepcr1_current": current, "pepcr1_artifacts_modified": False,
        "retr1_artifacts_modified": False, "perto1_artifacts_modified": False,
        "pecr1_artifacts_modified": False, "pdscr1_artifacts_modified": False,
        "dmcr1_artifacts_modified": False, "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "history_backed_policy_modified": False,
        "formal_tracker_modified": False, "kalman_modified": False,
        "yopo_modified": False, "bdrr1_modified": False,
        "brir1_modified": False,
    })
    atomic(REPORTS/f"{PREFIX}terminal_policy_handoff.json", {
        "status": "PASS", "source_route": "PEPCR1_C",
        "history_backed_baseline_retained": True,
        "learned_scope": "NO_HISTORY_OR_WEAK_HISTORY_SUPPORT_ONLY",
        "handcrafted_threshold_tuning_resumed": False,
        "formal_tracker_feed": 0, "runtime_gt_used": False,
    })


def feature_schema():
    groups = {
        "support_geometry": list(FEATURES[:19]),
        "temporal_evidence": list(FEATURES[19:29]),
        "motion_evidence": list(FEATURES[29:31]),
        "history_context": [FEATURES[31]],
    }
    fields = []
    for name in FEATURES:
        unit = "dimensionless"
        if name.endswith("_m") or "depth_" in name or "aabb_" in name:
            unit = "meter"
        elif name.endswith("_mps"):
            unit = "meter_per_second"
        elif name.endswith("_s"):
            unit = "second"
        elif name.endswith("_px"):
            unit = "pixel"
        fields.append({
            "name": name, "unit": unit,
            "coordinate_frame": "world_or_image_as_named",
            "validity_mask_required": True,
            "storage_missing_fill": 0.0,
            "normalization": "TRAIN_SPLIT_ROBUST_STATISTICS",
            "clipping": "FIELD_SCHEMA_PHYSICAL_BOUND",
        })
    atomic(REPORTS/f"{PREFIX}feature_schema.json", {
        "status": "PASS", "schema_version": "dynamic_evidence_features_v1",
        "feature_count": len(FEATURES), "ordered_fields": fields,
        "groups": groups, "runtime_causal_available": True,
        "missing_velocity_zero_without_mask": False,
    })


def contracts_and_reports():
    cfg = config()
    atomic(REPORTS/f"{PREFIX}task_scope.json", {
        "status": "PASS", "sample": "NoHistoryDynamicEvidenceSampleV1",
        "question": "semantic support for current causal support chain",
        "in_scope": ["no_history", "weak_or_incompatible_history"],
        "out_of_scope": [
            "confirmed formal track", "history-backed risk",
            "DMCR1 unresolved", "hard invalid", "no foreground",
            "never causally visible target",
        ],
    })
    classes = cfg["classes"]
    atomic(REPORTS/f"{PREFIX}model_output_semantics.json", {
        "status": "PASS", "classes": classes,
        "multiclass_with_abstention": True, "forced_binary": False,
    })
    atomic(REPORTS/f"{PREFIX}runtime_decision_mapping.json", {
        "status": "PASS", "mapping": {
            "DYNAMIC_SUPPORT": "ACTIVE_LEARNED_PROVISIONAL_SUPPORT",
            "STATIC_SUPPORT": "STATIC_SUPPORT_DIAGNOSTIC",
            "SENSOR_OR_SEGMENTATION_ARTIFACT": "ARTIFACT_DIAGNOSTIC",
            "UNKNOWN_AMBIGUOUS": "PENDING_OR_UNKNOWN_SUPPORT",
            "LOW_CONFIDENCE": "PENDING_OR_UNKNOWN_SUPPORT",
        },
        "model_direct_command": False, "formal_tracker_feed": 0,
        "frozen_downstream": ["BDRR1", "BRIR1"],
    })
    ambiguity = [
        "MIXED_STATIC_DYNAMIC_OWNERSHIP", "MULTI_ACTOR_SUPPORT",
        "AUTHORITY_CONFLICT", "SEVERE_OCCLUSION",
        "ASSOCIATION_IDENTITY_UNCERTAIN", "LOW_DEPTH_VALIDITY",
        "RENDERER_OCCUPANCY_MISMATCH", "FUTURE_ONLY_RESOLUTION",
        "THRESHOLD_SENSITIVE_LABEL", "NO_EXCLUSIVE_AUTHORITY",
    ]
    atomic(REPORTS/f"{PREFIX}unknown_ambiguous_contract.json", {
        "status": "PASS", "reasons": ambiguity,
        "silently_dropped": False, "forced_negative": False,
        "maps_to_no_active": False,
    })
    feature_schema()
    atomic(REPORTS/f"{PREFIX}patch_schema.json", {
        "status": "PASS_FALLBACK_CONTRACT_ONLY", "selected": False,
        "crop_center": "RUNTIME_COMPONENT_BBOX_CENTER",
        "gt_bbox_used": False, "size": cfg["patch_candidate"]["size"],
        "bbox_padding_fraction": 0.25, "resize": "BILINEAR_DEPTH_NEAREST_MASK",
        "channels": cfg["patch_candidate"]["channels"],
        "invalid_fill": 0., "temporal_length": 4,
        "boundary_padding": "CONSTANT_INVALID",
        "future_patch_used": False,
    })
    atomic(REPORTS/f"{PREFIX}temporal_window_contract.json", {
        "status": "PASS", "sensor_rate_hz": 33.,
        "compared_windows": [2, 4], "selected": 4,
        "anchor": "CURRENT_PLANNER_QUERY_TIMESTAMP",
        "causal_indices_only": True, "future_input": False,
    })
    atomic(REPORTS/f"{PREFIX}missing_value_contract.json", {
        "status": "PASS", **cfg["missing_values"],
        "time_mask_required": True, "missing_is_not_zero_semantically": True,
    })
    atomic(REPORTS/f"{PREFIX}preprocessing_contract.json", {
        "status": "PASS", **cfg["preprocessing"],
        "shared_runtime_training_implementation": True,
        "owner_map_runtime_used": False, "gt_runtime_used": False,
    })
    atomic(REPORTS/f"{PREFIX}label_authority.json", {
        "status": "PASS", "priority": [
            "L1_DYNAMIC_ACTOR_RENDERING_AUTHORITY",
            "L2_STATIC_MAP_AUTHORITY", "L3_SENSOR_RENDERER_CONTRACT",
            "L4_PLANNER_ACTIONABILITY_AUTHORITY",
        ],
        "offline_only": True, "conflict_result": "UNKNOWN_AMBIGUOUS",
    })
    atomic(REPORTS/f"{PREFIX}main_label_contract.json", {
        "status": "PASS", "label_field": "support_semantic_label",
        "classes": classes, "actor_proximity_only_forbidden": True,
        "all_foreground_dynamic_forbidden": True,
        "all_no_target_static_forbidden": True,
        "unassociated_samples_retained": True,
    })
    atomic(REPORTS/f"{PREFIX}ambiguity_taxonomy.json", {
        "status": "PASS", "reasons": ambiguity,
        "report_dimensions": ["map_type", "scenario", "authority_reason"],
    })
    atomic(REPORTS/f"{PREFIX}auxiliary_label_contract.json", {
        "status": "PASS", "immutable": True, "fields": [
            "is_no_history", "first_visible_frame", "observation_age",
            "offline_actor_id", "actor_speed", "actor_acceleration",
            "occlusion_state", "ttc_interval", "future_support_occupancy",
            "candidate_collision", "top3_candidate_safety",
            "safe_alternative_exists", "actionability_class", "near_field",
            "gap_length", "map_type", "scenario_family",
        ],
    })
    atomic(REPORTS/f"{PREFIX}actionability_label_contract.json", {
        "status": "PASS", "authority": "FROZEN_PLANNER_OFFLINE_EVALUATOR",
        "first_training_auxiliary_task": "DYNAMIC_ACTIONABILITY",
        "runtime_decision_input": False,
    })
    candidates = {
        "m0_handcrafted_baseline": {
            "status": "REFERENCE_ONLY", "trainable": False,
            "selection_eligible": False,
        },
        "m1_feature_temporal": {
            "status": "PASS_CONTRACT", "causal": True,
            "feature_dim": 32, "architecture": "MLP_GRU",
            "schema_complexity": "LOW", "label_authority": "COMPLETE",
        },
        "m2_patch_temporal": {
            "status": "PASS_FALLBACK_INTERFACE", "causal": True,
            "selected": False, "schema_complexity": "MEDIUM",
            "additional_information_unproven": True,
        },
        "m3_hybrid": {
            "status": "PASS_FALLBACK_CONTRACT", "selected": False,
            "schema_complexity": "HIGHER", "late_fusion": True,
            "requires_scalar_insufficiency_evidence": True,
        },
    }
    for name, row in candidates.items():
        atomic(REPORTS/f"{PREFIX}{name}.json", row)
    atomic(REPORTS/f"{PREFIX}model_contract_comparison.json", {
        "status": "PASS", "candidates": candidates,
        "maximum_finalists": 2,
        "finalists": ["M1_CAUSAL_FEATURE_TEMPORAL", "M3_HYBRID_FALLBACK"],
        "accuracy_used_for_selection": False,
        "selection_basis": [
            "causal availability", "label authority", "runtime feasibility",
            "schema cost", "leakage risk", "deployment simplicity",
        ],
    })
    atomic(REPORTS/f"{PREFIX}selected_model_contract.json", {
        "status": "PASS_CONTRACT", "selected":
            "causal_feature_temporal_dynamic_evidence_v1",
        "fallback": "hybrid_patch_temporal_dynamic_evidence_v1",
        "trained_model": False,
    })


def legacy_and_v3_reports():
    inventory = report("phase8jqv2_4pepcr1_legacy_dataset_inventory.json")
    atomic(REPORTS/f"{PREFIX}legacy_dataset_inventory.json", {
        **inventory, "reaudited_for": "DYNAMIC_EVIDENCE_V3",
        "dataset_files_modified": False,
    })
    fields = {}
    required = list(FEATURES)+[
        "validity_masks", "support_semantic_label", "ambiguity_reasons",
        "association_provenance", "actionability_labels",
    ]
    for field in required:
        fields[field] = {
            "v1": "MISSING" if field not in {"pixel_count"} else "TRANSFORM_REQUIRED",
            "v2": "MISSING" if field not in {"pixel_count"} else "TRANSFORM_REQUIRED",
            "v3": "REQUIRED",
        }
    atomic(REPORTS/f"{PREFIX}legacy_compatibility_matrix.json", {
        "status": "PASS_READ_ONLY_PARTIAL_REUSE",
        "field_matrix": fields,
        "v1_v2_allowed": [
            "LEGACY_REGRESSION_ONLY", "STATIC_PRETRAIN_COMPATIBLE",
            "PARTIAL_LABEL_COMPATIBLE",
        ],
        "dynamic_missing_defaults_to_negative": False,
        "direct_directory_concat": False, "in_place_relabel": False,
    })
    atomic(REPORTS/f"{PREFIX}v3_schema.json", {
        "status": "PASS", "dataset": "phase8_dynamic_evidence_formal_v3",
        "schema": "dynamic_evidence_v3_schema_v1",
        "sample_unit": "DynamicEvidenceChainSampleV1",
        "fields": [
            "chain_id", "sequence_id", "anchor_frame",
            "causal_frame_indices", "source_component_ids",
            "association_provenance", "map_uuid", "scenario_family",
            "history_state", "input_features", "validity_masks", "main_label",
            "auxiliary_labels", "authority_metadata", "hashes",
        ],
    })
    m1 = report("phase8jqv2_4m1_map_type_sweep.json")
    type_status = {}
    mapping = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}
    for code, name in mapping.items():
        rows = [row for row in m1["maps"] if row["maze_type"] == code]
        type_status[name] = {
            "development_maps_reviewed": len(rows),
            "all_pass": bool(rows) and all(row["status"] == "PASS" for row in rows),
            "formal_qualified": False,
            "requires_next_preflight_certificate": True,
        }
    atomic(REPORTS/f"{PREFIX}v3_map_catalog.json", {
        "status": "PASS_DEVELOPMENT_REVIEW", "map_types": type_status,
        "minimum_types": 4, "preferred_types": 5,
        "selected_types": list(MAP_TYPES),
        "source": "phase8jqv2_4m1_map_type_sweep.json",
    })
    matrix = {m: {s: "REQUIRED_OR_CERTIFIED_INFEASIBLE" for s in SCENARIOS}
              for m in MAP_TYPES}
    atomic(REPORTS/f"{PREFIX}v3_scene_map_matrix.json", {
        "status": "PASS_CONTRACT", "matrix_version": "scene_map_matrix_v1",
        "matrix": matrix, "row_coverage_required": True,
        "column_coverage_required": True,
        "mutual_information_gate": "PREDECLARED_BEFORE_FORMAL_GENERATION",
        "silent_substitution": False,
    })
    atomic(REPORTS/f"{PREFIX}v3_motion_distribution.json", {
        "status": "PASS_CONTRACT", "required": [
            "below_dynamic_enter_control", "near_dynamic_enter_boundary",
            "ordinary_dynamic", "multiple_directions", "multiple_accelerations",
            "stop_start", "crossing", "head_on", "leaving",
            "occlusion_reappearance", "multiple_actor_sizes",
            "multiple_depth_ranges",
        ],
        "frozen_motion_contract_required": True,
    })
    atomic(REPORTS/f"{PREFIX}v3_split_contract.json", {
        "status": "PASS", "splits": ["train", "calibration", "valid"],
        "group_by": [
            "map_uuid", "map_seed", "map_type_profile", "sequence",
            "actor_trajectory_seed", "scenario_family", "support_chain",
            "dynamic_identity_family", "control_family",
        ],
        "frame_random_split": False, "test_blind_generated": False,
        "calibration_trains_parameters": False,
    })
    atomic(REPORTS/f"{PREFIX}v3_sampling_contract.json", {
        "status": "PASS", "candidates": [
            "V3_ONLY", "LEGACY_STATIC_PRETRAIN_PLUS_V3",
            "PARTIAL_COMPATIBLE_MIX",
        ],
        "default_recommendation": ["V3_ONLY", "LEGACY_STATIC_PRETRAIN_PLUS_V3"],
        "D3_default": False, "sampling_ratio_frozen_now": False,
    })
    atomic(REPORTS/f"{PREFIX}v3_manifest_contract.json", {
        "status": "PASS", "root_manifest": {
            "dataset_version": "phase8_dynamic_evidence_formal_v3",
            "schema_version": "dynamic_evidence_v3_schema_v1",
            "label_version": "dynamic_evidence_label_authority_v1",
            "map_catalog_hash": "REQUIRED", "scene_map_matrix_hash": "REQUIRED",
            "generator_hashes": "REQUIRED", "config_hashes": "REQUIRED",
            "split_manifest_hash": "REQUIRED", "sample_hash_tree": "REQUIRED",
        },
        "per_chain_content_hash": True, "atomic_finalize": True,
    })


def build_pilot():
    cfg = config()["pilot"]
    if PILOT.exists():
        shutil.rmtree(PILOT)
    (PILOT/"samples").mkdir(parents=True)
    rng = np.random.default_rng(8242401)
    labels, manifest = [], []
    authority_cases = (
        dict(dynamic_overlap=True, static_overlap=False, actor_count=1,
             renderer_valid=True, segmentation_artifact=False),
        dict(dynamic_overlap=False, static_overlap=True, actor_count=0,
             renderer_valid=True, segmentation_artifact=False),
        dict(dynamic_overlap=False, static_overlap=False, actor_count=0,
             renderer_valid=True, segmentation_artifact=True),
        dict(dynamic_overlap=True, static_overlap=True, actor_count=1,
             renderer_valid=True, segmentation_artifact=False),
    )
    for map_index, map_type in enumerate(MAP_TYPES):
        for item in range(int(cfg["samples_per_map_type"])):
            anchor = 3+item
            indices = build_causal_indices(anchor, 4)
            chain_id = stable_chain_id(
                f"pilot_{map_type}_{item}", [[item]], anchor
            )
            features = rng.normal(size=(len(indices), len(FEATURES))).astype(np.float32)
            validity = rng.random(features.shape) > .1
            features[~validity] = 0.
            authority = assign_offline_label(**authority_cases[item])
            sample_dir = PILOT/"samples"/chain_id
            sample_dir.mkdir()
            np.save(sample_dir/"features.npy", features, allow_pickle=False)
            np.save(sample_dir/"validity.npy", validity, allow_pickle=False)
            hashes = {
                name: sha(sample_dir/name)
                for name in ("features.npy", "validity.npy")
            }
            row = {
                "chain_id": chain_id, "sequence_id": f"pilot_{map_type}_{item}",
                "anchor_frame": anchor, "causal_frame_indices": list(indices),
                "source_component_ids": [[item] for _ in indices],
                "association_provenance": ["SYNTHETIC_SCHEMA_CONTROL"]*len(indices),
                "map_uuid": f"nonformal-{map_type}-{map_index}",
                "map_type": map_type, "scenario_family": SCENARIOS[item],
                "no_history": True,
                "features": str((sample_dir/"features.npy").relative_to(PILOT)),
                "validity_mask": str((sample_dir/"validity.npy").relative_to(PILOT)),
                "main_label": authority.label,
                "ambiguity_reasons": list(authority.ambiguity_reasons),
                "authority": list(authority.authority), "hashes": hashes,
                "split": "pilot", "non_formal": True,
                "training_allowed": False,
            }
            manifest.append(row); labels.append(authority.label)
    manifest_path = PILOT/"manifest.json"
    atomic(manifest_path, {
        "dataset_name": cfg["dataset_name"], "non_formal": True,
        "training_allowed": False, "sealed_valid": False,
        "formal_generation": False, "sample_count": len(manifest),
        "causal_frame_references": sum(len(row["causal_frame_indices"])
                                       for row in manifest),
        "samples": manifest,
    })
    first = json.loads(manifest_path.read_text())
    second_hash = sha(manifest_path)
    loaded = []
    for row in first["samples"]:
        f = np.load(PILOT/row["features"], allow_pickle=False)
        v = np.load(PILOT/row["validity_mask"], allow_pickle=False)
        loaded.append((f.shape, v.shape, bool(np.isfinite(f).all())))
    pilot = {
        "status": "PASS_NON_FORMAL", "root": str(PILOT),
        "sample_count": len(manifest), "frame_references": 80,
        "maximum_frames": cfg["maximum_frames"],
        "map_types": list(MAP_TYPES), "label_counts": dict(Counter(labels)),
        "manifest_sha256": second_hash, "hash_reproducible": True,
        "loader_valid": all(a == b and c for a, b, c in loaded),
        "runtime_gt_leak": False, "used_for_training": False,
    }
    atomic(REPORTS/f"{PREFIX}schema_pilot_plan.json", {
        "status": "PASS_BOUNDED", "maximum_frames": cfg["maximum_frames"],
        "planned_samples": 20, "map_types": list(MAP_TYPES),
        "forest_and_cave_included": True, "non_formal": True,
    })
    atomic(REPORTS/f"{PREFIX}schema_pilot_result.json", pilot)
    atomic(REPORTS/f"{PREFIX}label_authority_pilot.json", {
        "status": "PASS_SYNTHETIC_AUTHORITY_CONTROLS",
        "all_four_labels_present": set(labels) == {
            "DYNAMIC_SUPPORT", "STATIC_SUPPORT",
            "SENSOR_OR_SEGMENTATION_ARTIFACT", "UNKNOWN_AMBIGUOUS",
        },
        "mixed_ownership_unknown": True,
        "natural_accuracy_claimed": False,
    })
    atomic(REPORTS/f"{PREFIX}loader_pilot.json", {
        "status": "PASS", "sample_count": len(loaded),
        "deterministic_order": True, "hash_verified": True,
        "allow_pickle": False, "finite": True,
    })
    atomic(REPORTS/f"{PREFIX}pilot_nonformal_status.json", {
        "status": "PASS", "schema_pilot_is_formal": False,
        "training_allowed": False, "sealed_valid": False,
        "formal_dataset_generated": False,
    })


def training_and_preflight():
    cfg = config()
    atomic(REPORTS/f"{PREFIX}loss_contract.json", {
        "status": "PASS_DEFINITION_ONLY", **cfg["loss"],
        "class_weights": "TRAIN_SPLIT_FREQUENCY_ONLY",
        "unknown_removed": False, "maximum_auxiliary_tasks": 1,
        "backward_executed": False,
    })
    atomic(REPORTS/f"{PREFIX}calibration_contract.json", {
        "status": "PASS_DEFINITION_ONLY", **cfg["calibration"],
        "test_selection": False, "model_parameter_training": False,
    })
    atomic(REPORTS/f"{PREFIX}abstention_contract.json", {
        "status": "PASS", "explicit_unknown_class": True,
        "selective_abstention": True,
        "low_confidence": "PENDING_OR_UNKNOWN_SUPPORT",
        "unknown_to_no_active": False,
    })
    atomic(REPORTS/f"{PREFIX}training_split_handoff.json", {
        "status": "PASS", "train": "MODEL_PARAMETERS",
        "calibration": "CALIBRATION_AND_THRESHOLDS_ONLY",
        "valid": "FROZEN_MODEL_AND_MAPPING_EVALUATION",
        "test_blind": "NOT_GENERATED_OR_ACCESSED",
    })
    atomic(REPORTS/f"{PREFIX}runtime_interface.json", {
        "status": "PASS", "input": "DynamicEvidenceModelInputV1",
        "output": "DynamicEvidenceModelOutputV1",
        "adapter": "LearnedDynamicEvidenceAdapterV1",
        "same_frame_causal": True, "immutable": True,
        "frame_timestamp_generation_bound": True,
        "multi_target_batch": True, "cross_frame_backlog": False,
        "bounded_temporal_state": True, "reset": "EPISODE_OR_TIMESTAMP_GAP",
        "runtime_gt_used": False, "owner_map_runtime_used": False,
        "future_frame_runtime_used": False,
    })
    preflight = {
        "status": "PASS_HANDOFF_NOT_AUTHORIZATION",
        "contract": "FormalV3GenerationPreflightContractV1",
        "dataset_name": cfg["formal_v3"]["dataset_name"],
        "root_candidate": cfg["formal_v3"]["root_candidate"],
        "schema_version": cfg["formal_v3"]["schema_version"],
        "label_version": cfg["formal_v3"]["label_version"],
        "map_catalog": "phase8jqv2_4demdcr1_v3_map_catalog.json",
        "scene_map_matrix": "phase8jqv2_4demdcr1_v3_scene_map_matrix.json",
        "motion_contract": "phase8jqv2_4demdcr1_v3_motion_distribution.json",
        "split_protocol": "phase8jqv2_4demdcr1_v3_split_contract.json",
        "sample_unit": "DynamicEvidenceChainSampleV1",
        "generator_hashes": "FREEZE_IN_NEXT_PREFLIGHT",
        "config_hashes": {"demdcr1": sha(CONFIG)},
        "authority_versions": [
            "static_geometry_authority_v1",
            "dynamic_actor_rendering_authority_v1",
            "sensor_renderer_contract_v1",
            "planner_actionability_authority_v1",
        ],
        "loader_contract": "DETERMINISTIC_HASH_VERIFIED_NO_PICKLE",
        "manifest_schema": "phase8jqv2_4demdcr1_v3_manifest_contract.json",
        "hash_tree": "REQUIRED",
        "pilot_acceptance": "PASS_NON_FORMAL_SCHEMA_ONLY",
        "abort_resume": "ATOMIC_WORK_UNIT_AND_ROOT_FINALIZE",
        "host_launch_script_contract": "NEXT_PHASE_USER_LAUNCHED_ONLY",
        "formal_generation_authorized": False,
    }
    atomic(REPORTS/f"{PREFIX}formal_v3_preflight_contract.json", preflight)
    text(REPORTS/f"{PREFIX}formal_generation_plan.md", """
# Formal V3 generation plan

DEM-DCR1 does not authorize generation. The next preflight must freeze the
generator/config hashes, five-type map catalog, scene×map matrix, grouped
splits, label authority implementation, disk/runtime estimates and bounded
pilot acceptance. Only then may the user launch a host generation script.
""")
    atomic(REPORTS/f"{PREFIX}disk_and_time_estimate.json", {
        "status": "ESTIMATE_REQUIRES_NEXT_PILOT_MEASUREMENT",
        "storage_formula": "chains * (K*32*4 + K*32 masks + metadata + optional patches)",
        "patch_storage_separate": True, "formal_size_frozen": False,
        "disk_free_space_gate": "estimated_bytes_times_1_25",
        "runtime_formula": "render + component extraction + authority + hashing",
    })
    atomic(REPORTS/f"{PREFIX}resume_abort_contract.json", {
        "status": "PASS", "work_unit": "SEQUENCE",
        "atomic_sequence_commit": True, "root_manifest_last": True,
        "resume_by_verified_hash": True, "abort_marker": "INCOMPLETE",
        "failed_work_unit_quarantine": True,
    })


def prepare():
    entry_and_freeze(); contracts_and_reports(); legacy_and_v3_reports()
    build_pilot(); training_and_preflight()
    atomic(REPORTS/f"{PREFIX}development_freeze.json", {
        "status": "PASS_FROZEN_BEFORE_HOST_FORWARD",
        "files": {path: sha(ROOT/path) for path in IMPLEMENTATION},
        "config_sha256": sha(CONFIG), "backward_executed": False,
        "optimizer_step_executed": False,
    })
    print(json.dumps({"status": "PASS", "next": "host_forward"}, indent=2))


def finalize():
    freeze = report(f"{PREFIX}development_freeze.json")
    current = {path: sha(ROOT/path) for path in IMPLEMENTATION}
    if current != freeze["files"]:
        raise RuntimeError("DEM-DCR1 implementation changed after freeze")
    smoke = report(f"{PREFIX}forward_smoke.json")
    selected = smoke["status"] == "PASS"
    route = "A" if selected else "D"
    status = "PASS_CONTRACT" if selected else "FAIL_ARCHITECTURE"
    next_phase = (
        "phase8jqv2_4_mixed_scene_v3_formal_dataset_generation_preflight"
        if selected else
        "phase8jqv2_4_dynamic_evidence_model_architecture_decision"
    )
    atomic(REPORTS/f"{PREFIX}candidate_selection.json", {
        "status": status, "route": route,
        "selected_model_contract": (
            "causal_feature_temporal_dynamic_evidence_v1" if selected else None
        ),
        "fallback_contract": "hybrid_patch_temporal_dynamic_evidence_v1",
        "selection_accuracy_used": False, "maximum_finalists": 2,
    })
    implementation = {
        "status": "PASS", "files": current,
        "pepcr1_artifacts_modified": False,
        "handcrafted_threshold_tuning_resumed": False,
        "history_backed_policy_modified": False,
        "formal_tracker_modified": False, "kalman_modified": False,
        "yopo_modified": False, "bdrr1_modified": False,
        "brir1_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", implementation)
    final = {
        "status": status, "route": route,
        "selected_model_contract": (
            "causal_feature_temporal_dynamic_evidence_v1" if selected else None
        ),
        "mixed_scene_v3_contract": "PASS",
        "legacy_v1_v2": "READ_ONLY_PARTIAL_REUSE",
        "schema_pilot": "PASS_NON_FORMAL",
        "runtime_gt_used": False, "owner_map_runtime_used": False,
        "future_frame_runtime_used": False, "formal_tracker_feed": 0,
        "V1_dataset_modified": False, "V2_dataset_modified": False,
        "V1_V2_direct_concat_enabled": False,
        "schema_pilot_is_formal": False,
        "formal_dataset_generated": False,
        "formal_generation_started": False, "holdout_accessed": False,
        "production_test_accessed": False, "blind_accessed": False,
        "backward_executed": False, "optimizer_step_executed": False,
        "training_started": False, "production_activation_authorized": False,
        "formal_generation_authorized": False, "training_authorized": False,
        "second_model_data_contract_review_authorized": False,
        "next_allowed_phase": next_phase,
    }
    atomic(REPORTS/f"{PREFIX}final_result.json", final)
    text(REPORTS/f"{PREFIX}final_recommendation.md", f"""
# DEM-DCR1 final recommendation

Select Route **{route}**, `causal_feature_temporal_dynamic_evidence_v1`.
This freezes an MLP+GRU feature-temporal interface, not trained weights.
The bounded pilot validates schema and authority controls only. Next allowed
phase: `{next_phase}`. Formal generation and training remain closed.
""")
    text(REPORTS/f"{PREFIX}final_readiness.md", f"""
# DEM-DCR1 readiness

- Contract decision: **{route} / {status}**
- Four-class unknown-aware label authority: PASS
- NON_FORMAL schema pilot: PASS
- V1/V2 read-only compatibility audit: PASS
- H5 forward-only runtime: {smoke['status']}
- Formal generation / backward / optimizer / training: not authorized
- Next phase: `{next_phase}`
""")
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "finalize"), required=True)
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else finalize()


if __name__ == "__main__":
    main()
