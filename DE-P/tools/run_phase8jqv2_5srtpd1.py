#!/usr/bin/env python3
"""SRTPD1 one-time project rebaseline and route decision.

This tool is deliberately read-only with respect to every dataset, model and
DEOACR1 artifact.  It writes reports only.  The pilot reads a bounded pair of
train sequences and never opens internal-test, test, blind or production-test
samples.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from config.config import cfg


REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_5srtpd1_"
V1 = ROOT / "data/phase8_authoritative_v1"
V2 = ROOT / "data/phase8_authoritative_v2"
V3 = ROOT / "data/phase8_dynamic_evidence_formal_v3"
RAW = V3 / "raw_authoritative"
DEO = ROOT / "data/phase8_dynamic_evidence_objective_canary_v2"
SEMANTIC_NAMES = {
    1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(name, value):
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{PREFIX}{name}.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return value


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deoacr1_freeze():
    final = load_json(REPORTS / "phase8jqv2_4deoacr1_final_result.json")
    terminal = load_json(
        REPORTS / "phase8jqv2_4deoacr1_terminal_decision.json"
    )
    stability = load_json(
        REPORTS / "phase8jqv2_4deoacr1_seed_stability.json"
    )
    required = {
        "status": "STOP_LEARNED_DYNAMIC_EVIDENCE_PATH",
        "route": "D", "next_allowed_phase": None,
        "full_derived_dataset_built": False,
        "long_training_started": False, "GT_input_used": False,
    }
    mismatches = {
        key: {"expected": value, "actual": final.get(key)}
        for key, value in required.items() if final.get(key) != value
    }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "project_rebaseline_explicitly_authorized": True,
        "deoacr1_terminal": terminal,
        "frozen_result": required,
        "seed_stability": stability,
        "mismatches": mismatches,
        "reopened_for_repair": False,
    }


def archive_learned_branch():
    checkpoints = []
    for path in sorted((DEO / "checkpoints").glob("*.pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "candidate": payload.get("candidate"),
            "objective": payload.get("objective"),
            "seed": payload.get("seed"),
            "original_non_formal": payload.get("non_formal") is True,
            "original_formal": payload.get("formal", False),
            "archive_labels": [
                "NON_FORMAL", "NOT_FOR_PRODUCTION",
                "NOT_FOR_LONG_TRAINING", "NOT_SELECTED",
            ],
            "modified": False,
        })
    status = (
        len(checkpoints) == 15
        and all(row["original_non_formal"] for row in checkpoints)
        and not any(row["original_formal"] for row in checkpoints)
    )
    return {
        "status": "PASS" if status else "FAIL",
        "archive_version": "LearnedDynamicEvidenceArchiveV1",
        "terminal_history": {
            "DETIAR1": "Route E / STOP_LEARNED_DYNAMIC_EVIDENCE_PATH",
            "DEOACR1": "Route D / STOP_LEARNED_DYNAMIC_EVIDENCE_PATH",
        },
        "frozen_architectures": ["S0", "S1", "S2"],
        "frozen_objectives": ["O0", "O1"],
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "retained_diagnostics": [
            "planner_counterfactual_actionability_v2",
            "counterfactual evaluator", "proposal-label purity audit",
            "temporal canary", "model input leakage guard",
        ],
        "prohibited": [
            "restart S1/S2 training", "create S3/S4/S5",
            "promote checkpoint", "build learned evidence derived dataset",
            "change O0/O1 or actionability thresholds",
        ],
    }


def map_coverage():
    counts = Counter()
    split_counts = {"train": Counter(), "valid": Counter()}
    for split in ("train", "valid"):
        root = RAW / "geometry_authority" / split
        for path in sorted(root.glob("*/mixed_scene_provenance.json")):
            row = load_json(path)
            name = SEMANTIC_NAMES.get(int(row["maze_type"]), "unsupported")
            counts[name] += 1
            split_counts[split][name] += 1
    return {
        "all": dict(sorted(counts.items())),
        "by_split": {
            split: dict(sorted(value.items()))
            for split, value in split_counts.items()
        },
        "required": sorted(SEMANTIC_NAMES.values()),
        "complete": all(counts[name] > 0 for name in SEMANTIC_NAMES.values()),
    }


def _first_sequence(suite, split="train", require_actor=False):
    paths = sorted((RAW / suite / split).glob("formal_*"))
    if not paths:
        raise RuntimeError(f"no {suite}/{split} sequence")
    if require_actor:
        for path in paths:
            diagnostics = load_json(path / "render_diagnostics.json")
            if int(diagnostics.get("actor_depth_pixels_total", 0)) > 0:
                return path
        raise RuntimeError(f"no actor-visible {suite}/{split} sequence")
    return paths[0]


def static_view_pilot():
    """Bounded train-only pilot; never emits a derived sample."""
    selected = {
        "static": _first_sequence("static"),
        "dynamic": _first_sequence("dynamic", require_actor=True),
    }
    rows = {}
    input_hashes = {}
    for suite, directory in selected.items():
        static = np.load(directory / "static_depth.npy", mmap_mode="r")
        composed = np.load(directory / "depth.npy", mmap_mode="r")
        owner = np.load(directory / "actor_owner.npy", mmap_mode="r")
        frames = [
            json.loads(line) for line in
            (directory / "frames.jsonl").read_text().splitlines()
        ]
        difference = np.abs(
            np.asarray(composed, dtype=np.float32)
            - np.asarray(static, dtype=np.float32)
        ) > 1e-5
        actor = np.asarray(owner) >= 0
        rows[suite] = {
            "sequence_id": directory.name,
            "split": "train",
            "static_depth_shape": list(static.shape),
            "composed_depth_shape": list(composed.shape),
            "owner_shape": list(owner.shape),
            "static_depth_dtype": str(static.dtype),
            "static_finite": bool(np.isfinite(static).all()),
            "changed_pixels": int(difference.sum()),
            "actor_pixels": int(actor.sum()),
            "changed_outside_actor_pixels": int(
                np.count_nonzero(difference & ~actor)
            ),
            "actor_pixels_without_depth_change": int(
                np.count_nonzero(actor & ~difference)
            ),
            "frame_count": len(frames),
            "map_uuid": frames[0]["map_uuid"],
            "input_state_shapes": {
                "velocity_body": list(
                    np.asarray(frames[0]["velocity_body"]).shape
                ),
                "acceleration_body": list(
                    np.asarray(frames[0]["acceleration_body"]).shape
                ),
                "goal_body": list(
                    np.asarray(frames[0]["goal_body"]).shape
                ),
                "observation": [9],
            },
        }
        input_hashes[suite] = {
            name: sha256(directory / name) for name in (
                "static_depth.npy", "depth.npy", "actor_owner.npy",
                "frames.jsonl",
            )
        }
    dynamic_path = selected["dynamic"]
    static = np.asarray(
        np.load(dynamic_path / "static_depth.npy", mmap_mode="r")[0],
        dtype=np.float32,
    )
    resized = cv2.resize(
        np.minimum(static, 20.) / 20.,
        (int(cfg["image_width"]), int(cfg["image_height"])),
        interpolation=cv2.INTER_NEAREST,
    )[None]
    frame = json.loads(
        (dynamic_path / "frames.jsonl").read_text().splitlines()[0]
    )
    observation = np.asarray(
        frame["velocity_body"] + frame["acceleration_body"]
        + frame["goal_body"], dtype=np.float32,
    )
    map_root = (
        RAW / "geometry_authority" / "train" / frame["map_uuid"]
    )
    authority = ExactAuthorityBVH(map_root)
    start = np.asarray(frame["position_world"], dtype=np.float64)
    goal = np.asarray(frame["goal_world"], dtype=np.float64)
    direction = goal - start
    direction /= max(np.linalg.norm(direction), 1e-9)
    points = np.asarray([
        start + direction * distance for distance in np.linspace(0., 2., 21)
    ])
    gaps = np.asarray([
        authority.query_one(point, .3)["minimum_gap_m"] for point in points
    ])
    cost = np.exp(np.clip(
        (float(cfg["d0"]) - gaps) / float(cfg["r"]), -60., 60.,
    ))
    coverage = map_coverage()
    static_row, dynamic_row = rows["static"], rows["dynamic"]
    checks = {
        "static_depth_exists": True,
        "static_suite_has_no_actor_contribution": (
            static_row["changed_pixels"] == 0
            and static_row["actor_pixels"] == 0
        ),
        "dynamic_composed_static_boundary": (
            dynamic_row["changed_pixels"] > 0
            and dynamic_row["changed_outside_actor_pixels"] == 0
        ),
        "input_shape": list(resized.shape) == [1, 96, 160],
        "observation_shape": list(observation.shape) == [9],
        "static_cost_finite": bool(np.isfinite(cost).all()),
        "loader_deterministic": (
            input_hashes["dynamic"]["static_depth.npy"]
            == sha256(dynamic_path / "static_depth.npy")
        ),
        "all_map_types": coverage["complete"],
        "actor_owner_used_as_input": False,
        "actor_future_used_as_input": False,
        "composed_depth_used_for_static_supervision": False,
        "derived_samples_written": 0,
    }
    positive_checks = (
        "static_depth_exists", "static_suite_has_no_actor_contribution",
        "dynamic_composed_static_boundary", "input_shape",
        "observation_shape", "static_cost_finite",
        "loader_deterministic", "all_map_types",
    )
    forbidden_checks = (
        "actor_owner_used_as_input", "actor_future_used_as_input",
        "composed_depth_used_for_static_supervision",
    )
    pilot_pass = (
        all(checks[key] for key in positive_checks)
        and all(not checks[key] for key in forbidden_checks)
        and checks["derived_samples_written"] == 0
    )
    return {
        "status": "PASS" if pilot_pass else "FAIL",
        "pilot_class": "NON_FORMAL_READ_ONLY",
        "selected_train_sequences": {
            key: value.name for key, value in selected.items()
        },
        "rows": rows, "checks": checks,
        "network_ready_input": {
            "depth_shape": list(resized.shape),
            "depth_dtype": str(resized.dtype),
            "depth_min": float(resized.min()),
            "depth_max": float(resized.max()),
            "observation_shape": list(observation.shape),
            "observation_finite": bool(np.isfinite(observation).all()),
        },
        "static_authority_cost_probe": {
            "map_uuid": frame["map_uuid"], "points": len(points),
            "minimum_gap_m": float(gaps.min()),
            "maximum_gap_m": float(gaps.max()),
            "cost_min": float(cost.min()), "cost_max": float(cost.max()),
            "finite": bool(np.isfinite(cost).all()),
            "formal_label_generated": False,
        },
        "map_coverage": coverage,
        "source_hashes": input_hashes,
    }


def compatibility():
    fields = [
        "depth_semantics", "depth_normalization", "camera_intrinsics",
        "camera_extrinsics", "image_resolution", "pose_frame",
        "velocity_acceleration_frame", "goal_frame",
        "static_map_authority", "map_uuid", "candidate_contract",
        "YOPO_label_cost_contract", "split_identity",
        "generator_config_hash",
    ]
    sources = {
        "V1": {
            "classification": "PRETRAIN_ONLY",
            "reason": "pre-V3 state/motion semantics; transform and independent validation required",
            "read_only": True,
        },
        "V2": {
            "classification": "PRETRAIN_ONLY",
            "reason": "inherits V1 authority and predates mixed-map V3 closure",
            "read_only": True,
        },
        "V3_STATIC_VIEW": {
            "classification": "COMPATIBLE_WITH_VERSIONED_TRANSFORM",
            "reason": "meter static_depth 96x160 -> finite validation, clipping and /20 normalization",
            "read_only": True,
        },
        "LEGACY_SHARED_YOPO_DATASET": {
            "classification": "PRETRAIN_ONLY",
            "reason": "legacy PNG/pose/map-index schema lacks V3 UUID and authority hashes",
            "read_only": True,
        },
    }
    per_field = {}
    for source in sources:
        per_field[source] = {
            field: (
                "VERSIONED_TRANSFORM_REQUIRED"
                if field in {
                    "depth_normalization", "image_resolution",
                    "map_uuid", "split_identity", "generator_config_hash",
                } else "COMPATIBLE_FOR_STATIC_PRETRAIN"
            ) for field in fields
        }
    per_field["V3_STATIC_VIEW"].update({
        "depth_semantics": "STATIC_AUTHORITY_ONLY",
        "static_map_authority": "FULLY_COMPATIBLE",
        "map_uuid": "FULLY_COMPATIBLE",
        "split_identity": "FULLY_COMPATIBLE",
        "generator_config_hash": "FULLY_COMPATIBLE",
        "candidate_contract": "FULLY_COMPATIBLE",
        "YOPO_label_cost_contract": "FULLY_COMPATIBLE",
    })
    return {
        "status": "PASS",
        "direct_directory_concatenation": False,
        "selected_data_route": "D1_V3_STATIC_ONLY",
        "selection_reason": (
            "V3 alone covers all five map types with closed UUID, split and "
            "static authority; V1/V2 remain optional pretrain-only assets."
        ),
        "sources": sources, "fields": per_field,
    }


def reports():
    freeze = atomic_json("deoacr1_terminal_freeze", deoacr1_freeze())
    archive = atomic_json("learned_branch_archive", archive_learned_branch())
    entry = atomic_json("entry_gate", {
        "status": "PASS" if (
            freeze["status"] == "PASS" and archive["status"] == "PASS"
        ) else "FAIL",
        "phase": "phase8jqv2_5_system_rebaseline_and_training_path_decision",
        "explicit_project_rebaseline_authorized": True,
        "deoacr1_next_allowed_phase_was_null": True,
        "learned_path_reopened": False,
        "formal_training_authorized": False,
        "source_manifest_identity": {
            "V1": sha256(V1 / "manifests/dataset_manifest.json"),
            "V2": sha256(V2 / "manifests/dataset_manifest.json"),
            "V3": sha256(V3 / "manifests/dataset_manifest.json"),
            "V3_expected":
                "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6",
            "V3_matches_frozen": (
                sha256(V3 / "manifests/dataset_manifest.json")
                == "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6"
            ),
        },
    })
    inventory = atomic_json("reusable_asset_inventory", {
        "status": "PASS",
        "assets": [
            {"name": "DE-P static YOPO network/candidates/cost",
             "class": "PRODUCTION_CANDIDATE"},
            {"name": "V1/V2 authoritative static views",
             "class": "PRETRAIN_ONLY"},
            {"name": "Formal V3 raw static_depth and static authority",
             "class": "PRODUCTION_CANDIDATE"},
            {"name": "Formal V3 actor future/owner/counterfactual labels",
             "class": "DIAGNOSTIC_ONLY"},
            {"name": "strict dynamic measurement",
             "class": "DEVELOPMENT_ONLY"},
            {"name": "MAR1/DMCR1 measurement availability",
             "class": "DEVELOPMENT_ONLY"},
            {"name": "TrackManager/Kalman",
             "class": "DEVELOPMENT_ONLY"},
            {"name": "BDRR1 bounded reachability",
             "class": "PRODUCTION_CANDIDATE"},
            {"name": "BRIR1 risk adapter/router",
             "class": "DEVELOPMENT_ONLY"},
            {"name": "RETR1 H5 environment contract",
             "class": "PRODUCTION_CANDIDATE"},
            {"name": "DETIAR1/DEOACR1 learned branch",
             "class": "ARCHIVED"},
            {"name": "weak provisional evidence births",
             "class": "UNSUPPORTED"},
        ],
    })
    risks = atomic_json("unresolved_risk_inventory", {
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "risks": [
            "no-history first observation",
            "L2 foreground availability",
            "outside-FOV actor",
            "never-observed full occlusion",
            "insufficient short evidence",
            "track drop after prior observation",
            "dynamic actor domain shift",
            "sphere/cylinder geometry limitation",
            "BRIR1 historical unsafe executions without active track",
            "PDSCR1 weak-support false availability",
        ],
        "hidden_by_odd_exclusion": False,
        "threshold_repair_authorized": False,
    })
    runtime = atomic_json("runtime_component_matrix", {
        "status": "PASS_FOR_ROUTE_SELECTION_NOT_PRODUCTION",
        "components": {
            "static_yopo": {
                "input": "static depth + 9D UAV/goal state",
                "output": "15 candidates + scores",
                "runtime_gt": False, "interface": "AVAILABLE",
            },
            "strict_measurement_or_history": {
                "input": "causal depth/pose/timestamps",
                "output": "legal measured/history-backed dynamic state",
                "runtime_gt": False, "interface": "DEVELOPMENT_ONLY",
            },
            "TrackManager_Kalman": {
                "input": "associated observations",
                "output": "causal tracks/covariance/lifecycle",
                "runtime_gt": False, "interface": "AVAILABLE",
            },
            "BDRR1": {
                "input": "tracked shape-motion hypothesis",
                "output": "bounded reachable occupancy or unresolved",
                "runtime_gt": False, "interface": "PASS_DEVELOPMENT",
            },
            "BRIR1_router": {
                "input": "candidate snapshot + bounded occupancy",
                "output": "keep/switch/safe-abort",
                "runtime_gt": False,
                "interface": "COMPOSABLE_DEVELOPMENT_ONLY",
                "historical_full_integration": "FAIL",
            },
        },
        "historical_evidence": {
            "BDRR1": "PASS",
            "BRIR1": "FAIL_34_UNSAFE_WITHOUT_ACTIVE_TRACK",
            "DMCR1": "PASS_DEVELOPMENT_ONLY",
            "PDSCR1": "FAIL_FALSE_AVAILABILITY",
            "RETR1_H5_ENVIRONMENT": {
                "status": "PASS", "p95_ms": 20.78930524694442,
                "p99_ms": 24.616421059399727,
                "deadline_miss_rate": .0013333333333333333,
            },
        },
    })
    odd = atomic_json("dynamic_odd_contract", {
        "status": "PASS",
        "version": "DeterministicDynamicODDV1",
        "claim": (
            "Within the frozen limited ODD, deterministic intervention is "
            "provided only for causally observed targets that form legal "
            "strict or history-backed states."
        ),
        "required": [
            "target enters valid camera FOV",
            "valid depth and causal foreground available",
            "strict measurement or legal history-backed support",
            "continuous timestamps",
            "depth/pose synchronization within frozen contract",
            "sufficient causal observations before reaction deadline",
            "speed and acceleration inside frozen bounds",
            "sphere/vertical-cylinder frozen geometry applicability",
            "BDRR1 forms bounded occupancy",
            "BRIR1 consumes risk and completes a decision",
        ],
        "odd_shrink_count": 0,
        "frozen_before_final_validation": True,
    })
    supported = atomic_json("supported_dynamic_cases", {
        "status": "PASS_NONEMPTY_ODD",
        "cases": [
            "history-backed visible target",
            "visible crossing with legal state",
            "visible head-on with legal state",
            "separable visible multi-target with legal states",
            "occluded-but-tracked within frozen lifecycle",
            "continuous timestamp and bounded motion",
        ],
        "claim_all_new_targets_solved": False,
    })
    out_odd = atomic_json("out_of_odd_cases", {
        "status": "PASS_EXPLICIT",
        "cases": [
            "first-frame near collision", "never-visible actor",
            "full occlusion without history", "extreme speed/acceleration",
            "transparent/reflective/invalid depth", "timestamp break",
            "extreme non-rigid object", "strongly overlapping targets",
            "unbounded or missing geometry support",
        ],
        "required_behavior": [
            "identify unsupported/unresolved",
            "do not emit a safe conclusion",
            "veto, hover, brake or safe-abort under frozen router semantics",
        ],
        "normal_navigation_success_required": False,
    })
    fail_closed = atomic_json("fail_closed_contract", {
        "status": "PASS_SEMANTIC_AUDIT_ONLY",
        "mappings": {
            "CONFIRMED_OR_HISTORY_BACKED_RISK": "BDRR1_BRIR1_INTERVENTION",
            "UNRESOLVED_MEASUREMENT_RISK": "CONSERVATIVE_VETO_OR_SAFE_ABORT",
            "PENDING_OR_UNKNOWN_SUPPORT": "UNRESOLVED_NOT_SAFE",
            "NO_VALID_EVALUATION": "INVALID_EVALUATION_SAFE_ABORT",
            "NO_SAFE_CANDIDATE": "SAFE_ABORT",
        },
        "forbidden": {
            "unknown_to_safe": False,
            "invalid_to_no_active": False,
            "cache_read_refreshes_lifecycle": False,
            "weak_provisional_support_enabled": False,
        },
        "note": (
            "Weak provisional evidence is excluded from the selected guard "
            "because PDSCR1 false availability failed; no threshold changed."
        ),
    })
    static_task = atomic_json("static_training_task", {
        "status": "PASS_CONTRACT",
        "task": "Mixed-Scene Static YOPO Base Planner",
        "inputs": ["static_depth", "UAV state", "goal/body state"],
        "authority": [
            "static occupancy", "static trajectory cost",
            "static map geometry",
        ],
        "forbidden_inputs": [
            "composed dynamic depth", "actor owner", "actor future",
            "dynamic label/actionability", "scenario/map type",
        ],
        "learned_dynamic_model_trained": False,
    })
    compat = atomic_json("legacy_dataset_compatibility", compatibility())
    v3_contract = atomic_json("v3_static_view_contract", {
        "status": "PASS",
        "source": "Formal V3 raw (read-only)",
        "depth_field": "static_depth.npy",
        "forbidden_depth_field": "depth.npy",
        "transform_version": "v3_static_yopo_input_transform_v1",
        "transform": [
            "load float32 meters", "validate finite and [0,20] contract",
            "validate native 96x160", "clip to 20 m", "divide by 20",
        ],
        "state_fields": [
            "position_world", "quaternion_world_from_body",
            "velocity_body", "acceleration_body", "goal_body", "map_uuid",
        ],
        "owner_or_future_as_input": False,
    })
    coverage = map_coverage()
    plan = atomic_json("mixed_map_training_plan", {
        "status": "PASS",
        "selected": "D1_V3_STATIC_ONLY",
        "derived_candidate":
            "phase8_mixed_scene_static_yopo_derived_v1",
        "map_coverage": coverage,
        "split_unit": "map_uuid_then_sequence",
        "directory_concatenation": False,
        "formal_dataset_built": False,
    })
    pilot = atomic_json("static_dataset_pilot", static_view_pilot())
    gate = atomic_json("planner_level_gates", {
        "status": "PLANNED_NOT_EXECUTED",
        "threshold_source": "existing frozen Gates only",
        "metrics": [
            "unsafe recommendation", "executed unsafe proxy",
            "top-3 unsafe miss", "no-target intervention",
            "static false intervention", "false emergency",
            "safe false-veto", "NO_SAFE frequency",
            "reaction-time margin", "track stale miss",
            "unresolved downgrade", "static goal/path availability",
            "runtime deadline",
        ],
        "strata": ["supported ODD", "out-of-ODD"],
        "threshold_tuning_loop": False,
    })
    evaluation = atomic_json("system_evaluation_plan", {
        "status": "PASS_PLAN",
        "system": [
            "Static YOPO", "deterministic dynamic measurement",
            "association/tracking", "BDRR1", "BRIR1",
            "final fail-closed recommendation",
        ],
        "corpora": {
            "static_navigation": sorted(SEMANTIC_NAMES.values()),
            "supported_dynamic_odd": supported["cases"],
            "out_of_odd": out_odd["cases"],
        },
        "production_pass_claimed": False,
        "hard_failure_policy": (
            "one ODD shrink before final validation, otherwise Route B or C; "
            "no threshold repair"
        ),
    })
    h5 = atomic_json("h5_integration_plan", {
        "status": "PASS_FEASIBILITY_PLAN_NOT_EXECUTED",
        "cycle": [
            "static perception/planner",
            "deterministic dynamic perception", "tracking",
            "BDRR1", "BRIR1", "final recommendation",
        ],
        "environment": {
            "CPU_affinity": "0-7",
            "OMP_MKL_OPENBLAS_NUMEXPR_threads": 1,
            "PyTorch_intra_inter_threads": 1,
            "episode_boundary_GC": True,
            "pretouch_bytes": 8 * 1024 * 1024,
            "low_overhead_logging": True,
        },
        "gates": {
            "p95_ms_max": 30.303, "p99_ms_max": 30.303,
            "deadline_miss_rate_max": .01,
            "consecutive_miss_max": 1, "queue_depth": 0,
            "backlog": False, "skipped_frames": 0,
            "runtime_GT": False,
        },
        "historical_environment_feasibility": {
            "p95_ms": 20.78930524694442,
            "p99_ms": 24.616421059399727,
            "deadline_miss_rate": .0013333333333333333,
        },
        "full_selected_system_H5_pass_claimed": False,
    })
    route_a = atomic_json("route_a_contract", {
        "status": "ELIGIBLE_SELECTED",
        "system":
            "mixed_scene_static_yopo_plus_deterministic_dynamic_guard_v1",
        "dynamic_claim": "LIMITED_ODD_DETERMINISTIC",
        "production_qualified": False,
        "conditions": {
            "static_training_contract": True,
            "deterministic_io_defined": True,
            "supported_odd_formalized": True,
            "out_of_odd_fail_closed": True,
            "runtime_gt_absent": True,
            "interfaces_composable": True,
            "h5_budget_feasible": True,
            "static_supervision_label_consistent": True,
        },
    })
    route_b = atomic_json("route_b_contract", {
        "status": "ELIGIBLE_FALLBACK_NOT_SELECTED",
        "system": "mixed_scene_static_yopo_v1",
        "dynamic_claim": "NONE",
        "trigger": (
            "Route A supported ODD cannot pass the one-time combination Gate"
        ),
    })
    route_c = atomic_json("route_c_contract", {
        "status": "NOT_SELECTED",
        "system": None,
        "trigger": "static training contract or project integrity fails",
        "future_research_only": [
            "occupancy flow", "scene flow", "object detection/tracking",
            "future occupancy prediction", "richer dynamic simulation",
            "real-world data collection",
        ],
    })
    comparison = atomic_json("route_comparison", {
        "status": "PASS_UNIQUE_SELECTION",
        "selected": "A",
        "A": {
            "eligible": all(route_a["conditions"].values()),
            "benefit": "limited-ODD deterministic intervention",
            "cost": "requires one-time static-data/training and full H5 qualification",
        },
        "B": {
            "eligible": True,
            "benefit": "simpler static delivery",
            "cost": "no dynamic avoidance claim",
        },
        "C": {
            "eligible": True,
            "benefit": "stops delivery risk",
            "cost": "abandons viable static and deterministic assets",
        },
        "reason": (
            "All eight Route A completeness predicates are supportable as "
            "a limited-ODD architecture decision. Historical BRIR1/PDSCR1 "
            "failures remain explicit qualification blockers, not hidden "
            "or relabeled as production PASS."
        ),
    })
    terminal = atomic_json("terminal_decision", {
        "status": "PASS_PROJECT_REBASELINE",
        "route": "A",
        "selected_system":
            "mixed_scene_static_yopo_plus_deterministic_dynamic_guard_v1",
        "learned_dynamic_evidence": "ARCHIVED_TERMINAL",
        "dynamic_claim": "LIMITED_ODD_DETERMINISTIC",
        "formal_training_authorized": False,
        "next_allowed_phase":
            "phase8jqv2_5_mixed_static_yopo_dataset_and_training_readiness",
    })
    final = atomic_json("final_result", {
        **terminal,
        "deoacr1_artifacts_modified": False,
        "detiar1_artifacts_modified": False,
        "learned_dynamic_evidence_restarted": False,
        "S1_training_restarted": False, "S2_training_restarted": False,
        "S3_S4_created": False,
        "actionability_threshold_modified": False,
        "learned_checkpoint_promoted": False,
        "source_v3_raw_modified": False,
        "V1_modified": False, "V2_modified": False,
        "composed_dynamic_depth_used_for_static_supervision": False,
        "actor_owner_used_as_training_input": False,
        "actor_future_used_as_training_input": False,
        "runtime_gt_used": False, "formal_tracker_feed": 0,
        "full_static_dataset_built": False,
        "long_training_script_created": False,
        "long_training_started": False,
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False, "production_test_accessed": False,
        "rebaseline_iteration_count": 1,
        "second_rebaseline_authorized": False,
        "static_pilot_status": pilot["status"],
        "static_pilot_initial_failure_preserved": (
            REPORTS / f"{PREFIX}static_dataset_pilot_initial_failure.json"
        ).exists(),
        "selected_data_route": compat["selected_data_route"],
        "current_production_qualified": False,
        "source_manifest_identity": entry["source_manifest_identity"],
    })
    REPORTS.joinpath(
        f"{PREFIX}learned_branch_terminal_status.md"
    ).write_text(
        "# Learned Dynamic Evidence Terminal Status\n\n"
        "DETIAR1 Route E and DEOACR1 Route D are frozen. All 15 checkpoints "
        "remain **NON_FORMAL / NOT_FOR_PRODUCTION / "
        "NOT_FOR_LONG_TRAINING / NOT_SELECTED**. The learned branch is "
        "archived and is not part of Route A.\n",
        encoding="utf-8",
    )
    REPORTS.joinpath(f"{PREFIX}final_readiness.md").write_text(
        "# SRTPD1 Final Readiness\n\n"
        "**Route A selected:** mixed-scene static YOPO plus a deterministic "
        "dynamic guard in `DeterministicDynamicODDV1`.\n\n"
        "This is a project-path decision, not a production safety PASS. "
        "Static training is not yet authorized. Historical BRIR1 unsafe "
        "executions and PDSCR1 weak-support false availability remain hard "
        "qualification blockers. The selected guard excludes weak "
        "provisional evidence and may claim dynamic intervention only after "
        "strict or legal history-backed state formation. The next phase must "
        "build the static-only V3 view, close its training contracts, and "
        "run the complete combined H5 Gate once.\n",
        encoding="utf-8",
    )
    REPORTS.joinpath(f"{PREFIX}final_recommendation.md").write_text(
        "# SRTPD1 Final Recommendation\n\n"
        "Proceed once to "
        "`phase8jqv2_5_mixed_static_yopo_dataset_and_training_readiness` "
        "using V3 `static_depth.npy` only. Keep learned evidence archived. "
        "Do not use composed depth with static-only labels. Freeze the "
        "limited dynamic ODD before final validation; if the deterministic "
        "guard fails inside that ODD, use the single allowed ODD reduction "
        "or fall back to Route B/C—do not start another threshold-review "
        "loop.\n",
        encoding="utf-8",
    )
    return {
        "entry": entry, "inventory": inventory, "risks": risks,
        "runtime": runtime, "odd": odd, "fail_closed": fail_closed,
        "static_task": static_task, "v3_contract": v3_contract,
        "plan": plan, "gate": gate, "evaluation": evaluation, "h5": h5,
        "comparison": comparison, "final": final,
    }


def main():
    initial = REPORTS / f"{PREFIX}static_dataset_pilot.json"
    if initial.exists():
        row = load_json(initial)
        archive = REPORTS / (
            f"{PREFIX}static_dataset_pilot_initial_failure.json"
        )
        if row.get("status") == "FAIL" and not archive.exists():
            shutil.copy2(initial, archive)
            for name in ("entry_gate", "final_result"):
                source = REPORTS / f"{PREFIX}{name}.json"
                target = REPORTS / f"{PREFIX}{name}_initial_failure.json"
                if source.exists() and not target.exists():
                    shutil.copy2(source, target)
    result = reports()
    print(json.dumps(result["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
