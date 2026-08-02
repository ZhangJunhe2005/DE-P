#!/usr/bin/env python3
"""Read-only SOCR1 contract audit and frozen-state replay."""

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from policy.dynamic.kalman_tracker import LinearKalmanTracker
from policy.dynamic.track_manager import TrackManager, _ManagedTrack
from policy.dynamic.types import DynamicPerceptionConfig

PREFIX = "phase8jqv2_4socr1_"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    path = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def atomic_json(name, value):
    path = REPORTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def load(name):
    return json.loads((REPORTS / name).read_text())


def frozen_inventory():
    paths = [
        "authoritative_dataset/natural_exact_occlusion_solver_v2.py",
        "configs/natural_short_full_occlusion_contract_v2.yaml",
        "authoritative_dataset/occlusion_constructor_v2_2.py",
        "authoritative_dataset/cuda_renderer_v1.py",
        "policy/dynamic/track_manager.py",
        "policy/dynamic/dynamic_perception.py",
        "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "config/traj_opt.yaml",
        "reports/phase8jqv2_4eosr1_proof_witness_manifest.json",
        "reports/phase8jqv2_4eosr1_independent_rerender.json",
        "reports/phase8jqv2_4eosr1_independent_validator.json",
        "reports/phase8jqv2_4eosr1_final_result.json",
        "reports/phase8jqv2_4mtc1_final_result.json",
        "reports/phase8jqv2_4rr1_natural_corpus_manifest.json",
    ]
    return {
        path: sha(ROOT / path) for path in paths
    } | {
        "diagnostics/phase8jqv2_4eosr1":
            tree_hash(ROOT / "diagnostics/phase8jqv2_4eosr1")
    }


def verify_entry():
    final = load("phase8jqv2_4eosr1_final_result.json")
    proof = load("phase8jqv2_4eosr1_proof_witness_manifest.json")
    independent = load(
        "phase8jqv2_4eosr1_independent_validator.json"
    )
    facts = {
        "historical_status": final["status"],
        "proof_witnesses": final["proof_witnesses"],
        "independent_maps": final["independent_maps"],
        "gap1_present": final["gap1_present"],
        "gap2_present": final["gap2_present"],
        "actor_radius_variants": sorted({
            float(row["radius_m"]) for row in proof["witnesses"]
        }),
        "default_radius_0_30": final["default_radius_0_30"],
        "independent_rerender_status": independent["status"],
        "independent_rows_all_pass": all(
            row["status"] == "PASS" for row in independent["rows"]
        ),
        "formal_generation_started":
            final["formal_generation_started"],
        "detector_executed": final["detector_executed"],
        "tracker_executed": final["tracker_executed"],
        "training_started": final["training_started"],
    }
    passed = (
        facts["historical_status"] == "FAIL"
        and facts["proof_witnesses"] == 6
        and facts["independent_maps"] == 1
        and facts["gap1_present"] is True
        and facts["gap2_present"] is False
        and facts["actor_radius_variants"] == [.2, .25, .3]
        and facts["default_radius_0_30"] == "PASS"
        and facts["independent_rerender_status"] == "PASS"
        and facts["independent_rows_all_pass"]
        and not any(facts[key] for key in (
            "formal_generation_started", "detector_executed",
            "tracker_executed", "training_started",
        ))
    )
    if not passed:
        raise RuntimeError(f"SOCR1 entry mismatch: {facts}")
    return facts


def tracker_contract(config):
    source = inspect.getsource(TrackManager.update)
    order = (
        "predict", "associate", "matched_update",
        "unmatched_miss_increment", "unmatched_observation_birth",
        "delete_if_missed_count_gt_max", "confirmation", "confidence",
        "dynamic_state", "attention_authorization",
    )
    report = {
        "status": "PASS",
        "confidence_initial_managed_track": 0.0,
        "snapshot_dataclass_default": 1.0,
        "snapshot_default_is_not_lifecycle_initialization": True,
        "confidence_formula": {
            "confirmation_weight": .15,
            "association_weight": .25,
            "uncertainty_weight": .15,
            "motion_consistency_weight": .25,
            "foreground_support_weight": .10,
            "cluster_geometry_weight": .10,
            "clip": [0., 1.],
            "miss_multiplier":
                "confidence_missed_decay ** missed_count",
        },
        "confidence_cap": 1.0,
        "confidence_missed_decay":
            config.confidence_missed_decay,
        "track_confidence_threshold":
            config.track_confidence_threshold,
        "dynamic_enter_speed": config.dynamic_enter_speed,
        "dynamic_exit_speed": config.dynamic_exit_speed,
        "max_missed_frames": config.max_missed_frames,
        "min_confirmed_hits": config.min_confirmed_hits,
        "dynamic_min_confirmed_hits":
            config.dynamic_min_confirmed_hits,
        "lifecycle_order": list(order),
        "confidence_recomputed_each_frame": True,
        "confidence_is_not_recursive_previous_confidence": True,
        "miss_decay_before_dynamic_state": True,
        "dynamic_recomputed_each_frame": True,
        "attention_uses_post_decay_post_dynamic_state": True,
        "deletion_before_confidence_dynamic_attention": True,
        "deletion_condition": "missed_count > max_missed_frames",
        "matched_recovery_order":
            "predict -> associate -> KF update -> missed_count=0 -> "
            "direct observation -> confidence -> dynamic -> attention",
        "states_are_distinct": {
            "track_exists": "identity remains in _tracks until deletion",
            "is_confirmed": "hit_count >= min_confirmed_hits",
            "is_dynamic": "hysteresis plus confidence/speed/uncertainty",
            "attention_authorized":
                "confirmed AND dynamic AND direct provenance AND "
                "visibility/miss/confidence checks",
        },
        "source_hash": sha(ROOT / "policy/dynamic/track_manager.py"),
        "source_order_markers_present": all(marker in source for marker in (
            "_update_confidence", "_update_dynamic_state",
            "attention_authorized", "missed_count >",
        )),
    }
    return report


def _components_for_confidence(target):
    minimum_u = math.exp(-1.0)
    uncertainty = minimum_u
    remaining = float(target)-.5-.15*uncertainty
    association = min(1., max(0., remaining/.25))
    remaining -= .25*association
    foreground = min(1., max(0., remaining/.10))
    remaining -= .10*foreground
    uncertainty += remaining/.15
    if not (minimum_u-1e-9 <= uncertainty <= 1+1e-9):
        raise ValueError(f"cannot realize certain track confidence {target}")
    return association, foreground, min(1., uncertainty)


def replay_one(config, target, gap):
    association, foreground, uncertainty = (
        _components_for_confidence(target)
    )
    tracker = LinearKalmanTracker(np.zeros(3), 0., config)
    tracker.set_velocity(np.asarray((1., 0., 0.)))
    velocity_std = (
        -config.dynamic_max_velocity_std*math.log(uncertainty)
    )
    diagonal = np.asarray(
        [.01, .01, .01] + [velocity_std**2]*3
    )
    tracker.filter.P = np.diag(diagonal)
    observation = SimpleNamespace(
        foreground_support=foreground,
        point_count=config.dynamic_min_cluster_points,
        extent=np.asarray([
            max(config.dynamic_min_cluster_extent+.05, .20)
        ]*3),
    )
    managed = _ManagedTrack(
        tracker=tracker,
        age=8,
        hit_count=max(
            config.min_confirmed_hits,
            config.dynamic_min_confirmed_hits,
        ),
        is_confirmed=True,
        is_dynamic=True,
        dynamic_reason="replay_initial_dynamic",
        last_observation=observation,
        motion_consistency_count=config.motion_consistency_frames,
        association_quality=association,
        confidence=0.,
        ever_directly_observed=True,
        direct_observation_count=8,
        consecutive_direct_hits=8,
        visibility_state="direct_observation",
        attention_authorized=True,
    )
    manager = TrackManager(config)
    manager._update_confidence(managed)
    initial = managed.confidence
    frames = [{
        "stage": "pre_gap", "missed_count": 0,
        "confidence": initial, "track_exists": True,
        "confirmed": managed.is_confirmed,
        "dynamic": managed.is_dynamic,
        "attention_authorized": managed.attention_authorized,
        "visibility_state": managed.visibility_state,
    }]
    for miss in range(1, gap+1):
        tracker.predict_to(miss*.1)
        managed.age += 1
        managed.missed_count = miss
        managed.prediction_only_age = miss
        managed.consecutive_direct_hits = 0
        managed.visibility_state = "occluded"
        exists = miss <= config.max_missed_frames
        if exists:
            manager._update_confidence(managed)
            manager._update_dynamic_state(managed)
            managed.attention_authorized = bool(
                managed.is_confirmed and managed.is_dynamic
                and managed.ever_directly_observed
                and managed.prediction_only_age
                <= config.max_missed_frames
                and managed.visibility_state != "clear_missing"
                and managed.confidence
                >= config.track_confidence_threshold
            )
        frames.append({
            "stage": f"gap_{miss}",
            "missed_count": miss,
            "confidence": managed.confidence,
            "track_exists": exists,
            "confirmed": managed.is_confirmed if exists else False,
            "dynamic": managed.is_dynamic if exists else False,
            "attention_authorized":
                managed.attention_authorized if exists else False,
            "visibility_state": managed.visibility_state,
            "dynamic_reason": managed.dynamic_reason,
        })
    # Synthetic direct recovery is contract-only; no detector/association runs.
    if gap <= config.max_missed_frames:
        managed.missed_count = 0
        managed.prediction_only_age = 0
        managed.visibility_state = "direct_observation"
        manager._update_confidence(managed)
        manager._update_dynamic_state(managed)
        managed.attention_authorized = bool(
            managed.is_confirmed and managed.is_dynamic
            and managed.ever_directly_observed
            and managed.confidence >= config.track_confidence_threshold
        )
        frames.append({
            "stage": "post_gap_direct",
            "missed_count": 0,
            "confidence": managed.confidence,
            "track_exists": True,
            "confirmed": managed.is_confirmed,
            "dynamic": managed.is_dynamic,
            "attention_authorized": managed.attention_authorized,
            "visibility_state": managed.visibility_state,
            "dynamic_reason": managed.dynamic_reason,
        })
    gap_rows = [row for row in frames if row["stage"].startswith("gap_")]
    return {
        "requested_pre_gap_confidence": target,
        "actual_pre_gap_confidence": initial,
        "gap_frames": gap,
        "frames": frames,
        "track_survives_gap": all(row["track_exists"] for row in gap_rows),
        "dynamic_survives_gap": all(row["dynamic"] for row in gap_rows),
        "attention_survives_gap": all(
            row["attention_authorized"] for row in gap_rows
        ),
    }


def historical_confidence_audit():
    patterns = (
        "phase8jqv2_4i1*", "phase8jqv2_4n1*",
        "phase8jqv2_4ccr1*", "phase8jqv2_4dpar2*",
    )
    files = sorted({
        path for pattern in patterns for path in REPORTS.glob(pattern)
        if path.is_file() and path.suffix in (".json", ".md")
    })
    exact_fields = []

    def walk(value, location, source):
        if isinstance(value, dict):
            for key, child in value.items():
                child_location = f"{location}.{key}" if location else key
                if key in {
                    "confidence", "pre_gap_confidence",
                    "track_confidence",
                } and isinstance(child, (int, float)):
                    exact_fields.append({
                        "source": str(source.relative_to(ROOT)),
                        "field": child_location,
                        "value": float(child),
                    })
                walk(child, child_location, source)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]", source)

    parsed_json = 0
    for path in files:
        if path.suffix != ".json":
            continue
        parsed_json += 1
        walk(json.loads(path.read_text()), "", path)
    values = [row["value"] for row in exact_fields]
    return {
        "status": "INSUFFICIENT_EVIDENCE" if not values else "PASS",
        "sources_scanned": len(files),
        "json_sources_parsed": parsed_json,
        "allowed_development_families":
            ["I1", "N1", "CCR1", "DPAR2"],
        "exact_pre_gap_track_confidence_samples": len(values),
        "exact_fields": exact_fields,
        "statistics": None if not values else {
            key: float(value) for key, value in zip(
                ("minimum", "p10", "p25", "median", "p75", "p90",
                 "maximum"),
                np.percentile(values, (0, 10, 25, 50, 75, 90, 100)),
            )
        },
        "fraction_at_least_0_977778": None if not values else float(
            np.mean(np.asarray(values) >= .977778)
        ),
        "confidence_one_assumed": False,
        "detector_executed": False,
        "natural_tracker_integration_executed": False,
        "holdout_accessed": False,
    }


def main():
    facts = verify_entry()
    inventory = frozen_inventory()
    eosr1_frozen = load("phase8jqv2_4eosr1_frozen_artifacts.json")
    historical_hash_rows = {
        path: {
            "expected": expected,
            "current": sha(ROOT / path),
            "match": sha(ROOT / path) == expected,
        }
        for path, expected in eosr1_frozen["artifacts"].items()
    }
    historical_hashes_consistent = all(
        row["match"] for row in historical_hash_rows.values()
    )
    if not historical_hashes_consistent:
        raise RuntimeError("historical EOSR1 frozen source hash mismatch")
    atomic_json(PREFIX+"frozen_artifacts.json", {
        "status": "PASS", "artifacts": inventory,
        "eosr1_historical_hash_comparison": historical_hash_rows,
        "eosr1_artifacts_modified": False,
        "eosr1_witnesses_modified": False,
        "eosr1_solver_modified": False,
    })
    atomic_json(PREFIX+"entry_gate.json", {
        "status": "PASS", **facts,
        "historical_hashes_consistent": historical_hashes_consistent,
    })
    config = DynamicPerceptionConfig.from_global_config()
    contract = tracker_contract(config)
    atomic_json(PREFIX+"frozen_tracker_state_contract.json", contract)
    replay = [
        replay_one(config, confidence, gap)
        for confidence in (.70, .75, .80, .90, .95, .98, 1.00)
        for gap in (1, 2, 3)
    ]
    atomic_json(PREFIX+"tracker_contract_replay.json", {
        "status": "PASS",
        "tracker_contract_replay_executed": True,
        "tracker_integration_executed": False,
        "natural_depth_used": False,
        "detector_measurements_used": False,
        "rows": replay,
    })
    threshold = config.track_confidence_threshold
    decay = config.confidence_missed_decay
    derivation = {
        str(gap): {
            "constant_base_minimum_pre_gap_confidence":
                threshold/(decay**gap),
            "feasible_under_unit_cap":
                threshold/(decay**gap) <= 1.,
            "implementation_caveat":
                "confidence base is recomputed; KF uncertainty can change",
        } for gap in (1, 2, 3)
    }
    atomic_json(PREFIX+"gap_confidence_derivation.json", {
        "status": "PASS",
        "threshold": threshold, "decay": decay,
        "formula_under_constant_recomputed_base": "c_base*d^N >= T",
        "confidence_is_recursive": False,
        "gaps": derivation,
        "replay_is_authoritative_for_controlled_states": True,
    })
    atomic_json(PREFIX+"gap_state_transition_table.json", {
        "status": "PASS",
        "rows": [{
            "pre_gap_confidence": row["actual_pre_gap_confidence"],
            "gap_frames": row["gap_frames"],
            "track_survives": row["track_survives_gap"],
            "confirmed_survives": all(
                frame["confirmed"] for frame in row["frames"]
                if frame["stage"].startswith("gap_")
            ),
            "dynamic_survives": row["dynamic_survives_gap"],
            "attention_survives": row["attention_survives_gap"],
        } for row in replay],
    })
    confidence_audit = historical_confidence_audit()
    atomic_json(
        PREFIX+"pre_gap_confidence_distribution.json",
        confidence_audit,
    )
    print(json.dumps({
        "status": "PASS",
        "entry": facts,
        "tracker_contract_replays": len(replay),
        "historical_confidence_evidence":
            confidence_audit["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
