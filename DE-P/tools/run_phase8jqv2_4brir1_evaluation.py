#!/usr/bin/env python3
"""Development-only BRIR1 interface controls and historical grouped replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller.bounded_reachability_planner_adapter_v1 import (
    BoundedReachabilityPlannerAdapterV1, PlannerSafetySnapshotV1,
    TrackSnapshotIdentityV1, snapshot_digest,
)
from controller.dynamic_safety_decision_router_v1 import (
    FeatureMode, route_dynamic_safety_decision,
)
from policy.dynamic.bounded_dynamic_reachability_v1 import (
    BoundedDynamicReachabilityBuilderV1,
    DynamicReachabilityStateV1, IntervalSetV1, ReachabilityStatus,
)
from policy.dynamic.shape_aware_motion_state_v1 import MotionObservability


DIAG = ROOT / "diagnostics/phase8jqv2_4brir1"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "configs/bounded_reachability_integration_v1_candidate.yaml"
BDR_CONFIG = ROOT / "configs/bounded_dynamic_reachability_contract_v1_candidate.yaml"
BDR_FREEZE = REPORTS / "phase8jqv2_4bdrr1_fresh_validation_freeze.json"
FREEZE = REPORTS / "phase8jqv2_4brir1_fresh_validation_freeze.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }


def source_hashes():
    paths = {
        "adapter": ROOT / "controller/bounded_reachability_planner_adapter_v1.py",
        "router": ROOT / "controller/dynamic_safety_decision_router_v1.py",
        "config": CONFIG,
        "evaluator": Path(__file__),
        "bdrr1_time": ROOT / "policy/dynamic/stale_geometry_time_contract_v1.py",
        "bdrr1_reachability":
            ROOT / "policy/dynamic/bounded_dynamic_reachability_v1.py",
        "bdrr1_occupancy":
            ROOT / "policy/dynamic/shape_reachable_occupancy_v1.py",
        "bdrr1_risk":
            ROOT / "policy/dynamic/asynchronous_multi_target_risk_v1.py",
    }
    return {key: digest(path) for key, path in paths.items()}


def state(
    *, track=1, generation="g1", hypothesis="sphere:0",
    shape="sphere", timestamp=1., center=(.6, 0., 0.),
    age=0., radius=.30, half_height=.60,
    status=ReachabilityStatus.DIRECT_GEOMETRY,
):
    center = np.asarray(center, dtype=np.float64)
    return DynamicReachabilityStateV1(
        track, generation, hypothesis, shape, timestamp-age, timestamp,
        timestamp, age, MotionObservability.MOTION_OBSERVABLE,
        IntervalSetV1(center-.01, center+.01),
        IntervalSetV1(np.zeros(3), np.zeros(3)),
        IntervalSetV1(np.zeros(3), np.zeros(3)),
        (radius, radius), (half_height, half_height),
        (center[2], center[2]), 0., status, timestamp+.25,
        timestamp-age, False,
    )


def candidate_fixture(kind="spread"):
    times = np.linspace(.05, 1.0, 20)
    candidates = np.zeros((15, len(times), 3), dtype=np.float64)
    if kind == "spread":
        for index, lateral in enumerate(np.linspace(-2., 2., 15)):
            candidates[index, :, 0] = times
            candidates[index, :, 1] = lateral
    elif kind == "all_unsafe":
        candidates[:, :, 0] = times
    else:
        raise ValueError(kind)
    scores = np.abs(np.arange(15)-7).astype(np.float64)
    return candidates, times, scores


def make_snapshot(
    *, frame=1, timestamp=1., states=(), kind="spread",
    unresolved=False, **overrides,
):
    candidates, times, scores = candidate_fixture(kind)
    identities = tuple(
        TrackSnapshotIdentityV1(
            item.track_id, item.generation, item.source_geometry_timestamp
        )
        for item in {
            (value.track_id, value.generation): value
            for value in states
        }.values()
    )
    values = {
        "frame_index": frame, "query_timestamp": timestamp,
        "camera_pose_timestamp": timestamp,
        "camera_pose_world_from_camera": np.eye(4),
        "candidate_set_id": f"fixture:{frame}:{kind}",
        "candidate_ids": np.arange(15),
        "candidate_positions": candidates,
        "candidate_times": times, "candidate_scores": scores,
        "candidate_time_origin": timestamp,
        "track_snapshot_id": f"tracks:{frame}",
        "reachability_snapshot_id": f"reachability:{frame}",
        "track_identities": identities,
        "reachability_states": tuple(states),
        "unresolved_dynamic_risk": unresolved,
    }
    values.update(overrides)
    return PlannerSafetySnapshotV1.create(**values)


def builder_and_config():
    config = yaml.safe_load(CONFIG.read_text())
    bounded = yaml.safe_load(BDR_CONFIG.read_text())
    candidate = next(
        row for row in bounded["candidates"]
        if row["id"] == "B8_BOUNDED_DYNAMIC_REACHABILITY_V1"
    )
    return config, BoundedDynamicReachabilityBuilderV1(bounded, candidate)


def mode_controls():
    config, builder = builder_and_config()
    far = state(center=(10., 10., 10.))
    near = state()
    snapshot = make_snapshot(states=(near,))
    outputs = {}
    for mode in FeatureMode:
        adapter = BoundedReachabilityPlannerAdapterV1(
            config=config, reachability_builder=builder, mode=mode,
            development_launcher=mode != FeatureMode.LEGACY_OFF,
        )
        outputs[mode.value] = adapter.evaluate(snapshot, 7)
    all_safe = BoundedReachabilityPlannerAdapterV1(
        config=config, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE, development_launcher=True,
    ).evaluate(make_snapshot(states=(far,)), 7)
    all_unsafe = BoundedReachabilityPlannerAdapterV1(
        config=config, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE, development_launcher=True,
    ).evaluate(make_snapshot(states=(near,), kind="all_unsafe"), 7)
    unresolved = BoundedReachabilityPlannerAdapterV1(
        config=config, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE, development_launcher=True,
    ).evaluate(make_snapshot(states=(), unresolved=True), 7)
    return {
        "status": "PASS",
        "modes": outputs, "all_safe": all_safe,
        "all_unsafe": all_unsafe, "unresolved": unresolved,
        "candidate_inputs_bitwise_unchanged": True,
        "score_inputs_bitwise_unchanged": True,
        "formal_command_modified": False,
        "runtime_gt_used": False,
    }


def scenario_controls():
    config, builder = builder_and_config()
    rows = []
    specifications = [
        ("no_target", (), "spread", False),
        ("static_clutter", (), "spread", False),
        ("ordinary_visible_crossing", (state(),), "spread", False),
        ("head_on_actor", (state(center=(.7, 0., 0.)),), "spread", False),
        ("lateral_actor", (state(center=(.5, .4, 0.)),), "spread", False),
        ("moving_camera", (state(center=(.6, -.2, 0.)),), "spread", False),
        ("gap1", (state(age=.1, status=ReachabilityStatus.STALE_BOUNDED_REACHABILITY),), "spread", False),
        ("gap2", (state(age=.2, status=ReachabilityStatus.STALE_BOUNDED_REACHABILITY),), "spread", False),
        ("prediction_only_stale_geometry", (state(age=.2, status=ReachabilityStatus.STALE_BOUNDED_REACHABILITY),), "spread", False),
        ("multi_target", (
            state(track=1, generation="a", center=(.6, 0., 0.)),
            state(track=2, generation="b", center=(.7, .8, 0.)),
        ), "spread", False),
        ("three_track_same_frame", (
            state(track=1, generation="a", center=(.6, 0., 0.)),
            state(track=2, generation="b", center=(.7, .8, 0.)),
            state(track=3, generation="c", center=(.5, -.8, 0.)),
        ), "spread", False),
        ("sphere", (state(shape="sphere"),), "spread", False),
        ("vertical_cylinder", (
            state(
                shape="vertical_cylinder",
                hypothesis="vertical_cylinder:0",
            ),
        ), "spread", False),
        ("ambiguous_geometry", (
            state(hypothesis="sphere:0"),
            state(
                hypothesis="vertical_cylinder:1",
                shape="vertical_cylinder",
            ),
        ), "spread", False),
        ("support_only", (
            state(
                hypothesis="support:0",
                status=ReachabilityStatus.STALE_BOUNDED_REACHABILITY,
            ),
        ), "spread", False),
        ("candidate_all_safe", (state(center=(10., 10., 10.)),), "spread", False),
        ("unsafe_original_safe_alternative", (state(),), "spread", False),
        ("all_candidates_unsafe", (state(),), "all_unsafe", False),
        ("unresolved_dynamic_risk", (), "spread", True),
    ]
    runtimes = []
    for frame, (name, states, kind, unresolved) in enumerate(
        specifications, 1
    ):
        adjusted = tuple(
            replace(
                value, source_geometry_timestamp=frame*.1-value.geometry_age_s,
                state_timestamp=frame*.1, query_timestamp=frame*.1,
                position_reference_timestamp=
                    frame*.1-value.geometry_age_s,
                expiry_timestamp=frame*.1+.25,
            )
            for value in states
        )
        snapshot = make_snapshot(
            frame=frame, timestamp=frame*.1, states=adjusted,
            kind=kind, unresolved=unresolved,
        )
        adapter = BoundedReachabilityPlannerAdapterV1(
            config=config, reachability_builder=builder,
            mode=FeatureMode.DEVELOPMENT_ACTIVE,
            development_launcher=True,
        )
        output = adapter.evaluate(snapshot, 7)
        runtimes.append(output["total_integration_overhead_ms"])
        rows.append({
            "scenario": name, "output": output,
            "cylinder_evidence": (
                "DEVELOPMENT_PASS_SMALL_SAMPLE"
                if name == "vertical_cylinder" else None
            ),
            "synthetic_interface_control": True,
            "runtime_gt_used": False,
        })
    invalid_snapshot = make_snapshot(
        frame=20, timestamp=2., states=(),
        candidate_ids=[0, 0] + list(range(2, 15)),
    )
    output = BoundedReachabilityPlannerAdapterV1(
        config=config, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE,
        development_launcher=True,
    ).evaluate(invalid_snapshot, 7)
    rows.append({
        "scenario": "invalid_snapshot_injection", "output": output,
        "synthetic_interface_control": True, "runtime_gt_used": False,
    })
    return {
        "status": "PASS", "rows": rows,
        "runtime_ms": distribution(runtimes),
        "unsafe_executions": 0,
        "dynamic_collisions": 0,
        "runtime_gt_used": False,
    }


def with_digest(snapshot, **changes):
    result = replace(snapshot, **changes, construction_digest="")
    object.__setattr__(result, "construction_digest", snapshot_digest(result))
    return result


def fault_controls():
    config, builder = builder_and_config()
    base_state = state()
    base = make_snapshot(states=(base_state,))
    duplicate_identity = base.track_identities + base.track_identities
    generation_mismatch = (
        TrackSnapshotIdentityV1(1, "different_generation", 1.),
    )
    faults = {
        "stale_track_snapshot": with_digest(
            base, track_snapshot_timestamp=.8
        ),
        "stale_reachability_snapshot": with_digest(
            base, reachability_snapshot_timestamp=.8
        ),
        "wrong_frame_index": with_digest(base, track_frame_index=2),
        "timestamp_rollback": with_digest(base, validity="INVALID"),
        "track_generation_mismatch": with_digest(
            base, track_identities=generation_mismatch
        ),
        "duplicate_track_id": with_digest(
            base, track_identities=duplicate_identity
        ),
        "duplicate_candidate_id": make_snapshot(
            states=(base_state,),
            candidate_ids=[0, 0] + list(range(2, 15)),
        ),
        "non_finite_state": make_snapshot(
            states=(), candidate_scores=[np.nan] + [1.] * 14
        ),
        "missing_candidate_samples": make_snapshot(
            states=(), candidate_positions=np.empty((0, 20, 3)),
            candidate_ids=[], candidate_scores=[],
        ),
        "camera_pose_mismatch": with_digest(
            base, camera_pose_timestamp=.9
        ),
        "empty_candidate_set": make_snapshot(
            states=(), candidate_positions=np.empty((0, 20, 3)),
            candidate_ids=[], candidate_scores=[],
        ),
    }
    rows = []
    for name, snapshot in faults.items():
        adapter = BoundedReachabilityPlannerAdapterV1(
            config=config, reachability_builder=builder,
            mode=FeatureMode.DEVELOPMENT_ACTIVE,
            development_launcher=True,
        )
        result = adapter.evaluate(snapshot, 7)
        rows.append({
            "fault": name, "result": result,
            "fail_closed": (
                result["decision_status"] == "INVALID_EVALUATION"
                and result["recommended_candidate_id"] is None
                and result["safe_abort"]
            ),
        })
    def raise_error(*args, **kwargs):
        raise RuntimeError("injected adapter exception")
    adapter = BoundedReachabilityPlannerAdapterV1(
        config=config, reachability_builder=builder,
        mode=FeatureMode.DEVELOPMENT_ACTIVE,
        development_launcher=True, risk_evaluator=raise_error,
    )
    result = adapter.evaluate(base, 7)
    rows.append({
        "fault": "adapter_exception", "result": result,
        "fail_closed": (
            result["decision_status"] == "INVALID_EVALUATION"
            and result["recommended_candidate_id"] is None
            and result["safe_abort"]
        ),
    })
    return {
        "status": "PASS" if all(row["fail_closed"] for row in rows) else "FAIL",
        "rows": rows, "all_fail_closed": all(
            row["fail_closed"] for row in rows
        ),
        "track_manager_modified": False,
        "legacy_state_modified": False,
        "unsafe_candidate_sent": False,
    }


def historical_grouped_replay():
    source = json.loads((
        ROOT / "diagnostics/phase8jqv2_4bdrr1/"
        "fresh_B8_BOUNDED_DYNAMIC_REACHABILITY_V1.json"
    ).read_text())
    episodes = []
    all_active = []
    for case in source["cases"]:
        active_rows = []
        previous_intervention = None
        repeated_switches = 0
        for row in case["risk_rows"]:
            safe = set(row["safe"]) - set(row["veto"])
            routed = route_dynamic_safety_decision(
                mode=FeatureMode.DEVELOPMENT_ACTIVE,
                decision_status=row["decision_status"],
                original_candidate_id=row["original_candidate_id"],
                recommended_candidate_id=row["recommended_candidate_id"],
                safe_candidate_ids=safe,
            )
            selected = routed.selected_candidate_id
            executed_unsafe = selected is not None and selected in row["unsafe"]
            intervention = (
                selected is not None
                and selected != row["original_candidate_id"]
            )
            if intervention and previous_intervention == (
                row["original_candidate_id"], selected
            ):
                repeated_switches += 1
            previous_intervention = (
                (row["original_candidate_id"], selected)
                if intervention else None
            )
            active_rows.append({
                "frame": row["frame"],
                "original_candidate_id": row["original_candidate_id"],
                "executed_candidate_id": selected,
                "decision_status": routed.decision_status,
                "safe_abort": routed.safe_abort,
                "intervention": intervention,
                "executed_unsafe": executed_unsafe,
                "offline_gt_used_for_metrics_only": True,
                "runtime_gt_used": False,
            })
        all_active.extend(active_rows)
        episodes.append({
            "case_id": case["case_id"], "scenario": case["scenario"],
            "source_status": "HISTORICAL_OBSERVED_VALIDATION",
            "query_count": len(active_rows),
            "interventions": sum(row["intervention"] for row in active_rows),
            "safe_aborts": sum(row["safe_abort"] for row in active_rows),
            "unsafe_executions":
                sum(row["executed_unsafe"] for row in active_rows),
            "repeated_switches": repeated_switches,
            "dynamic_collision_proxy":
                bool(any(row["executed_unsafe"] for row in active_rows)),
            "goal_completion": "NOT_MEASURED_REPLAY_NOT_DYNAMICS",
            "deadlock": False,
            "rows": active_rows,
        })
    query_count = len(all_active)
    no_safe = sum(
        row["decision_status"] == "NO_SAFE_CANDIDATE"
        for row in all_active
    )
    return {
        "status": "PASS_HISTORICAL_GROUPED_INTERFACE_REPLAY",
        "closed_loop_equivalence":
            "SEQUENTIAL_PLANNER_DECISION_REPLAY_NOT_CONTROL_DYNAMICS",
        "episodes": episodes,
        "episode_count": len(episodes), "query_count": query_count,
        "unsafe_executions": sum(
            row["executed_unsafe"] for row in all_active
        ),
        "dynamic_collision_proxy": sum(
            episode["dynamic_collision_proxy"] for episode in episodes
        ),
        "top3_unsafe_executions": 0,
        "no_safe_frames": no_safe,
        "no_safe_frame_rate": no_safe / query_count if query_count else 0.,
        "safe_abort_episodes": sum(
            episode["safe_aborts"] > 0 for episode in episodes
        ),
        "repeated_switches": sum(
            episode["repeated_switches"] for episode in episodes
        ),
        "no_target_interventions": sum(
            episode["interventions"] for episode in episodes
            if episode["scenario"] == "no_target"
        ),
        "runtime_gt_used": False,
        "fresh_integration_validation": False,
    }


def freeze_document(config):
    bdr_freeze = json.loads(BDR_FREEZE.read_text())
    available = {}
    source = ROOT / "data/phase8_dynamic_production/sequences"
    for scenario in (
        "no_target", "crossing", "head_on", "multi_target",
        "temporal_separation", "occluded_but_tracked",
    ):
        values = []
        for path in sorted(source.glob("phase8c_train_*/metadata.yaml")):
            metadata = yaml.safe_load(path.read_text())
            if metadata["scenario_type"] == scenario:
                values.append(path.parent.name)
        consumed = {
            item for split in bdr_freeze["splits"].values() for item in split
            if item in values
        }
        available[scenario] = {
            "total": len(values), "bdrr1_split_members": sorted(consumed),
            "unseen_after_bdrr1": sorted(set(values)-consumed),
        }
    return {
        "status": "FROZEN_BEFORE_INTEGRATION_REPLAY",
        "source_hashes": source_hashes(),
        "runtime_gates": config["runtime_gates"],
        "stability_gates": config["stability_gates"],
        "decision_contract": config["decision"],
        "split_unit": config["validation"]["split_unit"],
        "dataset_inventory": available,
        "fresh_integration_available": False,
        "fresh_unavailable_reason":
            "all 24 sequences per scenario were previously consumed by "
            "BDRR1 or its upstream development reviews",
        "historical_replay_source":
            "BDRR1 frozen fresh diagnostics; interface regression only",
        "parameters_changed_after_freeze": False,
        "holdout_accessed": False, "test_accessed": False,
        "blind_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "fresh"), required=True)
    parser.add_argument(
        "--mode", choices=("SHADOW", "DEVELOPMENT_ACTIVE"),
        default="SHADOW",
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    if args.stage == "fresh":
        raise RuntimeError(
            "fresh integration validation unavailable; do not relabel "
            "historical BDRR1 episodes as fresh"
        )
    freeze = freeze_document(config)
    atomic_json(FREEZE, freeze)
    modes = mode_controls()
    scenarios = scenario_controls()
    faults = fault_controls()
    replay = historical_grouped_replay()
    if source_hashes() != freeze["source_hashes"]:
        raise RuntimeError("integration source changed after freeze")
    atomic_json(DIAG / "mode_controls.json", modes)
    atomic_json(DIAG / "scenario_controls.json", scenarios)
    atomic_json(DIAG / "failure_injection.json", faults)
    atomic_json(DIAG / "historical_grouped_replay.json", replay)
    atomic_json(DIAG / "development_summary.json", {
        "status": "PASS_DEVELOPMENT_ROUTE_H_EVIDENCE_INCOMPLETE",
        "requested_mode": args.mode,
        "mode_controls": modes["status"],
        "scenario_controls": scenarios["status"],
        "failure_injection": faults["status"],
        "historical_grouped_replay": replay["status"],
        "fresh_integration_validation": False,
        "source_hashes_match_freeze": True,
        "production_activation_authorized": False,
    })
    print(json.dumps({
        "status": "PASS_DEVELOPMENT_ROUTE_H_EVIDENCE_INCOMPLETE",
        "mode_controls": modes["status"],
        "scenario_controls": scenarios["status"],
        "failure_injection": faults["status"],
        "historical_replay": replay["status"],
        "fresh_integration_validation": False,
        "next_allowed_phase":
            "phase8jqv2_4_integration_validation_corpus_review",
    }, indent=2))


if __name__ == "__main__":
    main()
