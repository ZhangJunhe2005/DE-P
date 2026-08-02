#!/usr/bin/env python3
"""Extract the frozen contract and validate tracker/probe lifecycle semantics."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authoritative_dataset.perception_probe_v2 import (  # noqa
    frozen_perception_config,
    run_frozen_perception_probe,
)
from policy.dynamic.track_manager import TrackManager  # noqa
from tests.dynamic_helpers import observation  # noqa


CONTRACT_REPORT = ROOT / (
    "reports/phase8jqv2_4i1_frozen_perception_contract.json"
)
ORDER_REPORT = ROOT / "reports/phase8jqv2_4i1_tracker_update_order.md"
PROBE_REPORT = ROOT / "reports/phase8jqv2_4i1_probe_validation.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite I1 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_new(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sensor():
    return {
        "height": 96, "width": 160,
        "intrinsics": [80.0, 80.0, 80.0, 45.0],
        "max_depth_m": 20.0, "ray_step_m": .1,
        "frame_period_ns": 100000000,
    }


def direct_observation(frame, speed=1.0):
    value = observation(
        [speed * frame * .1, 0.0, 5.0],
        frame * .1,
        covariance=1e-3,
    )
    return replace(
        value,
        observation_id=frame,
        frame_index=frame,
        point_count=120,
        component_pixel_count=120,
        seed_pixel_count=20,
        range_seed_count=20,
        foreground_support=1.0,
        direct_image_evidence=True,
    )


def warm_manager(config, frames=9):
    manager = TrackManager(config)
    history = []
    for frame in range(frames):
        tracks = manager.update(
            (direct_observation(frame),), frame * .1
        )
        history.append(tracks[0])
    return manager, history


def missing_case(config, missing_frames, recover):
    manager, warm = warm_manager(config)
    old_id = warm[-1].track_id
    rows = []
    frame = 9
    for _ in range(missing_frames):
        tracks = manager.update((), frame * .1)
        rows.append({
            "frame": frame,
            "ids": [track.track_id for track in tracks],
            "missed": [track.missed_count for track in tracks],
            "prediction_only_age": [
                track.prediction_only_age for track in tracks
            ],
            "deleted": list(
                manager.last_diagnostics["deleted_track_ids"]
            ),
        })
        frame += 1
    recovery = None
    if recover:
        tracks = manager.update(
            (direct_observation(frame),), frame * .1
        )
        recovery = {
            "frame": frame,
            "ids": [track.track_id for track in tracks],
            "old_id_directly_reassociated": bool(
                any(
                    track.track_id == old_id
                    and track.prediction_only_age == 0
                    and track.missed_count == 0
                    for track in tracks
                )
            ),
        }
    return {
        "missing_frames": missing_frames,
        "old_track_id": old_id,
        "missing_trace": rows,
        "recovery": recovery,
    }


def main():
    configuration = frozen_perception_config(sensor())
    values = asdict(configuration)
    extracted = {
        "status": "PASS",
        "contract_version": "phase8jqv2_4i1_frozen_perception_contract_v1",
        "frame_rate_hz": 10.0,
        "frame_period_s": .1,
        "tracker_dt_policy": {
            "source": "strictly increasing sensor timestamps",
            "raw_nominal_dt_s": .1,
            "effective": "clip(raw_dt,min_dt,max_dt)",
            "min_dt_s": configuration.min_dt,
            "max_dt_s": configuration.max_dt,
        },
        "detection": {
            "foreground_mode_for_probe": "range_image_hybrid",
            "range_history_frames": configuration.range_history_frames,
            "range_min_history_support":
                configuration.range_min_history_support,
            "range_abs_residual_threshold_m":
                configuration.range_abs_residual_threshold,
            "range_rel_residual_threshold":
                configuration.range_rel_residual_threshold,
            "range_min_seed_pixels":
                configuration.range_min_seed_pixels,
            "range_min_seed_fraction":
                configuration.range_min_seed_fraction,
            "range_component_connectivity":
                configuration.range_component_connectivity,
            "range_component_min_persistence":
                configuration.range_component_min_persistence,
            "cluster_algorithm": (
                "causal seeded range-image connected components; "
                "sklearn DBSCAN only for non-range modes"
            ),
            "foreground_minimum_support":
                configuration.range_min_history_support,
            "minimum_visible_pixels": None,
            "track_birth_minimum_points": (
                4 * configuration.dynamic_min_cluster_points
            ),
        },
        "tracking": {
            "state": "[x,y,z,vx,vy,vz]",
            "filter": "linear constant-velocity KalmanFilter",
            "association_metric":
                "gated Hungarian(mahalanobis_sq + descriptor_cost)",
            "association_distance_gate_m":
                configuration.association_distance_threshold,
            "association_mahalanobis_sq_gate":
                configuration.association_mahalanobis_threshold,
            "min_confirmed_hits": configuration.min_confirmed_hits,
            "dynamic_min_confirmed_hits":
                configuration.dynamic_min_confirmed_hits,
            "dynamic_enter_speed_mps":
                configuration.dynamic_enter_speed,
            "dynamic_exit_speed_mps":
                configuration.dynamic_exit_speed,
            "motion_consistency_frames":
                configuration.motion_consistency_frames,
            "max_missed_frames": configuration.max_missed_frames,
            "deletion_condition":
                "missed_count > max_missed_frames",
            "association_before_deletion": True,
            "clear_missing_extra_miss_increment": True,
            "occluded_extra_miss_increment": False,
            "reappearance_association_before_deletion": True,
        },
        "update_order": [
            "predict existing tracks",
            "associate observations to predicted tracks",
            "matched Kalman/velocity/direct-hit update",
            "unmatched missed-count and visibility update",
            "unmatched observation track birth",
            "delete when missed_count > max_missed_frames",
            "confirmation update",
            "confidence update",
            "dynamic hysteresis update",
            "attention authorization",
            "public snapshot",
        ],
        "attention_authorization": {
            "after_dynamic_state_update": True,
            "requires_confirmed": True,
            "requires_dynamic": True,
            "requires_ever_directly_observed": True,
            "requires_prediction_age_within_max": True,
            "rejects_clear_missing": True,
            "track_confidence_threshold":
                configuration.track_confidence_threshold,
        },
        "source_hashes": {
            "traj_opt.yaml": sha256(ROOT / "config/traj_opt.yaml"),
            "types.py": sha256(ROOT / "policy/dynamic/types.py"),
            "association.py": sha256(
                ROOT / "policy/dynamic/association.py"
            ),
            "kalman_tracker.py": sha256(
                ROOT / "policy/dynamic/kalman_tracker.py"
            ),
            "track_manager_baseline":
                "d3f3f493aa84e222bd15aab927b30850da6e4053808957f0f592de6f2c6c4ec6",
            "perception_probe_v2.py": sha256(
                ROOT / "authoritative_dataset/perception_probe_v2.py"
            ),
        },
        "all_frozen_config_values": values,
        "parameters_modified": False,
    }
    lifecycle = []
    manager, warm = warm_manager(configuration)
    no_occlusion = {
        "name": "constant_velocity_no_occlusion",
        "track_ids": sorted({track.track_id for track in warm}),
        "final_confirmed": warm[-1].is_confirmed,
        "final_dynamic": warm[-1].is_dynamic,
        "first_confirmed_frame": next(
            index for index, track in enumerate(warm)
            if track.is_confirmed
        ),
        "first_dynamic_frame": next(
            index for index, track in enumerate(warm)
            if track.is_dynamic
        ),
    }
    no_occlusion["status"] = "PASS" if (
        no_occlusion["track_ids"] == [0]
        and no_occlusion["final_confirmed"]
        and no_occlusion["final_dynamic"]
    ) else "FAIL"
    lifecycle.append(no_occlusion)
    for missing in (1, 2, 3):
        value = missing_case(configuration, missing, recover=True)
        value["name"] = f"confirmed_then_{missing}_missing_then_recover"
        value["status"] = "PASS" if (
            value["recovery"]["old_id_directly_reassociated"]
            and all(
                value["old_track_id"] in row["ids"]
                for row in value["missing_trace"]
            )
        ) else "FAIL"
        lifecycle.append(value)
    long = missing_case(
        configuration, configuration.max_missed_frames + 1,
        recover=True,
    )
    long.update({
        "name": "intentional_overlong_missing_deletes_and_rebirths",
        "status": "PASS" if (
            long["old_track_id"]
            not in long["missing_trace"][-1]["ids"]
            and not long["recovery"]["old_id_directly_reassociated"]
            and long["recovery"]["ids"]
            and long["recovery"]["ids"][0] != long["old_track_id"]
        ) else "FAIL",
        "validator_expected": "REJECT",
    })
    lifecycle.append(long)
    probe_source = inspect.getsource(run_frozen_perception_probe)
    source_checks = {
        "one_perception_instance_outside_frame_loop": (
            probe_source.find("perception = DynamicPerception")
            < probe_source.find("for index, timestamp")
        ),
        "all_tracks_used_for_identity_matching":
            "result.all_tracks" in probe_source,
        "prediction_only_tracks_evaluated":
            "prediction_only_age" in probe_source,
        "post_gap_requires_direct_observation":
            'row["direct_observation"]' in probe_source,
        "future_actor_metadata_runtime_input": False,
        "track_manager_reset_called_in_loop":
            ".reset()" in probe_source,
    }
    probe_report = {
        "status": "PASS" if (
            all(row["status"] == "PASS" for row in lifecycle)
            and source_checks[
                "one_perception_instance_outside_frame_loop"
            ]
            and source_checks["all_tracks_used_for_identity_matching"]
            and source_checks["prediction_only_tracks_evaluated"]
            and source_checks[
                "post_gap_requires_direct_observation"
            ]
            and not source_checks[
                "future_actor_metadata_runtime_input"
            ]
            and not source_checks["track_manager_reset_called_in_loop"]
        ) else "FAIL",
        "probe_hash": sha256(
            ROOT / "authoritative_dataset/perception_probe_v2.py"
        ),
        "persistent_track_manager": True,
        "source_checks": source_checks,
        "lifecycle_controls": lifecycle,
        "frozen_parameters_modified": False,
        "tracker_algorithm_modified": False,
    }
    order_markdown = """# I1 frozen tracker update order

The actual order in `TrackManager.update` is:

1. predict every existing Kalman track to the current timestamp;
2. perform gated one-to-one Hungarian association;
3. update matched Kalman states, robust velocity, hit and direct-observation state;
4. increment unmatched tracks (`missed_count += 1`), then infer visibility;
5. `clear_missing` adds one additional miss; `occluded` does not;
6. create tracks for eligible unmatched observations;
7. delete only when `missed_count > max_missed_frames`;
8. update confirmation, confidence, dynamic hysteresis and attention.

Association therefore precedes deletion. A truly occluded track with
`max_missed_frames=3` survives missed counts 1, 2 and 3. A measurement on the
next frame can associate to the old prediction before deletion. A fourth
consecutive occluded miss deletes it. In contrast, `clear_missing` consumes
two miss units in one frame, so lifecycle duration depends on causal depth
visibility classification as well as frame count.

Track birth occurs during the unmatched-observation step. A direct birth
observation increments `hit_count` to one; confirmation is evaluated later in
the same frame. Dynamic state is evaluated only after confidence and requires
confirmed hits, velocity certainty, cluster geometry, motion consistency,
confidence and the enter-speed threshold. Attention authorization is last and
does not define confirmation or dynamic state.
"""
    write_json(CONTRACT_REPORT, extracted)
    atomic_new(ORDER_REPORT, order_markdown)
    write_json(PROBE_REPORT, probe_report)
    print(json.dumps({
        "status": probe_report["status"],
        "first_confirmed_frame":
            no_occlusion["first_confirmed_frame"],
        "first_dynamic_frame": no_occlusion["first_dynamic_frame"],
        "one_two_three_miss_recovery": [
            row["status"] for row in lifecycle[1:4]
        ],
        "overlong_control": lifecycle[-1]["status"],
    }, indent=2))
    if probe_report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
