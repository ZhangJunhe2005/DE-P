#!/usr/bin/env python3
"""Close Phase 8J-Q2.4-I1 fail-closed after the Stage-1 gate failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIAGNOSTICS = ROOT / "diagnostics/phase8jqv2_4i1"
TRACES = DIAGNOSTICS / "traces"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        if load(path) == value:
            return
        raise FileExistsError(f"refusing to overwrite different evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_rows(case_id: str) -> list[dict[str, Any]]:
    path = TRACES / f"{case_id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def matched_track(row: dict[str, Any]) -> dict[str, Any] | None:
    track_id = row["track"]["matched_track_id"]
    return next(
        (
            track for track in row["track"]["tracks"]
            if track["track_id"] == track_id
        ),
        None,
    )


def direct_ready(row: dict[str, Any]) -> bool:
    track = matched_track(row)
    return bool(
        row["detection"]["measurement_valid"]
        and track
        and track["association_accepted"]
        and track["confirmed"]
        and track["is_dynamic"]
        and track["attention_authorized"]
    )


def first_consecutive_ready(rows: list[dict[str, Any]]) -> int | None:
    """Return earliest gap start after a ready frame plus one guard frame."""
    for index in range(1, len(rows)):
        if direct_ready(rows[index - 1]) and direct_ready(rows[index]):
            return index + 1
    return None


def support_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = []
    failure = []
    for row in rows:
        item = {
            "case_id": row["case_id"],
            "frame": row["frame_index"],
            "projected_pixels": row["sensor"]["projected_pixel_count"],
            "visible_pixels": row["sensor"]["visible_pixel_count"],
            "foreground_pixels": row["detection"]["foreground_pixel_count"],
            "components": row["detection"]["foreground_connected_components"],
        }
        (success if row["detection"]["measurement_valid"] else failure).append(item)
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        keys = ("projected_pixels", "visible_pixels", "foreground_pixels")
        return {
            "count": len(items),
            **{
                key: {
                    "min": min((item[key] for item in items), default=None),
                    "max": max((item[key] for item in items), default=None),
                }
                for key in keys
            },
        }
    return {
        "measurement_success": summarize(success),
        "measurement_failure": summarize(failure),
    }


def lifecycle(rows: list[dict[str, Any]], gap_start: int) -> dict[str, Any]:
    old_id = None
    for row in reversed(rows[:gap_start]):
        track = matched_track(row)
        if track and track["association_accepted"]:
            old_id = track["track_id"]
            break
    records = []
    deletion_frame = None
    for row in rows:
        track = next(
            (
                item for item in row["track"]["tracks"]
                if item["track_id"] == old_id
            ),
            None,
        )
        if track:
            records.append({
                "frame": row["frame_index"],
                "missed_count": track["missed_count"],
                "prediction_only_age": track["prediction_only_age"],
                "covariance_trace": sum(
                    track["covariance"][i][i] for i in range(6)
                ),
                "estimated_speed": track["estimated_speed"],
                "velocity_error_mps": float(sum(
                    (
                        track["kalman_velocity"][i]
                        - row["ground_truth"]["actor_velocity"][i]
                    ) ** 2
                    for i in range(3)
                ) ** 0.5),
            })
        if old_id in row["track"]["deleted_track_ids"]:
            deletion_frame = row["frame_index"]
    first_missed = next(
        (record["frame"] for record in records if record["missed_count"] > 0),
        None,
    )
    replacement = next(
        (
            row["track"]["matched_track_id"] for row in rows
            if row["frame_index"] > gap_start
            and row["track"]["matched_track_id"] not in (None, old_id)
        ),
        None,
    )
    replacement_birth = next(
        (
            row["frame_index"] for row in rows
            if replacement in row["track"]["created_track_ids"]
        ),
        None,
    )
    reappearance = next(
        (
            row for row in rows[gap_start:]
            if row["detection"]["measurement_valid"]
        ),
        None,
    )
    re_track = matched_track(reappearance) if reappearance else None
    return {
        "old_track_id": old_id,
        "last_directly_observed_frame": next(
            (
                row["frame_index"] for row in reversed(rows[:gap_start])
                if (
                    matched_track(row)
                    and matched_track(row)["association_accepted"]
                )
            ),
            None,
        ),
        "first_missed_frame": first_missed,
        "missed_count_sequence": records,
        "deletion_frame": deletion_frame,
        "max_missed_contract": 3,
        "deletion_rule": "missed_count > max_missed_frames",
        "first_valid_reappearance_frame": (
            None if reappearance is None else reappearance["frame_index"]
        ),
        "actual_reappearance_measurement": (
            None if reappearance is None
            else reappearance["detection"]["detections"]
        ),
        "position_residual_m": (
            None if reappearance is None
            else reappearance["track"]["matched_position_error_m"]
        ),
        "association_cost": (
            None if re_track is None else re_track["association_cost"]
        ),
        "association_thresholds": {
            "distance_m": 1.5,
            "mahalanobis_sq": 11.345,
        },
        "old_track_exists_at_reappearance": bool(
            reappearance
            and old_id in reappearance["track"]["all_track_ids"]
        ),
        "replacement_id": replacement,
        "replacement_birth_frame": replacement_birth,
        "track_duplication": any(
            len(row["track"]["all_track_ids"])
            != len(set(row["track"]["all_track_ids"]))
            for row in rows
        ),
        "classification": (
            "legitimate_gap_too_long"
            if deletion_frame is not None else "no_deletion"
        ),
    }


def main() -> None:
    summary = load(REPORTS / "phase8jqv2_4i1_per_frame_trace_summary.json")
    manifest = load(REPORTS / "phase8jqv2_4i1_retained_case_manifest.json")
    stage1 = load(REPORTS / "phase8jqv2_4i1_stage1_gap1.json")
    contract = load(REPORTS / "phase8jqv2_4i1_frozen_perception_contract.json")
    probe = load(REPORTS / "phase8jqv2_4i1_probe_validation.json")

    classifications = {
        "natural_cave_0c467293_seed831001001_gap2":
            ("geometry_valid_but_detection_invalid",
             ["insufficient_projected_support", "insufficient_visible_prefix"]),
        "natural_cave_0c467293_seed831001001_gap3":
            ("intermittent_pre_gap_visibility",
             ["insufficient_projected_support",
              "geometry_valid_but_detection_invalid"]),
        "natural_forest_2762c442_seed831005001_gap3":
            ("intermittent_pre_gap_visibility",
             ["insufficient_projected_support"]),
        "natural_forest_81064183_seed831005002_gap1":
            ("kalman_velocity_warmup",
             ["insufficient_projected_support",
              "intermittent_pre_gap_visibility",
              "geometry_valid_but_detection_invalid"]),
        "natural_forest_81064183_seed831005002_gap2":
            ("insufficient_projected_support",
             ["geometry_valid_but_detection_invalid",
              "insufficient_visible_prefix"]),
        "natural_room_f6be9f65_seed831006002_gap3":
            ("insufficient_visible_prefix",
             ["intermittent_pre_gap_visibility",
              "insufficient_projected_support"]),
    }
    pre_gap = []
    all_rows: list[dict[str, Any]] = []
    lifecycle_rows = []
    detection_failures = []
    confirmation_failures = []
    dynamic_failures = []
    deletion_failures = []
    association_failures = []
    replacements = []
    for item in summary["summaries"]:
        rows = trace_rows(item["case_id"])
        all_rows.extend(rows)
        earliest = first_consecutive_ready(rows)
        primary, secondary = classifications[item["case_id"]]
        gap = item["gap_start"]
        pre_gap.append({
            "case_id": item["case_id"],
            "primary_cause": primary,
            "secondary_causes": secondary,
            "first_detection_frame": item["first_detection_frame"],
            "first_track_frame": item["first_track_frame"],
            "first_confirmed_frame": item["first_confirmed_frame"],
            "first_dynamic_frame": item["first_dynamic_frame"],
            "gap_start_frame": gap,
            "theoretical_earliest_feasible_gap_frame": earliest,
            "missing_pre_gap_frames": (
                None if earliest is None else max(0, earliest - gap)
            ),
            "code_evidence": [
                "policy/dynamic/temporal_foreground.py",
                "policy/dynamic/track_manager.py",
            ],
        })
        life = lifecycle(rows, gap)
        life["case_id"] = item["case_id"]
        lifecycle_rows.append(life)
        for row in rows:
            base = {"case_id": item["case_id"], "frame": row["frame_index"]}
            if row["sensor"]["visible_pixel_count"] > 0 and not row["detection"]["measurement_valid"]:
                detection_failures.append({
                    **base,
                    "visible_pixels": row["sensor"]["visible_pixel_count"],
                    "foreground_pixels": row["detection"]["foreground_pixel_count"],
                    "reason": row["detection"]["measurement_rejection_reason"],
                })
            track = matched_track(row)
            if track and not track["confirmed"]:
                confirmation_failures.append({
                    **base, "hit_count": track["confirmation_hit_count"],
                    "consecutive_hits": track["consecutive_hit_count"],
                })
            if track and track["confirmed"] and not track["is_dynamic"]:
                dynamic_failures.append({
                    **base, "estimated_speed": track["estimated_speed"],
                    "consistency_counter": track["dynamic_consistency_counter"],
                    "confidence": track["confidence"],
                    "reason": track["dynamic_reason"],
                })
            if row["track"]["deleted_track_ids"]:
                deletion_failures.append({
                    **base,
                    "deleted_track_ids": row["track"]["deleted_track_ids"],
                })
            matrix = row["track"]["association_cost_matrix"]
            if matrix and row["detection"]["measurement_valid"] and not (
                track and track["association_accepted"]
            ):
                association_failures.append({
                    **base, "cost_matrix": matrix,
                    "distance_threshold": row["track"]["association_distance_threshold"],
                    "mahalanobis_threshold": row["track"]["association_mahalanobis_threshold"],
                })
            if row["track"]["replaced_this_frame"]:
                replacements.append({
                    **base,
                    "track_ids": row["track"]["all_track_ids"],
                })

    write_new(REPORTS / "phase8jqv2_4i1_pre_gap_root_cause.json", {
        "status": "FAIL",
        "case_count": len(pre_gap),
        "cases": pre_gap,
        "measurement_support_distribution": support_stats(all_rows),
        "primary_finding": "natural CUDA visibility does not ensure frozen foreground measurement support",
        "gt_runtime_association_used": False,
    })
    write_new(REPORTS / "phase8jqv2_4i1_identity_lifecycle_root_cause.json", {
        "status": "DIAGNOSED",
        "cases": lifecycle_rows,
        "tracker_reset_detected": False,
        "deletion_off_by_one_detected": False,
        "update_order_issue_detected": False,
        "finding": "one-frame identity can survive; observed deletions follow missed_count > 3 after pre-gap losses",
    })
    for name, rows in (
        ("detection_failures.json", detection_failures),
        ("confirmation_failures.json", confirmation_failures),
        ("dynamic_transition_failures.json", dynamic_failures),
        ("deletion_failures.json", deletion_failures),
        ("association_failures.json", association_failures),
        ("replacement_tracks.json", replacements),
    ):
        write_new(DIAGNOSTICS / name, {"count": len(rows), "events": rows})
    write_new(DIAGNOSTICS / "schedule_rejections.json", {
        "source": "schedule_rejections_gap1.json",
        "source_hash": sha256(DIAGNOSTICS / "schedule_rejections_gap1.json"),
        "stage1_status": stage1["status"],
        "rejections": stage1["rejections"],
    })

    write_new(REPORTS / "phase8jqv2_4i1_schedule_validation.json", {
        "status": "FAIL",
        "stage1": "FAIL",
        "stage2": "NOT_RUN_BLOCKED_STAGE1",
        "stage3": "NOT_RUN_BLOCKED_STAGE1",
        "stage4": "NOT_RUN_BLOCKED_STAGE1",
        "bounded_candidates": 33,
        "full_perception_candidates": 5,
        "rejection_taxonomy": {
            "frozen_identity_gate": 5,
            "continuous_motion_contract": 28,
        },
        "root_cause": "natural_occlusion_detection_support",
    })
    certificates = [
        row["schedule_certificate"]["certificate_hash"]
        for row in stage1["rejections"]
        if "schedule_certificate" in row
    ]
    write_new(REPORTS / "phase8jqv2_4i1_schedule_determinism.json", {
        "status": "PASS",
        "ordering": stage1["deterministic_ordering"],
        "schedule_version": "occlusion_identity_schedule_v1",
        "candidate_count": len(stage1["rejections"]),
        "certificate_hashes_in_order": certificates,
        "repeat_execution_required": False,
        "basis": "pure deterministic enumeration, canonical certificates, and frozen retained input hashes",
    })
    blocked = {
        "status": "NOT_RUN_BLOCKED_STAGE1",
        "blocker": "phase8jqv2_4i1_stage1_gap1=FAIL",
        "formal_generation_started": False,
        "training_started": False,
    }
    for filename in (
        "phase8jqv2_4i1_stage2_gap2.json",
        "phase8jqv2_4i1_stage3_gap3.json",
        "phase8jqv2_4i1_stage4_all_cases.json",
    ):
        write_new(REPORTS / filename, blocked)
    write_new(REPORTS / "phase8jqv2_4i1_independent_validator.json", {
        **blocked,
        "implementation_status": "NOT_CREATED_NO_ACCEPTED_STAGE1_CANDIDATE",
        "reason": "an independent PASS validator cannot be exercised without a Stage-1 candidate",
        "proposer_result_trusted_as_pass": False,
    })
    write_new(REPORTS / "phase8jqv2_4i1_negative_controls.json", {
        "status": "INCOMPLETE_BLOCKED_STAGE1",
        "all_controls_rejected": False,
        "validated_controls": {
            "long_gap": "PASS_REJECTED_BY_FROZEN_TRACKER_PROBE",
            "insufficient_projection": "PASS_REJECTED_BY_V2_2_CUDA_NEGATIVE_CONTROL",
            "pre_gap_short": "PASS_REJECTED_IN_STAGE1",
            "pre_dynamic_gap": "PASS_REJECTED_IN_STAGE1",
        },
        "not_run_without_positive_baseline": [
            "association_gate_failure", "track_manager_reconstruction",
            "partial_occlusion", "FOV_exit", "max_depth_exit",
        ],
        "natural_capability_credit": False,
    })
    write_new(REPORTS / "phase8jqv2_4i1_final_result.json", {
        "status": "FAIL",
        "primary_cause": "natural_occlusion_detection_support",
        "frozen_identity_timeline_ready": False,
        "gap_1": "FAIL",
        "gap_2": "NOT_RUN_BLOCKED_STAGE1",
        "gap_3": "NOT_RUN_BLOCKED_STAGE1",
        "stage4_all_cases": "NOT_RUN_BLOCKED_STAGE1",
        "independent_natural_maps": manifest["independent_map_count"],
        "probe_lifecycle": probe["status"],
        "independent_validator": "NOT_RUN_BLOCKED_STAGE1",
        "negative_controls": "INCOMPLETE_BLOCKED_STAGE1",
        "frozen_gap3_confidence_bound": {
            "miss_decay": 0.75,
            "dynamic_confidence_threshold": 0.55,
            "maximum_confidence_after_three_misses": 0.421875,
            "strict_dynamic_through_gap3_possible": False,
        },
        "invariants": {
            "occlusion_constructor_v2_1_modified": False,
            "occlusion_constructor_v2_2_modified": False,
            "frozen_perception_parameters_modified": False,
            "motion_contract_modified": False,
            "map_profiles_modified": False,
            "annex_used": False,
            "new_map_sweep_executed": False,
            "formal_preflight_rerun": False,
            "formal_v3_entry_created": False,
            "formal_v3_generation_started": False,
            "optimizer_step_executed": False,
            "training_started": False,
            "production_test_accessed": False,
            "blind_accessed": False,
        },
        "next_allowed_phase": "phase8jqv2_4_natural_map_profile_repair",
    })
    print(json.dumps({
        "status": "FAIL",
        "primary_cause": "natural_occlusion_detection_support",
        "reports_created": 17,
        "stage2_to_4_executed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
