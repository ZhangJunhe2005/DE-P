#!/usr/bin/env python3
"""Expand the frozen RR1 attempt log into an evidence-preserving taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "data/phase8_natural_representation_audit_v1/generation_attempts.json"
INVENTORY = ROOT / "diagnostics/phase8jqv2_4rr1/corpus_generation/allowed_map_inventory.json"
REPORT_JSON = ROOT / "reports/phase8jqv2_4ce1_rr1_rejection_taxonomy.json"
REPORT_MD = ROOT / "reports/phase8jqv2_4ce1_rr1_rejection_summary.md"
REPLAY = ROOT / "diagnostics/phase8jqv2_4ce1/rr1_attempt_replay.jsonl"

TYPE_NAMES = {1: "cave", 2: "pillar", 5: "forest", 6: "room", 7: "wall"}
TAXONOMY = {
    "A": "no_connected_occluder_patch",
    "B": "v2_2_ratio_interval_empty",
    "C": "camera_pose_infeasible",
    "D": "camera_static_collision",
    "E": "camera_path_collision",
    "F": "actor_initial_collision",
    "G": "actor_path_static_collision",
    "H": "actor_camera_collision",
    "I": "OOB",
    "J": "future_horizon_unsafe",
    "K": "insufficient_pre_visibility",
    "L": "insufficient_post_visibility",
    "M": "FOV_exit",
    "N": "behind_camera",
    "O": "beyond_max_depth",
    "P": "partial_occlusion_only",
    "Q": "gap_zero",
    "R": "gap_longer_than_one",
    "S": "multiple_disjoint_gaps",
    "T": "CUDA_owner_mismatch",
    "U": "rerender_mismatch",
    "V": "timeout",
    "W": "implementation_error",
    "X": "other_with_evidence",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row: dict) -> tuple[str, str, str, str]:
    """Return code, category, stage, and exact evidence interpretation."""
    if row["status"] == "PASS":
        return "PASS", "accepted", "independent_cuda_rerender", "RR1 recorded PASS"
    error = row.get("error", "")
    if error.startswith("TrialTimeout:"):
        return (
            "V", TAXONOMY["V"],
            "bounded_constructor_search",
            "the per-trajectory 30 s wall-clock budget expired",
        )
    if "no detection-feasible natural authority patch" in error:
        return (
            "B", TAXONOMY["B"],
            "camera_patch_broadphase",
            "no sampled natural patch/standoff admitted a v2.2 detection-feasible interval",
        )
    if "pre/post actor exceeds detection-distance contract" in error:
        return (
            "X", TAXONOMY["X"],
            "post_construction_contract_check",
            "constructed actor center exceeded the frozen 1.8 m detection contract before or after the gap",
        )
    if "bounded v2.2 natural-occlusion construction exhausted" in error:
        if "observed_gap_lengths={}" in error:
            return (
                "X", TAXONOMY["X"],
                "exact_raster_gap_search",
                "bounded candidates passed far enough to rasterize but no accepted exact gap length was observed",
            )
        return (
            "X", TAXONOMY["X"],
            "exact_raster_gap_search",
            "the constructor exhausted its recorded raster candidate budget",
        )
    return "W", TAXONOMY["W"], "unknown", error


def main() -> None:
    attempts = json.loads(ATTEMPTS.read_text())
    inventory = json.loads(INVENTORY.read_text())
    maps = {row["map_uuid"]: row for row in inventory["maps"]}
    expanded = []
    for index, row in enumerate(attempts):
        map_row = maps[row["map_uuid"]]
        code, category, stage, evidence = classify(row)
        expanded.append({
            "attempt_id": f"rr1-attempt-{index:04d}",
            "historical_attempt_index": index,
            "status": row["status"],
            "case_id": row.get("case_id"),
            "map_uuid": row["map_uuid"],
            "maze_type": int(row["maze_type"]),
            "natural_type": TYPE_NAMES[int(row["maze_type"])],
            "map_seed": int(map_row["seed"]),
            "authority_hash": map_row["authority_hash"],
            "occupancy_hash": map_row["occupancy_hash"],
            "camera_motion": row["camera_motion"],
            "trajectory_seed": int(row["trajectory_seed"]),
            "requested_gap_frames": 1,
            "patch_id": None,
            "patch_id_evidence": "not_recorded_by_frozen_rr1_attempt_log",
            "actor_direction": None,
            "actor_direction_evidence": "not_recorded_by_frozen_rr1_attempt_log",
            "v2_2_ratio_interval": None,
            "ratio_interval_evidence": "not_recorded_by_frozen_rr1_attempt_log",
            "broadphase_result": (
                "FAIL" if stage == "camera_patch_broadphase" else
                "PASS_OR_NOT_DISTINGUISHABLE"
            ),
            "exact_geometry_result": (
                "FAIL" if stage in {
                    "bounded_constructor_search", "exact_raster_gap_search",
                    "post_construction_contract_check",
                } else "NOT_REACHED_OR_NOT_RECORDED"
            ),
            "continuous_certificate_result": "NOT_RECORDED",
            "cuda_visibility_result": (
                "FAIL" if stage == "exact_raster_gap_search" else
                "NOT_REACHED_OR_NOT_RECORDED"
            ),
            "independent_rerender_result": (
                "PASS" if row["status"] == "PASS" else "NOT_REACHED"
            ),
            "rejection_category": category,
            "rejection_taxonomy_code": code,
            "rejection_stage": stage,
            "evidence": evidence,
            "historical_error": row.get("error", ""),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "replay_kind": "deterministic_log_replay_without_geometry_invention",
        })

    counts = Counter(row["rejection_category"] for row in expanded)
    by_type = defaultdict(Counter)
    by_motion = defaultdict(Counter)
    for row in expanded:
        by_type[row["natural_type"]][row["rejection_category"]] += 1
        by_motion[row["camera_motion"]][row["rejection_category"]] += 1
    payload = {
        "status": "PASS" if len(expanded) == 58 and counts["accepted"] == 3 else "FAIL",
        "phase": "phase8jqv2_4_natural_gap1_representation_corpus_expansion",
        "source_attempt_log": str(ATTEMPTS),
        "source_attempt_log_sha256": sha256(ATTEMPTS),
        "attempts": len(expanded),
        "accepted": counts["accepted"],
        "rejected": len(expanded) - counts["accepted"],
        "taxonomy_counts": dict(sorted(counts.items())),
        "required_taxonomy": {
            code: {
                "name": name,
                "count": counts.get(name, 0),
            } for code, name in TAXONOMY.items()
        },
        "by_natural_type": {
            key: dict(sorted(value.items())) for key, value in sorted(by_type.items())
        },
        "by_camera_motion": {
            key: dict(sorted(value.items())) for key, value in sorted(by_motion.items())
        },
        "evidence_limit": (
            "RR1 did not persist patch IDs, actor directions, feasibility intervals, "
            "or stage-level certificates for rejected attempts; CE1 does not infer them."
        ),
        "forbidden_actions": {
            "detector_used": False,
            "tracker_used": False,
            "sealed_holdout_accessed": False,
            "formal_generation_started": False,
            "training_started": False,
        },
        "rows": expanded,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPLAY.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with REPLAY.open("w") as handle:
        for row in expanded:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# Phase 8J-Q2.4-CE1 RR1 rejection taxonomy",
        "",
        f"- Frozen attempts: {len(expanded)}",
        f"- Accepted: {counts['accepted']}",
        f"- Rejected: {len(expanded) - counts['accepted']}",
        "",
        "## Rejection classes",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines += [
        "",
        "## Evidence boundary",
        "",
        payload["evidence_limit"],
        "The replay is therefore an exact expansion of the immutable log, not a claim that unrecorded geometry was rerun.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print(json.dumps({
        "status": payload["status"],
        "attempts": len(expanded),
        "taxonomy_counts": payload["taxonomy_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
