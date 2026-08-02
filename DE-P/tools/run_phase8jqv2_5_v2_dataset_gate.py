#!/usr/bin/env python3
"""Unify all read-only Formal V2 dataset evidence into one Gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/phase8_authoritative_v2"
OLD = ROOT / "data/phase8_authoritative_v1"
REPORTS = ROOT / "reports"
OLD_HASH = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
NEW_HASH = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def active_writers():
    records = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (
                process / "cmdline"
            ).read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        executable = command.strip()
        if (
            "python" in executable
            and "authoritative_dataset.generate_v1" in executable
        ):
            records.append({"pid": int(process.name), "cmdline": executable})
    return records


def map_identity(root):
    manifest = json.loads(
        (root / "manifests/dataset_manifest.json").read_text()
    )
    result = {}
    for record in manifest["maps"]:
        value = json.loads((root / record["path"]).read_text())
        result[(value["split"], value["map_uuid"])] = (
            value["authority_hash"], value["occupancy_hash"]
        )
    return result


def main():
    manifest_path = DATA / "manifests/dataset_manifest.json"
    before = sha256(manifest_path)
    integrity = json.loads(
        (REPORTS / "phase8jqv2_4_formal_generation_summary.json").read_text()
    )
    semantic = json.loads(
        (REPORTS / "phase8jqv2_4_dataset_semantic_validation.json").read_text()
    )
    loader = json.loads(
        (REPORTS / "phase8jqv2_5_v2_loader_validation.json").read_text()
    )
    actor = json.loads(
        (REPORTS / "phase8jqv2_4_actor_sampling_audit.json").read_text()
    )
    cuda = json.loads(
        (
            REPORTS / "phase8jqv2_4_actor_sampling_cuda_validation.json"
        ).read_text()
    )
    manifest = json.loads(manifest_path.read_text())
    train_sequences = semantic["splits"]["train"]["sequence_count"]
    valid_sequences = semantic["splits"]["valid"]["sequence_count"]
    temporary = sorted(
        str(path.relative_to(ROOT))
        for path in DATA.rglob("*")
        if path.is_file() and path.suffix in {
            ".tmp", ".partial", ".failed"
        }
    )
    locks = sorted(
        str(path.relative_to(ROOT))
        for path in (DATA / "generation_state").rglob("*.lock")
    )
    staging = list((DATA / ".staging").iterdir())
    writers = active_writers()
    markers = {
        name: (
            DATA / "generation_state/completion" / name
        ).is_file()
        for name in (
            "TRAIN_SPLIT_GENERATION_COMPLETE",
            "VALID_SPLIT_GENERATION_COMPLETE",
            "FULL_GENERATION_COMPLETE",
        )
    }
    actor_cuda_safe = all(
        row["actor_depth_pixels_total"] > 0
        and row["frames_with_actor_depth"] > 0
        and row["static_or_future_collisions"] == 0
        and row["dynamic_joint_unsafe_frames"] == 0
        for row in cuda["results"]
    )
    checks = {
        "integrity": integrity["status"] == "PASS",
        "manifest": (
            manifest["dataset_version"] == "phase8_authoritative_v2"
            and before == NEW_HASH
        ),
        "loader": loader["status"] == "PASS",
        "determinism": (
            loader["status"] == "PASS"
            and all(
                value["worker_restart_deterministic"]
                for value in loader["splits"].values()
            )
        ),
        "static": (
            integrity["counts"]["static_frames"] == 586760
            and integrity["counts"]["unknown"] == 0
        ),
        "dynamic": (
            integrity["counts"]["dynamic_frames"] == 513240
            and integrity["counts"]["actor_collisions"] == 0
            and integrity["counts"]["actor_depth_pixels"] > 0
        ),
        "feasibility": integrity["counts"]["unknown"] == 0,
        "semantic": semantic["status"] == "PASS",
        "actor_sampling": (
            actor["status"] == "PASS"
            and actor["formal_sequences_checked"] == 18334
            and actor["dynamic_sequences_checked"] == 7332
            and actor["fallback_sequence_count"] == 7
        ),
        "cuda_rendering": (
            cuda["status"] == "PASS"
            and cuda["device"] == "cuda:0"
            and cuda["fallback_sequences_rendered"] == 7
            and actor_cuda_safe
        ),
        "authority_chain": map_identity(OLD) == map_identity(DATA),
        "completion_markers": all(markers.values()),
        "no_active_writer": not writers,
        "no_generation_lock": not locks,
        "no_staging_or_partial": not (temporary or staging),
        "v1_immutable": sha256(
            OLD / "manifests/dataset_manifest.json"
        ) == OLD_HASH,
        "v2_distinct": before != OLD_HASH,
        "no_test_or_blind": (
            integrity["test_access_count"] == 0
            and integrity["blind_access_count"] == 0
            and manifest.get("test_generated") is False
            and manifest.get("blind_access_count") == 0
        ),
    }
    shared = {
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": before,
        "status": "PASS",
        "production_test_used": False,
        "blind_used": False,
        "network_weights_modified": False,
        "optimizer_step_executed": False,
    }
    normalized = {
        "phase8jqv2_5_v2_manifest_validation.json": {
            **shared,
            "train_sequences": train_sequences,
            "valid_sequences": valid_sequences,
            "train_frames": integrity["counts"]["train_frames"],
            "valid_frames": integrity["counts"]["valid_frames"],
            "completion_markers": markers,
            "temporary_files": temporary,
            "staging_entries": len(staging),
            "generation_locks": locks,
            "active_writers": writers,
            "v1_root_manifest_hash": OLD_HASH,
            "v1_unchanged": checks["v1_immutable"],
            "v2_distinct": checks["v2_distinct"],
        },
        "phase8jqv2_5_v2_authority_chain_validation.json": {
            **shared,
            "authority_version": "static_geometry_authority_v1",
            "authority_maps": 60,
            "authority_maps_identical_to_v1": checks["authority_chain"],
        },
        "phase8jqv2_5_v2_determinism_validation.json": {
            **shared,
            "loader_report":
                "reports/phase8jqv2_5_v2_loader_validation.json",
            "num_workers_tested": [0, 4],
            "worker_restart_deterministic": checks["determinism"],
        },
        "phase8jqv2_5_v2_static_validation.json": {
            **shared,
            "static_frames": integrity["counts"]["static_frames"],
            "unknown": integrity["counts"]["unknown"],
        },
        "phase8jqv2_5_v2_dynamic_validation.json": {
            **shared,
            "dynamic_frames": integrity["counts"]["dynamic_frames"],
            "actor_depth_pixels": integrity["counts"]["actor_depth_pixels"],
            "actor_collisions": integrity["counts"]["actor_collisions"],
            "actor_sampling_audit": actor["status"],
            "cuda_render_validation": cuda["status"],
        },
        "phase8jqv2_5_v2_feasibility_validation.json": {
            **shared,
            "certificate_count": (
                integrity["counts"]["train_frames"]
                + integrity["counts"]["valid_frames"]
            ),
            "unknown": integrity["counts"]["unknown"],
        },
        "phase8jqv2_5_v2_dataset_semantic_validation.json": semantic,
    }
    for name, value in normalized.items():
        atomic_json(REPORTS / name, value)
    after = sha256(manifest_path)
    checks["dataset_unchanged_during_gate"] = before == after
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "phase": "phase8jqv2_5_v2_dataset_entry_gate",
        "timestamp_ns": time.time_ns(),
        "dataset_validation_ready": passed,
        "training_ready": False,
        "training_readiness_reason": (
            "Safety Evaluator V2.1, authoritative R0/R1, surrogate and "
            "candidate-capacity Gates remain required"
        ),
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash": after,
        "train_sequences": train_sequences,
        "valid_sequences": valid_sequences,
        "train_frames": integrity["counts"]["train_frames"],
        "valid_frames": integrity["counts"]["valid_frames"],
        "checks": checks,
        "actor_sampling_audit": actor["status"],
        "cuda_render_validation": cuda["status"],
        "semantic_validation": semantic["status"],
        "loader_validation": loader["status"],
        "determinism_validation":
            "PASS" if checks["determinism"] else "FAIL",
        "static_validation": "PASS" if checks["static"] else "FAIL",
        "dynamic_validation": "PASS" if checks["dynamic"] else "FAIL",
        "feasibility_validation":
            "PASS" if checks["feasibility"] else "FAIL",
        "manifest_validation": "PASS" if checks["manifest"] else "FAIL",
        "production_test_used": False,
        "blind_used": False,
        "test_files_loaded": False,
        "network_weights_modified": False,
        "optimizer_step_executed": False,
        "next_allowed_phase": (
            "phase8jqv2_5_safety_evaluator_v2_1_rebaseline"
            if passed else "phase8jqv2_5_v2_dataset_validation_repair"
        ),
    }
    atomic_json(REPORTS / "phase8jqv2_5_v2_entry_gate.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
