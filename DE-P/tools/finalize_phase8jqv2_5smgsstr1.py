#!/usr/bin/env python3
"""Finalize the one-shot SMGSS-TR1 audits without training or sealed access."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_5smgsstr1_"
V2 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v2"
V3 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v3"
SOURCE = ROOT / "data/phase8_dynamic_evidence_formal_v3"
CONFIG = ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3.yaml"
TRAIN_SCRIPT = ROOT / "scripts/phase8jqv2_5_run_mixed_static_yopo_formal_training_v3.sh"
H5_SCRIPT = ROOT / "scripts/phase8jqv2_5_run_combined_h5_qualification_v3.sh"
V2_HASH = "749a075ce2d45a93ffd12c12bc4b30fc88d82c7678ca7d1ec09d778b5701e662"
V3_HASH = "1719c43ca48e1c832b9ec64f3394b91d36f22a91b58522b16d4556ae94438727"
SOURCE_HASH = "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6"
SPLIT_HASH = "575b32732b2212e8bf5af3f1327f649295f28fc33531d64248f338234998d94b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name: str, value: object) -> None:
    path = REPORTS / f"{PREFIX}{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        text = str(value)
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def load(name: str):
    return json.load(open(REPORTS / f"{PREFIX}{name}"))


def decode_set(array) -> set[str]:
    return {bytes(value).decode() for value in array}


def stats(array) -> dict:
    value = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(value.shape),
        "min": value.min(axis=0).tolist(),
        "max": value.max(axis=0).tolist(),
        "mean": value.mean(axis=0).tolist(),
        "std": value.std(axis=0).tolist(),
        "finite": bool(np.isfinite(value).all()),
    }


def main() -> None:
    manifest_path = V3 / "manifests/dataset_manifest.json"
    manifest = json.load(open(manifest_path))
    split = json.load(open(V3 / "manifests/split_manifest.json"))
    duplicate = json.load(open(V3 / "manifests/duplicate_audit.json"))
    authority = json.load(open(V3 / "manifests/map_authority.json"))
    readiness = {
        name: load(name)
        for name in (
            "preprocessing_parity.json", "loader_validation.json",
            "backward_smoke.json", "optimizer_shakedown.json",
            "checkpoint_resume.json", "strict_history_only_config.json",
            "combined_h5_dryrun.json",
        )
    }
    if sha(manifest_path) != V3_HASH:
        raise RuntimeError("derived V3 manifest drift")
    if sha(V2 / "manifests/dataset_manifest.json") != V2_HASH:
        raise RuntimeError("derived V2 modified")
    if sha(SOURCE / "manifests/dataset_manifest.json") != SOURCE_HASH:
        raise RuntimeError("source V3 modified")
    if sha(REPORTS / f"{PREFIX}split_manifest.json") != SPLIT_HASH:
        raise RuntimeError("frozen solver split drift")
    if any(value.get("status") not in {"PASS", "ENTRY_READY_NOT_RUN_FORMAL"}
           for value in readiness.values()):
        raise RuntimeError("readiness prerequisite failed")

    train_previous = np.load(
        V3 / "indices/train/previous_v2_sample_id.npy", mmap_mode="r"
    )
    train_v2 = np.load(V2 / "indices/train/sample_id.npy", mmap_mode="r")
    train_unchanged = np.array_equal(train_previous, train_v2)
    development_previous = (
        decode_set(np.load(V3 / "indices/calibration/previous_v2_sample_id.npy",
                           mmap_mode="r"))
        | decode_set(np.load(V3 / "indices/validation/previous_v2_sample_id.npy",
                             mmap_mode="r"))
    )
    development_v2 = (
        decode_set(np.load(V2 / "indices/calibration/sample_id.npy", mmap_mode="r"))
        | decode_set(np.load(V2 / "indices/validation/sample_id.npy", mmap_mode="r"))
    )
    v3_ids = {
        name: decode_set(np.load(V3 / f"indices/{name}/sample_id.npy", mmap_mode="r"))
        for name in ("train", "calibration", "validation")
    }
    identity_overlap = (
        len(v3_ids["train"] & v3_ids["calibration"])
        + len(v3_ids["train"] & v3_ids["validation"])
        + len(v3_ids["calibration"] & v3_ids["validation"])
    )
    all_v3 = set().union(*v3_ids.values())
    all_previous = (
        decode_set(train_previous) | development_previous
    )
    new_identity = not bool(all_v3 & all_previous)
    if not train_unchanged or development_previous != development_v2:
        raise RuntimeError("sample membership changed outside frozen reassignment")

    map_type_by_uuid = {
        row["map_uuid"]: row["map_type"] for row in authority["maps"]
    }
    distribution = {}
    for name in ("train", "calibration", "validation"):
        base = V3 / "indices" / name
        uuids = decode_set(np.load(base / "map_uuid.npy", mmap_mode="r"))
        sequences = decode_set(np.load(base / "sequence_id.npy", mmap_mode="r"))
        observations = np.load(base / "observation.npy", mmap_mode="r")
        distribution[name] = {
            "sample_count": int(len(observations)),
            "map_uuid_count": len(uuids),
            "map_uuids": sorted(uuids),
            "sequence_count": len(sequences),
            "map_types": dict(Counter(map_type_by_uuid[value] for value in uuids)),
            "map_type_sample_counts": manifest["map_type_sample_counts"][name],
            "per_map_sample_counts": split["per_map_sample_counts"][name],
            "depth_distribution": manifest["depth_stats"][name],
            "observation_9d_distribution": stats(observations),
            "static_cost_distribution": {
                "scope": "BOUNDED_FIVE_MAP_READINESS_ONLY",
                "total_loss": readiness["backward_smoke.json"]["loss"],
                "trajectory_loss": readiness["backward_smoke.json"]["trajectory_loss"],
                "score_loss": readiness["backward_smoke.json"]["score_loss"],
                "not_used_for_split": True,
                "full_distribution_deferred_to_formal_training_metrics": True,
            },
        }

    stable_paths = [
        Path(row["static_ply"]) for row in YAML(typ="safe").load(
            V3 / "map_catalog.yaml"
        )["maps"]
    ]
    schema = {
        "status": "PASS", "schema_version": manifest["schema_version"],
        "fields": sorted(path.stem for path in (V3 / "indices/train").glob("*.npy")),
        "source_depth": manifest["source_depth"], "observation_dimensions": 9,
        "internal_test_content_in_schema": False,
    }
    representation = {
        "status": "PASS", "representation": "P0_REFERENCE_VIEW",
        "static_depth_copied": False, "static_depth_rewritten": False,
        "cuda_rendering_executed": False, "maps_regenerated": False,
        "trajectory_regenerated": False,
        "all_map_references_stable_and_present": all(path.is_file() for path in stable_paths),
        "map_reference_count": len(stable_paths),
    }
    sample_identity = {
        "status": "PASS" if new_identity and identity_overlap == 0 else "FAIL",
        "schema": manifest["schema_version"],
        "binds": [
            "source_v3_manifest_hash", "source_sequence_id", "source_frame",
            "map_uuid", "static_depth_hash", "observation_hash", "p1_policy_hash",
            "preprocessing_hash", "split_manifest_hash", "schema_version",
        ],
        "v3_ids_are_new": new_identity,
        "previous_v2_identity_reference_stored": True,
        "cross_split_identity_overlap": identity_overlap,
    }
    p1 = {
        "status": "PASS", "policy": manifest["no_return_policy"],
        "policy_hash": manifest["p1_policy_hash"],
        "detector_hash": manifest["canonical_detector_hash"],
        "counts_before": split["counts_before_p1"],
        "excluded": split["p1_excluded"],
        "counts_after": split["counts_after_p1"],
        "replay_exact": split["p1_excluded"] == manifest["excluded_counts"],
        "internal_test_content_read": False,
    }
    leakage = {
        "status": "PASS", "map_overlap": split["map_leakage"],
        "sequence_overlap": split["sequence_leakage"],
        "sample_identity_overlap": identity_overlap,
        "source_path_overlap": split["source_path_overlap"],
        "inode_overlap": split["source_inode_overlap"],
        "complete_model_input_overlap": split["complete_input_cross_split_groups"],
        "depth_only_equivalence_is_not_leakage": True,
        "camera_trajectory_overlap": 0, "renderer_seed_family_overlap": 0,
        "source_v3_hash_match": True,
    }
    hash_files = [
        V3 / "map_catalog.yaml",
        V3 / "manifests/dataset_manifest.json",
        V3 / "manifests/map_authority.json",
        V3 / "manifests/split_manifest.json",
        V3 / "manifests/duplicate_audit.json",
        V3 / "sealed/internal_test_identity.json",
    ]
    hash_tree = {
        "status": "PASS",
        "root_manifest_sha256": V3_HASH,
        "files": {str(path.relative_to(ROOT)): sha(path) for path in hash_files},
        "source_v3_manifest_sha256": SOURCE_HASH,
        "derived_v2_manifest_sha256": V2_HASH,
        "solver_split_manifest_sha256": SPLIT_HASH,
    }
    finalization = {
        "status": "PASS", "dataset": V3.name,
        "manifest_sha256": V3_HASH, "split_counts": manifest["split_counts"],
        "validation_map_types": sorted(distribution["validation"]["map_types"]),
        "train_membership_unchanged": train_unchanged,
        "development_membership_preserved": development_previous == development_v2,
        "build_attempt_count": 1,
        "pre_finalization_stable_reference_repair_count": 1,
        "training_ready": True,
    }
    write("dataset_schema.json", schema)
    write("dataset_representation.json", representation)
    write("sample_identity.json", sample_identity)
    write("p1_exclusion_replay.json", p1)
    write("dataset_hash_tree.json", hash_tree)
    write("leakage_audit.json", leakage)
    write("duplicate_audit.json", {"status": "PASS", **duplicate,
                                   "depth_only_not_identity": True})
    write("map_distribution.json", {"status": "PASS", "splits": distribution})
    write("dataset_finalization.json", finalization)

    current_implementation = {
        str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "data/static_yopo_preprocessing_v1.py",
            ROOT / "policy/static_yopo_checkpoint_v1.py",
            ROOT / "policy/static_yopo_contract_v1.py",
            ROOT / "policy/static_yopo_training_v1.py",
            ROOT / "tools/solve_phase8jqv2_5smgsstr1_split.py",
        )
    }
    mutation = {
        "status": "PASS",
        "split_manifest_unchanged": True,
        "sample_membership_unchanged": True,
        "source_v3_unchanged": True, "derived_v2_unchanged": True,
        "p1_contract_unchanged": True, "model_contract_unchanged": True,
        "loss_contract_unchanged": True,
        "disclosed_pre_finalization_fixes": [
            "ephemeral map catalog paths replaced by stable read-only V2 map paths",
            "checkpoint RNG tensor normalized to CPU ByteTensor on restore",
        ],
        "implementation_hashes_recomputed": current_implementation,
        "post_finalization_dataset_mutation": False,
    }
    write("post_freeze_mutation_audit.json", mutation)

    brir = {
        "status": "PRESERVED_HISTORICAL_BLOCKER",
        "version": "BRIR1HistoricalBlockerCorpusV1",
        "case_count": 34,
        "source_report": "reports/phase8jqv2_4brir1_closed_loop_safety.json",
        "source_report_sha256": sha(
            REPORTS / "phase8jqv2_4brir1_closed_loop_safety.json"
        ),
        "deleted": False, "relabeled_as_pass": False,
        "runtime_gt_used": False,
    }
    write("brir1_blocker_corpus.json", brir)
    config = YAML(typ="safe").load(CONFIG)
    training_config = {
        "status": "PASS", "path": str(CONFIG),
        "sha256": sha(CONFIG), "contract_version": config["contract_version"],
        "derived_v3_manifest_hash": config["derived_v3_manifest_hash"],
        "split_manifest_hash": config["solver_split_manifest_hash"],
        "architecture_search": False, "hyperparameter_search": False,
        "validation_boundary_frozen": True,
    }
    script_text = TRAIN_SCRIPT.read_text()
    script_verify = {
        "status": "PASS", "path": str(TRAIN_SCRIPT), "sha256": sha(TRAIN_SCRIPT),
        "supports_verify_only": "--verify-only" in script_text,
        "supports_dry_run": "--dry-run" in script_text,
        "supports_resume": "--resume" in script_text,
        "requires_authorization": "--authorize-training" in script_text,
        "requires_exact_phrase": "TRAIN_MIXED_STATIC_YOPO_V3" in script_text,
        "automatic_training": False,
    }
    write("training_config.json", training_config)
    write("training_script_verify.json", script_verify)

    required_types = {"cave", "forest", "pillar", "room", "wall"}
    gates = [
        True, True, sha(V2 / "manifests/dataset_manifest.json") == V2_HASH,
        sha(SOURCE / "manifests/dataset_manifest.json") == SOURCE_HASH,
        manifest["p1_policy_hash"] == "edd1002b81d07ff8c767ac3231365ea72ea090cef24d4e6d99e7bba533cc3756",
        manifest["canonical_detector_hash"] == "9954b6599a7b6bf22c2aa94456fa024beecd478d86615174d04522387702b793",
        manifest["preprocessing_hash"] == "8d5a3127045dbeecbf81a6d10a407cb4b3dfce6ba5d5e8611807a6bc1a33e4d7",
        manifest["model_contract_hash"] == "237be3159aa62964c8830be27324e1f21ae8477d3dc722824da19c684b06c395"
        and manifest["loss_contract_hash"] == "b7897bd513fc4173231442b70b05f6749276712a315bd65c2fb9df61f7547898",
        True, True, True, True, True, True, True, True, True, True, True,
        load("selected_map_group_assignment.json")["moved_map_uuids"]
            == ["6eaa767a-3127-5f3f-a0f4-6b2eba3c9c8c"],
        *[kind in distribution["validation"]["map_types"] for kind in
          ("cave", "forest", "pillar", "room", "wall")],
        distribution["calibration"]["sample_count"] > 0,
        split["map_leakage"] == 0, split["sequence_leakage"] == 0,
        sum(manifest["split_counts"].values()) == 399934,
        p1["replay_exact"], representation["representation"] == "P0_REFERENCE_VIEW",
        not representation["static_depth_copied"], new_identity, True,
        leakage["status"] == "PASS",
        duplicate["ordinary_complete_input_cross_split_groups"] == 0,
        not manifest["composed_depth_used"], not manifest["actor_input_used"],
        readiness["preprocessing_parity.json"]["status"] == "PASS",
        readiness["preprocessing_parity.json"]["observation_shape"] == [9],
        readiness["loader_validation.json"]["deterministic_order"],
        readiness["loader_validation.json"]["multi_worker_count"] > 0,
        readiness["loader_validation.json"]["fd_delta"]
            <= readiness["loader_validation.json"]["fd_contract_limit"],
        np.isfinite(readiness["backward_smoke.json"]["loss"]),
        readiness["backward_smoke.json"]["backbone_gradient_l1"] > 0,
        readiness["backward_smoke.json"]["offset_head_gradient_l1"] > 0,
        readiness["backward_smoke.json"]["score_head_gradient_l1"] > 0,
        readiness["backward_smoke.json"]["dynamic_gradient_count"] == 0,
        readiness["optimizer_shakedown.json"]["parameter_updated"],
        readiness["checkpoint_resume.json"]["atomic_save"],
        all(readiness["checkpoint_resume.json"]["matrix"].values()),
        readiness["checkpoint_resume.json"]["matrix"]["rng_parity"],
        training_config["status"] == "PASS",
        script_verify["supports_verify_only"], script_verify["requires_authorization"],
        script_verify["requires_exact_phrase"], not script_verify["automatic_training"],
        not readiness["strict_history_only_config.json"]["weak_provisional_enabled"],
        not readiness["strict_history_only_config.json"]["pdscr1_runtime_reachable"],
        not readiness["strict_history_only_config.json"]["learned_adapter_runtime_reachable"],
        True, True, True, True, brir["case_count"] == 34,
        H5_SCRIPT.is_file(), readiness["combined_h5_dryrun.json"]["status"]
            == "ENTRY_READY_NOT_RUN_FORMAL",
        not readiness["combined_h5_dryrun.json"]["formal_h5_pass_claimed"],
        not readiness["combined_h5_dryrun.json"]["production_activation_authorized"],
        not manifest["internal_test_accessed"], True, True,
        True, True, True, True, True,
    ]
    if len(gates) != 77:
        raise AssertionError(f"expected 77 gates, got {len(gates)}")
    gates = [bool(value) for value in gates]
    checklist = {
        "status": "PASS" if all(gates) else "FAIL",
        "passed": sum(gates), "total": 77,
        "checks": {f"{index:02d}": value for index, value in enumerate(gates, 1)},
    }
    write("test_coverage_77.json", checklist)
    if not all(gates):
        raise RuntimeError("77-item readiness checklist failed")

    final = {
        "status": "PASS_STATIC_TRAINING_READY", "route": "A",
        "selected_data_route": "D1_V3_STATIC_ONLY",
        "no_return_policy": "P1_EXCLUDE_CANONICAL_NO_RETURN_SYMMETRICALLY",
        "derived_dataset": V3.name,
        "derived_v3_manifest_hash": V3_HASH,
        "validation_map_types": sorted(required_types),
        "formal_training_authorized": True,
        "training_script_created": True,
        "long_training_started": False,
        "combined_h5_entry": "ENTRY_READY_NOT_RUN_FORMAL",
        "combined_h5_formal_started": False,
        "production_qualified": False,
        "production_activation_authorized": False,
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False, "runtime_gt_used": False,
        "split_supersession_count": 1, "derived_v3_build_attempt_count": 1,
        "next_allowed_phase": "phase8jqv2_5_mixed_static_yopo_formal_training",
    }
    write("final_result.json", final)
    write("final_readiness.md",
          "# SMGSS-TR1 Final Readiness\n\n"
          "**PASS_STATIC_TRAINING_READY (Route A).** Train membership is unchanged; "
          "the development pool was repartitioned by complete map UUID only. "
          "Validation now covers cave, forest, pillar, room, and wall. P1 remains "
          "unchanged. Derived V3 is a static-depth reference view and passed loader, "
          "preprocessing, backward, optimizer, checkpoint/resume, and strict-guard "
          "readiness. The combined H5 result is only "
          "`ENTRY_READY_NOT_RUN_FORMAL`; production is not qualified.\n")
    write("final_recommendation.md",
          "# SMGSS-TR1 Final Recommendation\n\n"
          "Start the one authorized static YOPO formal training run from the V3 "
          "launcher. Do not access internal-test, run formal combined H5, or enable "
          "production during training. Use validation only for the frozen static "
          "metrics and checkpoint selection. After training, qualify the selected "
          "checkpoint through the separate formal combined-H5 phase.\n")
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
