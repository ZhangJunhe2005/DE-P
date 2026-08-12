#!/usr/bin/env python3
"""Formal static Route-A trainer. It is reachable only through the authorized shell entry."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from data.static_yopo_loader_v1 import make_static_yopo_loader_v1
from data.static_yopo_subset_v1 import StaticYOPOMapTypeSubsetV1
from policy.static_yopo_wide_state_v1 import StaticYOPOWideStateDatasetV1
from policy.static_yopo_recovery_state_v2 import StaticYOPORecoveryStateDatasetV2
from policy.static_yopo_recovery_state_v4_8 import (
    StaticYOPORecoveryStateDatasetV48,
)
from policy.static_yopo_original_state_v4_4 import (
    StaticYOPOOriginalStateDatasetV44,
)
from data.static_yopo_manifest_v1 import (
    INITIAL_CHECKPOINT_SHA256, SOURCE_V3_MANIFEST_SHA256, sha256_file,
)
from policy.static_yopo_checkpoint_v1 import (
    load_static_yopo_checkpoint_v1, save_static_yopo_checkpoint_v1,
)
from policy.static_yopo_contract_v1 import (
    LOSS_CONTRACT_HASH, MODEL_CONTRACT_HASH, NORMALIZATION_CONTRACT_HASH,
)
from policy.static_yopo_training_v1 import (
    MixedSceneStaticYOPOObjectiveV1, MixedSceneStaticYOPOV1,
)
from policy.static_yopo_objective_contract_v1 import (
    LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH,
)

TRAINING_IMPLEMENTATION_FILES = (
    ROOT / "tools/train_mixed_static_yopo_v1.py",
    ROOT / "policy/static_yopo_training_v1.py",
    ROOT / "policy/static_yopo_checkpoint_v1.py",
    ROOT / "policy/checkpoint_utils.py",
    ROOT / "policy/models/head.py",
    ROOT / "data/static_yopo_loader_v1.py",
    ROOT / "policy/static_yopo_wide_state_v1.py",
    ROOT / "policy/static_yopo_recovery_state_v2.py",
    ROOT / "policy/static_yopo_recovery_state_v4_8.py",
    ROOT / "policy/state_transform.py",
    ROOT / "policy/static_yopo_safety_first_v1.py",
    ROOT / "policy/static_yopo_kinodynamic_v2.py",
    ROOT / "policy/static_yopo_preventive_safety_v1.py",
    ROOT / "policy/static_yopo_progress_safety_v1.py",
    ROOT / "policy/static_yopo_goal_progress_v2.py",
    ROOT / "policy/static_yopo_feasibility_score_v1.py",
    ROOT / "policy/static_yopo_projected_score_v3.py",
    ROOT / "policy/static_yopo_simple_v4_3.py",
    ROOT / "policy/static_yopo_original_state_v4_4.py",
    ROOT / "policy/static_yopo_parity_v4_4.py",
    ROOT / "policy/static_yopo_parity_v4_5.py",
    ROOT / "policy/static_yopo_parity_v4_5_1.py",
    ROOT / "policy/static_yopo_parity_v4_5_2.py",
    ROOT / "policy/static_yopo_parity_v4_5_3.py",
    ROOT / "policy/static_yopo_parity_v4_5_4.py",
    ROOT / "policy/static_yopo_parity_v4_5_5.py",
    ROOT / "policy/static_yopo_parity_v4_5_6.py",
    ROOT / "policy/static_yopo_parity_v4_5_8.py",
    ROOT / "policy/static_yopo_parity_v4_5_9.py",
    ROOT / "policy/static_yopo_parity_v4_5_10.py",
    ROOT / "policy/static_yopo_recovery_coverage_v4_8.py",
    ROOT / "data/static_yopo_subset_v1.py",
    ROOT / "loss/loss_function.py",
    ROOT / "loss/safety_loss.py",
)


def training_implementation_hash():
    import hashlib

    digest = hashlib.sha256()
    for path in TRAINING_IMPLEMENTATION_FILES:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def source_identity_fields(source_manifest_hash):
    """Return the current identity plus the checkpoint-v1 compatibility key."""
    return {
        "source_dataset_manifest_hash": source_manifest_hash,
        # Checkpoint v1 predates non-V3 source datasets.  Keep its frozen key
        # as an alias so old checkpoints remain readable without weakening
        # the identity value.
        "source_v3_manifest_hash": source_manifest_hash,
    }


def training_contract_classification(config):
    """Classify qualification semantics without interpreting loss as a Gate."""
    validation_contract = config.get("validation", {}).get("contract_version")
    return {
        "three_layer": validation_contract == "route_a_v4_2_9_three_layer_v1",
        "ungated_static": validation_contract in {
            "route_a_v4_3_single_loss_v1",
            "route_a_v4_4_static_parity_v1",
            "route_a_v4_5_bounded_danger_v1",
            "route_a_v4_5_1_relative_kinematic_v1",
            "route_a_v4_5_2_calibrated_relative_kinematic_v1",
            "route_a_v4_5_3_continuous_clearance_two_stage_v1",
            "route_a_v4_5_4_independent_score_only_v1",
            "route_a_v4_5_5_dense_static_esdf_v1",
            "route_a_v4_5_6_continuous_score_safety_v1",
            "route_a_v4_5_7_high_safety_retimed_v1",
            "route_a_v4_5_8_localized_safety_retimed_v1",
            "route_a_v4_5_9_time_mean_localized_safety_v1",
            "route_a_v4_5_10_tail_aware_safety_v1",
            "route_a_v4_8_recovery_capacity_v1",
        },
        "diagnostic": config.get("experiment_role") in {
            "map_type_convergence_probe",
            "mixed_five_epoch_shakedown",
            "mixed_two_stage_eight_epoch_shakedown",
            "independent_score_five_epoch_shakedown",
            "dense_static_esdf_five_epoch_shakedown",
            "continuous_safety_score_calibration",
            "high_safety_joint_retiming_shakedown",
            "localized_safety_retiming_shakedown",
            "time_mean_localized_safety_shakedown",
            "tail_aware_safety_shakedown",
            "tail_aware_controlled_continuation",
            "recovery_capacity_five_epoch_shakedown",
        },
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def load_contract(config_path):
    config_path = Path(config_path).expanduser().resolve()
    config = YAML(typ="safe").load(config_path)
    derived = Path(config["derived_dataset_root"])
    manifest_path = derived / "manifests" / "dataset_manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    source_manifest_path = (
        Path(config["source_dataset_root"]) / "manifests/dataset_manifest.json"
    )
    expected_source = config.get(
        "source_dataset_manifest_hash",
        config.get("source_v3_manifest_hash", SOURCE_V3_MANIFEST_SHA256),
    )
    expected_normalization = config.get(
        "normalization_contract_hash", NORMALIZATION_CONTRACT_HASH
    )
    expected_model = config.get("model_contract_hash", MODEL_CONTRACT_HASH)
    expected_loss = config.get("loss_contract_hash", LOSS_CONTRACT_HASH)
    checks = {
        "source_v3": (
            sha256_file(source_manifest_path), expected_source,
        ),
        "initial_checkpoint": (
            sha256_file(config["model"]["initial_checkpoint"]),
            config["model"].get(
                "initial_checkpoint_sha256", INITIAL_CHECKPOINT_SHA256
            ),
        ),
        "preprocessing": (manifest["preprocessing_hash"],
                          __import__("data.static_yopo_preprocessing_v1",
                                     fromlist=["DEFAULT_STATIC_YOPO_PREPROCESSOR_V1"])
                          .DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash),
        "normalization": (manifest["normalization_hash"], expected_normalization),
        "model": (manifest["model_contract_hash"], expected_model),
        "loss": (manifest["loss_contract_hash"], expected_loss),
        "training_implementation": (
            training_implementation_hash(), config["training_implementation_hash"]
        ),
    }
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    if config.get("p1_policy_hash") is not None:
        checks["p1_policy"] = (
            manifest.get("p1_policy_hash"), config["p1_policy_hash"]
        )
    if config.get("derived_v3_manifest_hash") is not None:
        checks["derived_v3"] = (
            sha256_file(manifest_path), config["derived_v3_manifest_hash"]
        )
    if config.get("solver_split_manifest_hash") is not None:
        checks["solver_split"] = (
            manifest.get("solver_split_manifest_sha256"),
            config["solver_split_manifest_hash"],
        )
    if config.get("derived_dataset_manifest_hash") is not None:
        checks["derived_dataset"] = (
            sha256_file(manifest_path),
            config["derived_dataset_manifest_hash"],
        )
    if config.get("objective_contract_hash") is not None:
        checks["objective"] = (
            LOCAL_GOAL_OBJECTIVE_CONTRACT_V1_HASH,
            config["objective_contract_hash"],
        )
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    if mismatches:
        raise RuntimeError(f"training identity preflight failed: {mismatches}")
    identities = {
        "dataset_manifest_hash": sha256_file(manifest_path),
        "split_hash": manifest["split_hash"],
        "preprocessing_hash": manifest["preprocessing_hash"],
        "normalization_hash": manifest["normalization_hash"],
        "model_contract_hash": manifest["model_contract_hash"],
        "loss_contract_hash": manifest["loss_contract_hash"],
        "training_config_hash": sha256_file(config_path),
        "training_implementation_hash": training_implementation_hash(),
        "training_contract_version": str(config["contract_version"]),
        **source_identity_fields(expected_source),
    }
    if config.get("objective_contract_hash") is not None:
        identities["objective_contract_hash"] = config[
            "objective_contract_hash"
        ]
    return config_path, config, derived, manifest, identities


def verify(config_path):
    config_path, config, derived, manifest, identities = load_contract(config_path)
    train_samples = int(manifest["split_counts"]["train"])
    validation_samples = int(manifest["split_counts"]["validation"])
    map_type_filter = config.get("loader", {}).get("map_type_filter")
    if map_type_filter is not None:
        authority = json.load(open(
            derived / "manifests" / "map_authority.json", encoding="utf-8"
        ))["maps"]
        by_map_id = {
            int(row["map_id"]): str(row["map_type"]) for row in authority
        }
        filtered_counts = {}
        for split in ("train", "validation"):
            map_ids = np.load(
                derived / "indices" / split / "map_id.npy", mmap_mode="r"
            )
            filtered_counts[split] = sum(
                by_map_id[int(map_id)] == map_type_filter
                for map_id in map_ids
            )
        train_samples = int(filtered_counts["train"])
        validation_samples = int(filtered_counts["validation"])
    result = {
        "status": "PASS", "mode": "VERIFY_ONLY",
        "config": str(config_path), "identities": identities,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "derived_status": manifest["status"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "map_type_filter": map_type_filter,
        "internal_test_accessed": False, "long_training_started": False,
    }
    if manifest["status"] != "COMPLETE_FROZEN":
        raise RuntimeError("formal derived dataset is not frozen/complete")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def finite_details(details):
    names = [
        "total_loss", "trajectory_loss", "score_loss", "smoothness_loss",
        "static_safety_loss", "guidance_loss", "dynamic_safety_loss",
        "ranking_loss", "safety_cvar_loss", "kinematic_loss",
        "preventive_safety_loss", "preventive_ranking_loss",
        "progress_safety_loss", "progress_ranking_loss",
        "stopping_distance_loss",
        "candidate_smooth_cost", "candidate_static_cost",
        "candidate_guidance_cost", "candidate_kinodynamic_cost",
        "candidate_preventive_cost", "candidate_progress_cost", "score_label",
        "candidate_stopping_distance_cost",
    ]
    if "dangerous_segment_loss" in details:
        names.extend((
            "dangerous_segment_loss", "candidate_dangerous_segment_cost",
        ))
    if "clearance_barrier_loss" in details:
        names.extend((
            "clearance_barrier_loss", "candidate_clearance_barrier_cost",
        ))
    if "safe_sector_coverage_loss" in details:
        names.extend((
            "safe_sector_coverage_loss",
            "candidate_safe_sector_coverage_cost",
        ))
    if "candidate_score_target_cost" in details:
        names.extend((
            "candidate_proposal_total_cost", "candidate_score_target_cost",
        ))
    return {
        name: bool(torch.isfinite(details[name]).all())
        for name in names
    }


def batch_identity(batch):
    return {
        "sample_id": list(batch["sample_id"]),
        "map_id": batch["map_id"].detach().cpu().tolist(),
        "depth_min": float(batch["depth"].amin()),
        "depth_max": float(batch["depth"].amax()),
        "observation_min": float(batch["observation"].amin()),
        "observation_max": float(batch["observation"].amax()),
    }


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    optimizer_name = str(optimizer_config["name"])
    optimizer_class = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    default_lr = float(optimizer_config["learning_rate"])
    backbone_lr = optimizer_config.get("backbone_learning_rate")
    head_lr = optimizer_config.get("head_learning_rate")
    candidate_head_lr = optimizer_config.get("candidate_head_learning_rate")
    score_head_lr = optimizer_config.get("score_head_learning_rate")
    if (candidate_head_lr is None) != (score_head_lr is None):
        raise ValueError(
            "candidate_head_learning_rate and score_head_learning_rate "
            "must be specified together"
        )
    if candidate_head_lr is not None:
        if backbone_lr is None:
            raise ValueError("split-head optimizer requires backbone_learning_rate")
        if model.network.head_variant not in {"split", "independent"}:
            raise ValueError(
                "independent candidate/score LR requires a branched head"
            )
        backbone, candidate, score = [], [], []
        score_ids = {
            id(value) for value in model.network.dep_head.score_parameters()
        }
        for name, parameter in model.named_parameters():
            if name.startswith("network.image_backbone."):
                backbone.append(parameter)
            elif id(parameter) in score_ids:
                score.append(parameter)
            else:
                candidate.append(parameter)
        if not backbone or not candidate or not score:
            raise RuntimeError("split-head optimizer partition is empty")
        values = backbone + candidate + score
        if len({id(value) for value in values}) != len(values):
            raise RuntimeError("split-head optimizer partition has duplicates")
        return optimizer_class([
            {"params": backbone, "lr": float(backbone_lr),
             "group_name": "backbone"},
            {"params": candidate, "lr": float(candidate_head_lr),
             "group_name": "candidate_head"},
            {"params": score, "lr": float(score_head_lr),
             "group_name": "score_head"},
        ], lr=default_lr, weight_decay=float(optimizer_config["weight_decay"]))
    if (backbone_lr is None) != (head_lr is None):
        raise ValueError(
            "backbone_learning_rate and head_learning_rate must be specified together"
        )
    if backbone_lr is None:
        parameters = model.parameters()
    else:
        backbone, head = [], []
        for name, parameter in model.named_parameters():
            if name.startswith("network.image_backbone."):
                backbone.append(parameter)
            else:
                head.append(parameter)
        if not backbone or not head:
            raise RuntimeError("backbone/head optimizer partition is empty")
        if len({id(value) for value in backbone + head}) != len(backbone) + len(head):
            raise RuntimeError("optimizer parameter partition contains duplicates")
        parameters = [
            {"params": backbone, "lr": float(backbone_lr), "group_name": "backbone"},
            {"params": head, "lr": float(head_lr), "group_name": "head"},
        ]
    return optimizer_class(
        parameters, lr=default_lr,
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def optimizer_learning_rates(optimizer):
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def apply_score_only_warmup(optimizer, config, epoch, start_epoch, model=None):
    """Freeze proposal generation briefly without resetting scheduler state.

    This transition is applied exactly once on a fresh run (epoch zero) and
    once at the configured unfreeze boundary. Resume checkpoints retain their
    saved group rates everywhere else.
    """
    warmup_epochs = int(config["training"].get("score_only_warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError("score_only_warmup_epochs must be non-negative")
    groups = {str(group.get("group_name")): group for group in optimizer.param_groups}
    if warmup_epochs and not {"backbone", "candidate_head", "score_head"} <= set(groups):
        raise ValueError("score-only warmup requires split-head optimizer groups")
    transition = None
    if warmup_epochs and epoch < warmup_epochs:
        for name in ("backbone", "candidate_head"):
            groups[name]["lr"] = 0.0
            for parameter in groups[name]["params"]:
                parameter.requires_grad_(False)
        for parameter in groups["score_head"]["params"]:
            parameter.requires_grad_(True)
        groups["score_head"]["lr"] = float(
            config["optimizer"].get(
                "score_only_learning_rate",
                config["optimizer"]["score_head_learning_rate"],
            )
        )
        if model is not None:
            # ``model.train()`` runs before this function. Frozen BatchNorm
            # buffers must not drift, otherwise proposal outputs are not
            # actually fixed even though their parameters require no gradient.
            model.network.image_backbone.eval()
        if epoch == start_epoch:
            transition = (
                "score_only_warmup_start" if start_epoch == 0
                else "score_only_warmup_resume"
            )
    elif warmup_epochs and epoch == warmup_epochs:
        for name in ("backbone", "candidate_head", "score_head"):
            for parameter in groups[name]["params"]:
                parameter.requires_grad_(True)
        groups["backbone"]["lr"] = float(
            config["optimizer"]["backbone_learning_rate"]
        )
        groups["candidate_head"]["lr"] = float(
            config["optimizer"]["candidate_head_learning_rate"]
        )
        groups["score_head"]["lr"] = float(
            config["optimizer"]["score_head_learning_rate"]
        )
        transition = "joint_finetune_start"
    return transition


def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    name = str(scheduler_config["name"])
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["max_epochs"]),
            eta_min=float(scheduler_config["minimum_learning_rate"]),
        )
    if name == "ReduceLROnPlateau":
        minimum = scheduler_config["minimum_learning_rate"]
        if isinstance(minimum, dict):
            minimum = [
                float(minimum[str(group.get("group_name", f"group_{index}"))])
                for index, group in enumerate(optimizer.param_groups)
            ]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            threshold=float(scheduler_config.get("threshold", 0.0)),
            threshold_mode="abs",
            min_lr=minimum,
        )
    raise ValueError(f"unsupported scheduler: {name}")


def map_type_contract(derived, dataset):
    authority = json.load(open(
        derived / "manifests" / "map_authority.json", encoding="utf-8"
    ))["maps"]
    by_map_id = {int(row["map_id"]): str(row["map_type"]) for row in authority}
    missing = sorted(set(map(int, dataset.arrays["map_id"])) - set(by_map_id))
    if missing:
        raise RuntimeError(f"map type authority missing map ids: {missing}")
    return by_map_id


def projected_validation_contract_v3(validation_means, validation_config):
    """Return the deliberately small V4.2.9 offline readiness contract.

    Candidate availability answers whether the projected generator can offer
    at least one visible and physically stoppable action. Conditional score
    error is evaluated only on those frames. Closed-loop arrival, collision,
    and path efficiency remain external evidence and are never disguised as
    additional offline booleans here.
    """
    gate = validation_config["offline_gate"]
    availability = float(validation_means["projected_candidate_availability"])
    numerator = float(
        validation_means["projected_conditional_selection_error_numerator"]
    )
    conditional_error = numerator / max(availability, 1.0e-12)
    layers = {
        "candidate_capability": {
            "value": availability,
            "minimum": float(gate["candidate_availability_rate_min"]),
            "passed": availability >= float(
                gate["candidate_availability_rate_min"]
            ),
        },
        "score_selection": {
            "value": conditional_error,
            "maximum": float(gate["conditional_selection_error_rate_max"]),
            "passed": conditional_error <= float(
                gate["conditional_selection_error_rate_max"]
            ),
        },
    }
    # This assertion is intentional: V4.2.9 must not silently grow another
    # collection of overlapping readiness gates.
    if tuple(layers) != ("candidate_capability", "score_selection"):
        raise AssertionError("V4.2.9 offline Gate proliferation detected")
    return layers, conditional_error, all(
        value["passed"] for value in layers.values()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v1.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--authorized", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_only:
        verify(args.config)
        return
    if not args.authorized:
        raise SystemExit("formal training requires the authorized shell entry")
    config_path, config, derived, manifest, identities = load_contract(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("formal H5 training requires CUDA")
    seed = int(config["training"]["seed"])
    seed_all(seed)
    device = torch.device("cuda:0")
    train_data = StaticYOPODatasetV1(
        derived, "train", mmap_cache_size=config["loader"]["mmap_cache_size"]
    )
    valid_data = StaticYOPODatasetV1(
        derived, "validation", mmap_cache_size=config["loader"]["mmap_cache_size"]
    )
    type_by_map_id = map_type_contract(derived, train_data)
    map_type_filter = config["loader"].get("map_type_filter")
    if map_type_filter is not None:
        train_data = StaticYOPOMapTypeSubsetV1(
            train_data, type_by_map_id, map_type_filter
        )
        valid_data = StaticYOPOMapTypeSubsetV1(
            valid_data, type_by_map_id, map_type_filter
        )
    observation_contract = config.get("observation", {}).get(
        "contract", "frozen_dataset_observation"
    )
    if observation_contract == "route_a_wide_state_v1":
        train_data = StaticYOPOWideStateDatasetV1(train_data, seed=seed)
        valid_data = StaticYOPOWideStateDatasetV1(valid_data, seed=seed + 1)
    elif observation_contract == "route_a_original_yopo_state_v4_4":
        train_data = StaticYOPOOriginalStateDatasetV44(train_data, seed=seed)
        valid_data = StaticYOPOOriginalStateDatasetV44(
            valid_data, seed=seed + 1
        )
    elif observation_contract == "route_a_recovery_state_v2":
        train_data = StaticYOPORecoveryStateDatasetV2(train_data, seed=seed)
        valid_data = StaticYOPORecoveryStateDatasetV2(valid_data, seed=seed + 1)
    elif observation_contract == "route_a_recovery_state_v4_8":
        observation_seed = int(config.get("observation", {}).get("seed", seed))
        train_data = StaticYOPORecoveryStateDatasetV48(
            train_data, seed=observation_seed,
        )
        valid_data = StaticYOPORecoveryStateDatasetV48(
            valid_data, seed=observation_seed + 1,
        )
    elif observation_contract != "frozen_dataset_observation":
        raise ValueError(
            f"unsupported observation contract: {observation_contract}"
        )
    type_by_map_id = map_type_contract(derived, train_data)
    balance_strategy = config["loader"].get("sampling_strategy", "uniform")
    if balance_strategy not in {"uniform", "map_type_balanced"}:
        raise ValueError(f"unsupported sampling strategy: {balance_strategy}")
    train_map_types = None
    if balance_strategy == "map_type_balanced":
        train_map_types = [
            type_by_map_id[int(map_id)]
            for map_id in train_data.arrays["map_id"]
        ]
    train_loader, train_sampler = make_static_yopo_loader_v1(
        train_data, config["training"]["batch_size"], seed,
        config["loader"]["num_workers"], True, config["loader"]["prefetch_factor"],
        config["loader"]["pin_memory"], True,
        balance_map_types=train_map_types,
    )
    valid_loader, valid_sampler = make_static_yopo_loader_v1(
        valid_data, config["training"]["batch_size"], seed + 1,
        config["loader"]["num_workers"], False, config["loader"]["prefetch_factor"],
        config["loader"]["pin_memory"], False,
    )
    model = MixedSceneStaticYOPOV1(
        config["model"]["initial_checkpoint"],
        head_variant=config["model"].get("head_variant", "unified"),
        allow_unified_to_split=bool(
            config["model"].get("allow_unified_to_split", False)
        ),
    ).to(device)
    objective = MixedSceneStaticYOPOObjectiveV1(
        derived / "map_catalog.yaml",
        local_goal_horizon_m=config.get("objective", {}).get(
            "local_goal_horizon_m"
        ),
        safety_first_config=config.get("safety_first"),
        kinodynamic_config=config.get("kinodynamic_v2"),
        preventive_safety_config=config.get("preventive_safety"),
        progress_safety_config=config.get("progress_safety"),
        feasibility_score_config=config.get("feasibility_score"),
        goal_progress_config=config.get("goal_progress_v2"),
        projected_score_config=config.get("projected_score_v3"),
        simple_yopo_v4_3_config=config.get("simple_yopo_v4_3"),
        static_yopo_v4_4_config=config.get("static_yopo_v4_4"),
        static_yopo_v4_5_config=config.get("static_yopo_v4_5"),
        static_yopo_v4_5_1_config=config.get("static_yopo_v4_5_1"),
        static_yopo_v4_5_2_config=config.get("static_yopo_v4_5_2"),
        static_yopo_v4_5_3_config=config.get("static_yopo_v4_5_3"),
        static_yopo_v4_5_4_config=config.get("static_yopo_v4_5_4"),
        static_yopo_v4_5_5_config=config.get("static_yopo_v4_5_5"),
        static_yopo_v4_5_6_config=config.get("static_yopo_v4_5_6"),
        static_yopo_v4_5_8_config=config.get("static_yopo_v4_5_8"),
        static_yopo_v4_5_9_config=config.get("static_yopo_v4_5_9"),
        static_yopo_v4_5_10_config=config.get("static_yopo_v4_5_10"),
        static_yopo_v4_8_config=config.get("static_yopo_v4_8"),
    ).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"]["amp"]))
    start_epoch, global_step, best = 0, 0, float("inf")
    if args.resume:
        payload = load_static_yopo_checkpoint_v1(
            args.resume, model, optimizer, scheduler, scaler, identities,
            train_sampler, map_location=device,
        )
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best = float(payload["best_validation_metric"])
        run_root = Path(args.resume).resolve().parents[1]
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        run_root = Path(config["output_root"]) / run_id
        if run_root.exists():
            raise FileExistsError(run_root)
        (run_root / "checkpoints").mkdir(parents=True)
    lock = run_root / "training.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise RuntimeError(f"active training lock exists: {lock}")
    write_json(run_root / "dataset_identity.json", identities)
    write_json(run_root / "environment.json", {
        "hostname": socket.gethostname(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "device": torch.cuda.get_device_name(0),
    })
    write_json(run_root / "run_state.json", {
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "long_training_started": not args.dry_run,
    })
    metrics_path = run_root / "metrics.jsonl"
    events_path = run_root / config["logging"]["batch_event_log"]
    nonfinite_path = run_root / config["logging"]["nonfinite_event_log"]
    maximum_epochs = 1 if args.dry_run else int(config["training"]["max_epochs"])
    maximum_train_batches = 8 if args.dry_run else None
    maximum_valid_batches = 1 if args.dry_run else None
    amp = bool(config["numerics"]["cnn_amp"])
    v45_enabled = bool(config.get("static_yopo_v4_5", {}).get("enabled"))
    v451_enabled = bool(config.get("static_yopo_v4_5_1", {}).get("enabled"))
    v452_enabled = bool(config.get("static_yopo_v4_5_2", {}).get("enabled"))
    v453_enabled = bool(config.get("static_yopo_v4_5_3", {}).get("enabled"))
    v454_enabled = bool(config.get("static_yopo_v4_5_4", {}).get("enabled"))
    v455_enabled = bool(config.get("static_yopo_v4_5_5", {}).get("enabled"))
    v456_enabled = bool(config.get("static_yopo_v4_5_6", {}).get("enabled"))
    v458_enabled = bool(config.get("static_yopo_v4_5_8", {}).get("enabled"))
    v459_enabled = bool(config.get("static_yopo_v4_5_9", {}).get("enabled"))
    v4510_enabled = bool(config.get("static_yopo_v4_5_10", {}).get("enabled"))
    v48_enabled = bool(config.get("static_yopo_v4_8", {}).get("enabled"))
    v45_family_enabled = (
        v45_enabled or v451_enabled or v452_enabled or v453_enabled
        or v454_enabled or v455_enabled or v456_enabled or v458_enabled
        or v459_enabled or v4510_enabled or v48_enabled
    )
    progress_every = int(config["logging"]["progress_every_batches"])
    minimum_epoch = int(config["validation"]["minimum_epoch"])
    patience = int(config["validation"]["patience"])
    best_epoch = None
    epochs_without_improvement = 0
    stop_reason = "MAX_EPOCHS"
    consecutive_amp_overflows = 0
    total_amp_overflows = 0
    if args.resume and metrics_path.is_file():
        previous_rows = [
            json.loads(line) for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if not previous_rows:
            raise RuntimeError("resume metrics file is empty")
        last = previous_rows[-1]
        if int(last["epoch"]) != start_epoch - 1 \
                or int(last["global_step"]) != global_step:
            raise RuntimeError("resume checkpoint/metrics boundary mismatch")
        best_row = min(previous_rows, key=lambda row: row.get(
            "selection_metric", row["validation_total_static_loss"]
        ))
        best_epoch = int(best_row["epoch"])
        if abs(float(best_row.get(
            "selection_metric", best_row["validation_total_static_loss"]
        )) - best) > 1e-9:
            raise RuntimeError("resume checkpoint/best metric mismatch")
        epochs_without_improvement = int(last["epoch"]) - best_epoch
    try:
        for epoch in range(start_epoch, maximum_epochs):
            epoch_started = time.perf_counter()
            train_sampler.set_epoch(epoch)
            model.train()
            stage_transition = apply_score_only_warmup(
                optimizer, config, epoch, start_epoch, model=model
            )
            if stage_transition is not None:
                event = {
                    "event": "optimizer_stage_transition",
                    "epoch": epoch,
                    "stage": stage_transition,
                    "learning_rates": optimizer_learning_rates(optimizer),
                }
                append_jsonl(events_path, event)
                print(json.dumps(event, sort_keys=True), flush=True)
            train_sums = {
                "total_loss": 0.0, "trajectory_loss": 0.0, "score_loss": 0.0,
                "smoothness_loss": 0.0, "static_safety_loss": 0.0,
                "guidance_loss": 0.0, "ranking_loss": 0.0,
                "safety_cvar_loss": 0.0, "kinematic_loss": 0.0,
                "preventive_safety_loss": 0.0,
                "preventive_ranking_loss": 0.0,
                "progress_safety_loss": 0.0,
                "progress_ranking_loss": 0.0,
                "stopping_distance_loss": 0.0,
            }
            if v45_family_enabled:
                train_sums["dangerous_segment_loss"] = 0.0
            if v4510_enabled or v48_enabled:
                train_sums["static_time_mean_cost"] = 0.0
                train_sums["static_worst_five_cost"] = 0.0
            if v48_enabled:
                train_sums["safe_sector_coverage_loss"] = 0.0
            if v453_enabled or v454_enabled or v456_enabled:
                train_sums["clearance_barrier_loss"] = 0.0
            if v456_enabled:
                train_sums["clearance_pairwise_ranking_loss"] = 0.0
            print(json.dumps({
                "event": "epoch_start", "epoch": epoch,
                "epochs_total": maximum_epochs, "train_batches": len(train_loader),
                "validation_batches": len(valid_loader),
                "learning_rate": max(optimizer_learning_rates(optimizer).values()),
                "learning_rates": optimizer_learning_rates(optimizer),
            }), flush=True)
            for batch_no, batch in enumerate(train_loader):
                batch = {key: (value.to(device, non_blocking=True)
                               if torch.is_tensor(value) else value)
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    details = objective(model, batch)
                component_finite = finite_details(details)
                if not all(component_finite.values()):
                    event = {
                        "event": "nonfinite_loss", "epoch": epoch,
                        "batch": batch_no, "global_step": global_step,
                        "finite_components": component_finite,
                        **batch_identity(batch),
                    }
                    append_jsonl(nonfinite_path, event)
                    write_json(run_root / "run_state.json", {
                        "status": "FAILED_NONFINITE", "phase": "train",
                        "epoch": epoch, "batch": batch_no,
                        "long_training_started": not args.dry_run,
                    })
                    raise FloatingPointError(
                        f"nonfinite training loss at epoch={epoch} batch={batch_no}"
                    )
                scaler.scale(details["total_loss"]).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config["training"]["gradient_clip_norm"]),
                    error_if_nonfinite=False,
                )
                gradient_finite = bool(torch.isfinite(gradient_norm))
                optimizer_step_applied = gradient_finite
                scale_before = float(scaler.get_scale())
                if not gradient_finite and not amp:
                    event = {
                        "event": "nonfinite_gradient", "epoch": epoch,
                        "batch": batch_no, "global_step": global_step,
                        "gradient_norm": str(float(gradient_norm)),
                        **batch_identity(batch),
                    }
                    append_jsonl(nonfinite_path, event)
                    write_json(run_root / "run_state.json", {
                        "status": "FAILED_NONFINITE", "phase": "backward",
                        "epoch": epoch, "batch": batch_no,
                        "long_training_started": not args.dry_run,
                    })
                    raise FloatingPointError(
                        f"nonfinite gradient at epoch={epoch} batch={batch_no}"
                    )
                if not gradient_finite:
                    consecutive_amp_overflows += 1
                    total_amp_overflows += 1
                    event = {
                        "event": "amp_gradient_overflow", "epoch": epoch,
                        "batch": batch_no, "global_step": global_step,
                        "scale_before": scale_before,
                        "consecutive_amp_overflows": consecutive_amp_overflows,
                        **batch_identity(batch),
                    }
                    append_jsonl(events_path, event)
                    print(json.dumps({
                        "event": event["event"], "epoch": epoch,
                        "batch": batch_no, "global_step": global_step,
                        "scale_before": scale_before,
                        "consecutive_amp_overflows": consecutive_amp_overflows,
                        "sample_count": len(batch["sample_id"]),
                        "details_written_to": str(events_path),
                    }, sort_keys=True), flush=True)
                    if consecutive_amp_overflows > int(
                        config["numerics"]["max_consecutive_amp_overflows"]
                    ):
                        append_jsonl(nonfinite_path, event)
                        raise FloatingPointError(
                            "persistent AMP gradient overflow exceeded contract"
                        )
                else:
                    consecutive_amp_overflows = 0
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if optimizer_step_applied:
                    global_step += 1
                for name in train_sums:
                    train_sums[name] += float(details[name].detach())
                completed = batch_no + 1
                if completed == 1 or completed % progress_every == 0 \
                        or completed == len(train_loader):
                    elapsed = time.perf_counter() - epoch_started
                    rate = completed / max(elapsed, 1e-9)
                    event = {
                        "event": "train_progress", "epoch": epoch,
                        "epochs_total": maximum_epochs, "batch": completed,
                        "batches_total": len(train_loader),
                        "percent": 100.0 * completed / len(train_loader),
                        "global_step": global_step,
                        "loss": float(details["total_loss"].detach()),
                        "running_loss": train_sums["total_loss"] / completed,
                        "trajectory_loss": float(details["trajectory_loss"].detach()),
                        "score_loss": float(details["score_loss"].detach()),
                        "gradient_norm": (
                            float(gradient_norm) if gradient_finite else None
                        ),
                        "optimizer_step_applied": optimizer_step_applied,
                        "amp_scale": scale_after,
                        "amp_overflows_total": total_amp_overflows,
                        "learning_rate": max(
                            optimizer_learning_rates(optimizer).values()
                        ),
                        "learning_rates": optimizer_learning_rates(optimizer),
                        "batches_per_second": rate,
                        "epoch_eta_seconds": (len(train_loader) - completed) / rate,
                        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                    }
                    append_jsonl(events_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                if maximum_train_batches and batch_no + 1 >= maximum_train_batches:
                    break
            model.eval()
            validation_sums = {
                "total_loss": 0.0, "trajectory_loss": 0.0, "score_loss": 0.0,
                "smoothness_loss": 0.0, "static_safety_loss": 0.0,
                "guidance_loss": 0.0,
                "ranking_loss": 0.0,
                "safety_cvar_loss": 0.0,
                "kinematic_loss": 0.0,
                "preventive_safety_loss": 0.0,
                "preventive_ranking_loss": 0.0,
                "progress_safety_loss": 0.0,
                "progress_ranking_loss": 0.0,
                "stopping_distance_loss": 0.0,
                "preventive_required_clearance": 0.0,
                "preventive_sample_weight": 0.0,
                "clear_candidate_count": 0.0,
                "selected_clearance": 0.0,
                "anticipatory_unsafe_selection": 0.0,
                "safe_progress_candidate_count": 0.0,
                "preferred_progress_candidate_count": 0.0,
                "insufficient_progress_selection": 0.0,
                "selected_goal_progress": 0.0,
                "selected_goal_alignment": 0.0,
                "reverse_selection": 0.0,
                "unsafe_selection": 0.0,
                "hardware_unsafe_selection": 0.0,
                "selected_trajectory_max_speed": 0.0,
                "selected_trajectory_max_acceleration": 0.0,
                "feasible_candidate_count": 0.0,
                "candidate_time_dilation_mean": 0.0,
                "selected_time_dilation": 0.0,
                "selected_normal_acceleration": 0.0,
                "selected_endpoint_distance": 0.0,
                "selected_endpoint_speed": 0.0,
                "hover_selection": 0.0,
                "route_goal_distance": 0.0,
                "objective_goal_distance": 0.0,
                "projected_candidate_availability": 0.0,
                "projected_safe_candidate_count": 0.0,
                "projected_conditional_selection_error_numerator": 0.0,
                "projected_selected_stopping_reserve": 0.0,
                "projected_selected_visible": 0.0,
                "projected_selection_regret": 0.0,
                "score_oracle_regret": 0.0,
                "stoppable_candidate_count": 0.0,
            }
            if v45_family_enabled:
                validation_sums.update({
                    "dangerous_segment_loss": 0.0,
                    "selected_vertical_displacement": 0.0,
                    "oracle_vertical_displacement": 0.0,
                    "selected_primitive_row": 0.0,
                    "oracle_primitive_row": 0.0,
                    "selected_vertical_primitive": 0.0,
                    "oracle_vertical_primitive": 0.0,
                    "selected_upward_primitive": 0.0,
                    "selected_level_primitive": 0.0,
                    "selected_downward_primitive": 0.0,
                    "oracle_upward_primitive": 0.0,
                    "oracle_level_primitive": 0.0,
                    "oracle_downward_primitive": 0.0,
                })
            if v4510_enabled or v48_enabled:
                validation_sums.update({
                    "static_time_mean_cost": 0.0,
                    "static_worst_five_cost": 0.0,
                })
            if v48_enabled:
                validation_sums.update({
                    "safe_sector_coverage_loss": 0.0,
                    "recovery_sample_fraction": 0.0,
                    "recovery_open_sector_soft_count": 0.0,
                    "recovery_mean_endpoint_distance": 0.0,
                    "recovery_max_endpoint_distance": 0.0,
                })
            if (v453_enabled or v454_enabled or v456_enabled
                    or v458_enabled or v459_enabled or v4510_enabled
                    or v48_enabled):
                validation_sums["clearance_barrier_loss"] = 0.0
            if (v456_enabled or v458_enabled or v459_enabled
                    or v4510_enabled or v48_enabled):
                validation_sums["clearance_pairwise_ranking_loss"] = 0.0
            if (v451_enabled or v452_enabled or v453_enabled or v454_enabled
                    or v455_enabled or v456_enabled or v458_enabled
                    or v459_enabled or v4510_enabled or v48_enabled):
                validation_sums.update({
                    "score_label_scale": 0.0,
                    "selected_absolute_vertical_displacement": 0.0,
                    "oracle_absolute_vertical_displacement": 0.0,
                    "selected_large_vertical_maneuver": 0.0,
                    "oracle_large_vertical_maneuver": 0.0,
                })
            if (v452_enabled or v453_enabled or v454_enabled or v455_enabled
                    or v456_enabled or v458_enabled or v459_enabled
                    or v4510_enabled or v48_enabled):
                validation_sums.update({
                    "speed_unsafe_selection": 0.0,
                    "acceleration_unsafe_selection": 0.0,
                    "collision_free_candidate_count": 0.0,
                    "hardware_feasible_candidate_count": 0.0,
                    "collision_free_candidate_available": 0.0,
                    "physical_feasible_candidate_available": 0.0,
                    "conditional_collision_selection_error": 0.0,
                    "conditional_physical_selection_error": 0.0,
                    "oracle_collision_unsafe": 0.0,
                    "oracle_speed_unsafe": 0.0,
                    "oracle_acceleration_unsafe": 0.0,
                    "oracle_hardware_unsafe": 0.0,
                    "oracle_physical_unsafe": 0.0,
                })
            if (v455_enabled or v456_enabled or v458_enabled
                    or v459_enabled or v4510_enabled or v48_enabled):
                validation_sums.update({
                    "coarse_false_safe_candidate_count": 0.0,
                    "dense_clearance_drop": 0.0,
                })
            validation_count = 0
            validation_by_type = {}
            with torch.inference_mode():
                for batch_no, batch in enumerate(valid_loader):
                    batch = {key: (value.to(device, non_blocking=True)
                                   if torch.is_tensor(value) else value)
                             for key, value in batch.items()}
                    with torch.amp.autocast("cuda", enabled=amp):
                        details = objective(model, batch)
                    component_finite = finite_details(details)
                    if not all(component_finite.values()):
                        event = {
                            "event": "nonfinite_validation_loss", "epoch": epoch,
                            "batch": batch_no, "finite_components": component_finite,
                            **batch_identity(batch),
                        }
                        append_jsonl(nonfinite_path, event)
                        raise FloatingPointError(
                            f"nonfinite validation loss at epoch={epoch} batch={batch_no}"
                        )
                    batch_size = int(batch["depth"].shape[0])
                    validation_count += batch_size
                    per_sample = {}
                    for name in validation_sums:
                        key = f"per_sample_{name}"
                        if key in details:
                            per_sample[name] = details[key].detach()
                        elif name == "score_oracle_regret":
                            per_sample[name] = details[
                                "per_sample_projected_selection_regret"
                            ].detach()
                        else:
                            raise KeyError(
                                f"objective omitted validation metric: {key}"
                            )
                    for name, values in per_sample.items():
                        validation_sums[name] += float(values.sum())
                    for map_type in sorted(set(
                        type_by_map_id[int(value)]
                        for value in batch["map_id"].detach().cpu().tolist()
                    )):
                        mask = torch.tensor([
                            type_by_map_id[int(value)] == map_type
                            for value in batch["map_id"].detach().cpu().tolist()
                        ], device=device, dtype=torch.bool)
                        target = validation_by_type.setdefault(map_type, {
                            "samples": 0,
                            **{name: 0.0 for name in validation_sums},
                            "score_top1_matches": 0,
                        })
                        target["samples"] += int(mask.sum())
                        for name, values in per_sample.items():
                            target[name] += float(values[mask].sum())
                        target["score_top1_matches"] += int(
                            details["per_sample_score_top1_match"][mask].sum()
                        )
                    completed = batch_no + 1
                    if completed == 1 or completed % progress_every == 0 \
                            or completed == len(valid_loader):
                        event = {
                            "event": "validation_progress", "epoch": epoch,
                            "batch": completed, "batches_total": len(valid_loader),
                            "percent": 100.0 * completed / len(valid_loader),
                            "running_loss": (
                                validation_sums["total_loss"] / validation_count
                            ),
                        }
                        append_jsonl(events_path, event)
                        print(json.dumps(event, sort_keys=True), flush=True)
                    if maximum_valid_batches and batch_no + 1 >= maximum_valid_batches:
                        break
            train_count = min(len(train_loader), maximum_train_batches or len(train_loader))
            train_means = {key: value / train_count for key, value in train_sums.items()}
            validation_means = {
                key: value / validation_count
                for key, value in validation_sums.items()
            }
            if v48_enabled:
                recovery_fraction = validation_means[
                    "recovery_sample_fraction"
                ]
                if recovery_fraction <= 0.0:
                    raise RuntimeError(
                        "V4.8 validation contains no recovery observations"
                    )
                for name in (
                    "recovery_open_sector_soft_count",
                    "recovery_mean_endpoint_distance",
                    "recovery_max_endpoint_distance",
                ):
                    # The objective emits zeros for ordinary samples.  Report
                    # these diagnostics conditionally over the deterministic
                    # recovery subset, while the loss itself remains averaged
                    # over the full 80/20 training population.
                    validation_means[name] /= recovery_fraction
            validation_map_type_metrics = {
                map_type: {
                    "samples": values["samples"],
                    **{
                        name: values[name] / values["samples"]
                        for name in validation_sums
                    },
                    "score_top1_label_agreement": (
                        values["score_top1_matches"] / values["samples"]
                    ),
                }
                for map_type, values in sorted(validation_by_type.items())
            }
            if v48_enabled:
                for values in validation_map_type_metrics.values():
                    recovery_fraction = values["recovery_sample_fraction"]
                    if recovery_fraction <= 0.0:
                        raise RuntimeError(
                            "V4.8 map-type validation contains no recovery "
                            "observations"
                        )
                    for name in (
                        "recovery_open_sector_soft_count",
                        "recovery_mean_endpoint_distance",
                        "recovery_max_endpoint_distance",
                    ):
                        values[name] /= recovery_fraction
            for values in validation_map_type_metrics.values():
                values["projected_conditional_selection_error_rate"] = (
                    values[
                        "projected_conditional_selection_error_numerator"
                    ] / max(
                        values["projected_candidate_availability"], 1.0e-12
                    )
                )
            macro_map_type_total = float(np.mean([
                values["total_loss"]
                for values in validation_map_type_metrics.values()
            ]))
            validation_contract = config["validation"].get(
                "contract_version", "legacy_selection_gate"
            )
            if validation_contract in {
                "route_a_v4_3_single_loss_v1",
                "route_a_v4_4_static_parity_v1",
                "route_a_v4_5_bounded_danger_v1",
                "route_a_v4_5_1_relative_kinematic_v1",
                "route_a_v4_5_2_calibrated_relative_kinematic_v1",
                "route_a_v4_5_3_continuous_clearance_two_stage_v1",
                "route_a_v4_5_4_independent_score_only_v1",
                "route_a_v4_5_5_dense_static_esdf_v1",
                "route_a_v4_5_6_continuous_score_safety_v1",
                "route_a_v4_5_7_high_safety_retimed_v1",
                "route_a_v4_5_8_localized_safety_retimed_v1",
                "route_a_v4_5_9_time_mean_localized_safety_v1",
                "route_a_v4_5_10_tail_aware_safety_v1",
                "route_a_v4_8_recovery_capacity_v1",
            }:
                # These contracts deliberately have no qualification lattice. The
                # scalar held-out loss selects best.pth; physical deployment
                # is evaluated later by closed-loop collision/arrival tests.
                conditional_selection_error = None
                validation_layers = None
                selection_gate = {"single_validation_loss": True}
                selection_gate_pass = True
            elif validation_contract == "route_a_v4_2_9_three_layer_v1":
                validation_layers, conditional_selection_error, \
                    selection_gate_pass = projected_validation_contract_v3(
                        validation_means, config["validation"]
                    )
                selection_gate = {
                    name: value["passed"]
                    for name, value in validation_layers.items()
                }
            else:
                conditional_selection_error = None
                validation_layers = None
                selection_gate_config = config["validation"].get(
                    "selection_gate", {}
                )
                selection_gate = {
                "selected_endpoint_distance_mean": (
                    validation_means["selected_endpoint_distance"]
                    >= float(selection_gate_config.get(
                        "selected_endpoint_distance_mean_min", 0.0
                    ))
                ),
                "selected_endpoint_speed_mean": (
                    validation_means["selected_endpoint_speed"]
                    >= float(selection_gate_config.get(
                        "selected_endpoint_speed_mean_min", 0.0
                    ))
                ),
                "hover_selection_rate": (
                    validation_means["hover_selection"]
                    <= float(selection_gate_config.get(
                        "hover_selection_rate_max", 1.0
                    ))
                ),
                "unsafe_selection_rate": (
                    validation_means["unsafe_selection"]
                    <= float(selection_gate_config.get(
                        "unsafe_selection_rate_max", 1.0
                    ))
                ),
                "hardware_unsafe_selection_rate": (
                    validation_means["hardware_unsafe_selection"]
                    <= float(selection_gate_config.get(
                        "hardware_unsafe_selection_rate_max", 1.0
                    ))
                ),
                "feasible_candidate_count_mean": (
                    validation_means["feasible_candidate_count"]
                    >= float(selection_gate_config.get(
                        "feasible_candidate_count_mean_min", 0.0
                    ))
                ),
                "selected_time_dilation_mean": (
                    validation_means["selected_time_dilation"]
                    <= float(selection_gate_config.get(
                        "selected_time_dilation_mean_max", float("inf")
                    ))
                ),
                "anticipatory_unsafe_selection_rate": (
                    validation_means["anticipatory_unsafe_selection"]
                    <= float(selection_gate_config.get(
                        "anticipatory_unsafe_selection_rate_max", 1.0
                    ))
                ),
                "selected_clearance_mean": (
                    validation_means["selected_clearance"]
                    >= float(selection_gate_config.get(
                        "selected_clearance_mean_min", 0.0
                    ))
                ),
                "clear_candidate_count_mean": (
                    validation_means["clear_candidate_count"]
                    >= float(selection_gate_config.get(
                        "clear_candidate_count_mean_min", 0.0
                    ))
                ),
                "safe_progress_candidate_count_mean": (
                    validation_means["safe_progress_candidate_count"]
                    >= float(selection_gate_config.get(
                        "safe_progress_candidate_count_mean_min", 0.0
                    ))
                ),
                "preferred_progress_candidate_count_mean": (
                    validation_means["preferred_progress_candidate_count"]
                    >= float(selection_gate_config.get(
                        "preferred_progress_candidate_count_mean_min", 0.0
                    ))
                ),
                "insufficient_progress_selection_rate": (
                    validation_means["insufficient_progress_selection"]
                    <= float(selection_gate_config.get(
                        "insufficient_progress_selection_rate_max", 1.0
                    ))
                ),
                "selected_goal_progress_mean": (
                    validation_means["selected_goal_progress"]
                    >= float(selection_gate_config.get(
                        "selected_goal_progress_mean_min", float("-inf")
                    ))
                ),
                "selected_goal_alignment_mean": (
                    validation_means["selected_goal_alignment"]
                    >= float(selection_gate_config.get(
                        "selected_goal_alignment_mean_min", float("-inf")
                    ))
                ),
                "reverse_selection_rate": (
                    validation_means["reverse_selection"]
                    <= float(selection_gate_config.get(
                        "reverse_selection_rate_max", 1.0
                    ))
                ),
                }
                selection_gate_pass = all(selection_gate.values())
            primary_metric = config["validation"]["primary_metric"]
            if primary_metric == "total_static_loss":
                metric = validation_means["total_loss"]
            elif primary_metric == "macro_map_type_total_static_loss":
                metric = macro_map_type_total
            elif primary_metric == "anti_hover_feasible_macro_loss":
                # Keep JSON/checkpoint state finite while ensuring that any
                # gate-passing epoch outranks every collapsed/hovering epoch.
                metric = macro_map_type_total + (
                    0.0 if selection_gate_pass else 1_000_000.0
                )
            elif primary_metric == "hard_safety_weighted_macro_loss":
                weights = config["validation"].get(
                    "hard_safety_metric_weights", {}
                )
                safety_weighted_metric = (
                    macro_map_type_total
                    + float(weights.get("unsafe_selection", 20.0))
                    * validation_means["unsafe_selection"]
                    + float(weights.get("hardware_unsafe_selection", 10.0))
                    * validation_means["hardware_unsafe_selection"]
                    + float(weights.get("hover_selection", 2.0))
                    * validation_means["hover_selection"]
                    + float(weights.get("reverse_selection", 0.0))
                    * validation_means["reverse_selection"]
                    + float(weights.get("insufficient_progress_selection", 0.0))
                    * validation_means["insufficient_progress_selection"]
                )
                metric = safety_weighted_metric + (
                    0.0 if selection_gate_pass else 1_000_000.0
                )
            elif primary_metric == "projected_contract_macro_loss":
                if validation_layers is None:
                    raise RuntimeError(
                        "projected metric requires the V4.2.9 validation contract"
                    )
                weights = config["validation"].get("metric_weights", {})
                metric = (
                    macro_map_type_total
                    + float(weights.get("candidate_unavailability", 10.0))
                    * (1.0 - validation_means[
                        "projected_candidate_availability"
                    ])
                    + float(weights.get("conditional_selection_error", 20.0))
                    * conditional_selection_error
                    + (0.0 if selection_gate_pass else 1_000_000.0)
                )
            else:
                raise ValueError(
                    f"unsupported validation primary metric: {primary_metric}"
                )
            improved = metric < best
            if improved:
                best = metric
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            warmup_epochs = int(config["training"].get(
                "score_only_warmup_epochs", 0
            ))
            if epoch < warmup_epochs:
                pass
            elif isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(metric)
            else:
                scheduler.step()
            row = {
                "epoch": epoch, "global_step": global_step,
                "train_total_loss": train_means["total_loss"],
                "train_components": train_means,
                "validation_total_static_loss": metric,
                "validation_components": validation_means,
                "validation_by_map_type": validation_map_type_metrics,
                "validation_macro_map_type_total_static_loss": macro_map_type_total,
                "selection_metric_name": primary_metric,
                "selection_metric": metric,
                "selection_gate_pass": selection_gate_pass,
                "selection_gate": selection_gate,
                "validation_layers": validation_layers,
                "projected_conditional_selection_error_rate": (
                    conditional_selection_error
                ),
                "learning_rate": max(optimizer_learning_rates(optimizer).values()),
                "learning_rates": optimizer_learning_rates(optimizer),
                "best_validation_metric": best, "best_epoch": best_epoch,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "amp_overflows_total": total_amp_overflows,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            append_jsonl(metrics_path, row)
            if not args.dry_run:
                epoch_checkpoint = (
                    run_root / "checkpoints" / f"epoch_{epoch:03d}.pth"
                )
                save_static_yopo_checkpoint_v1(
                    epoch_checkpoint,
                    model, optimizer, scheduler, scaler, epoch, global_step, best,
                    identities, train_sampler,
                    environment={"torch": torch.__version__,
                                 "cuda": torch.cuda.get_device_name(0)},
                    git_commit=subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                        capture_output=True,
                    ).stdout.strip() or "unknown",
                )
                if improved:
                    destination = (
                        run_root / "checkpoints/best.pth"
                        if selection_gate_pass
                        else run_root / "checkpoints/best_unqualified.pth"
                    )
                    atomic_copy(epoch_checkpoint, destination)
            write_json(run_root / "run_state.json", {
                "status": "RUNNING", "epoch_completed": epoch,
                "global_step": global_step, "best_epoch": best_epoch,
                "best_validation_metric": best,
                "epochs_without_improvement": epochs_without_improvement,
                "long_training_started": not args.dry_run,
            })
            print(json.dumps({"event": "epoch_end", **row}, sort_keys=True), flush=True)
            if not args.dry_run and epoch >= minimum_epoch \
                    and epochs_without_improvement >= patience:
                stop_reason = "EARLY_STOPPING_PATIENCE"
                print(json.dumps({
                    "event": "early_stop", "epoch": epoch,
                    "best_epoch": best_epoch, "best_validation_metric": best,
                    "patience": patience,
                }), flush=True)
                break
        metric_eligible = bool(best < 1_000_000.0)
        completion_contract = training_contract_classification(config)
        three_layer_contract = completion_contract["three_layer"]
        ungated_static_contract = completion_contract["ungated_static"]
        diagnostic_run = completion_contract["diagnostic"]
        # Ungated objectives and bounded diagnostics never become production
        # qualified from a scalar validation loss alone.
        training_gate_passed = (
            None if (ungated_static_contract or diagnostic_run)
            else metric_eligible
        )
        closed_loop_pending = three_layer_contract or ungated_static_contract
        if args.dry_run:
            completion_status = "DRY_RUN_PASS"
        elif diagnostic_run:
            completion_status = "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION"
        elif three_layer_contract and training_gate_passed:
            completion_status = (
                "TRAINING_COMPLETE_OFFLINE_QUALIFIED_PENDING_CLOSED_LOOP"
            )
        elif ungated_static_contract and metric_eligible:
            completion_status = "TRAINING_COMPLETE_PENDING_CLOSED_LOOP"
        elif training_gate_passed:
            completion_status = "TRAINING_COMPLETE"
        else:
            completion_status = "TRAINING_COMPLETE_UNQUALIFIED"
        write_json(run_root / "training_complete.json", {
            "status": completion_status,
            "global_step": global_step, "best_validation_metric": best,
            "best_epoch": best_epoch, "stop_reason": stop_reason,
            "training_gate_passed": training_gate_passed,
            "offline_gate_passed": (
                None if (ungated_static_contract or diagnostic_run)
                else metric_eligible
            ),
            "closed_loop_gate_passed": (
                False if closed_loop_pending and not diagnostic_run else None
            ),
            "production_qualified": (
                False if (closed_loop_pending or diagnostic_run)
                else training_gate_passed
            ),
            "post_training_h5_started": False, "production_activated": False,
        })
        write_json(run_root / "run_state.json", {
            "status": (
                "DRY_RUN_COMPLETE" if args.dry_run else completion_status
            ),
            "global_step": global_step, "best_epoch": best_epoch,
            "best_validation_metric": best, "stop_reason": stop_reason,
            "training_gate_passed": training_gate_passed,
            "offline_gate_passed": (
                None if (ungated_static_contract or diagnostic_run)
                else metric_eligible
            ),
            "closed_loop_gate_passed": (
                False if closed_loop_pending and not diagnostic_run else None
            ),
            "production_qualified": (
                False if (closed_loop_pending or diagnostic_run)
                else training_gate_passed
            ),
            "long_training_started": not args.dry_run,
        })
    except Exception as exc:
        write_json(run_root / "failure.json", {"status": "FAIL", "error": repr(exc)})
        write_json(run_root / "run_state.json", {
            "status": "FAILED",
            "error": repr(exc),
            "global_step": global_step,
            "long_training_started": not args.dry_run,
        })
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
