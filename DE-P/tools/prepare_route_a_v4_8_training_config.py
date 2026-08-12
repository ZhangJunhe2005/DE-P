#!/usr/bin/env python3
"""Prepare the bounded V4.8 recovery-capacity five-epoch shakedown.

V4.8 deliberately reuses the frozen V4.7 dataset and starts from the newest
completed V4.7 best checkpoint.  It changes only the deterministic observation
mixture and the recovery-capacity objective; it does not create a qualification
Gate or authorize training by itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.static_yopo_manifest_v1 import sha256_file
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


PARENT = ROOT / "configs/route_a_v4_7_original_density_finetune.yaml"
PARENT_RUN_ROOT = (
    ROOT / "runs/route_a_static_yopo_v4_7_original_density_finetune"
)
SOURCE = ROOT / "data/route_a_v4_7_raw_static"
DERIVED = ROOT / "data/route_a_v4_7_static_yopo"
OUTPUT = ROOT / "configs/route_a_v4_8_recovery_capacity_shakedown.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_8_recovery_capacity_shakedown"

TRAINING_CONTRACT = (
    "route_a_static_yopo_training_v4_8_recovery_capacity_shakedown_v1"
)
VALIDATION_CONTRACT = "route_a_v4_8_recovery_capacity_v1"
OBSERVATION_CONTRACT = "route_a_recovery_state_v4_8"
EXPERIMENT_ROLE = "recovery_capacity_five_epoch_shakedown"
OBSERVATION_SEED = 84710
PARENT_COMPLETION_STATUS = "TRAINING_COMPLETE_PENDING_CLOSED_LOOP"
EXPECTED_TYPES = {"cave", "forest", "pillar", "wall"}
SOURCE_VERSION = "route_a_v4_7_raw_static_v1"
DERIVED_VERSION = "route_a_v4_7_static_yopo"


def latest_parent_checkpoint(run_root: Path = PARENT_RUN_ROOT):
    """Return the newest completed V4.7 run and its strict best checkpoint."""
    candidates = []
    for run in sorted(path for path in run_root.glob("*") if path.is_dir()):
        completion_path = run / "training_complete.json"
        checkpoint = run / "checkpoints/best.pth"
        if not (completion_path.is_file() and checkpoint.is_file()):
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != PARENT_COMPLETION_STATUS:
            continue
        candidates.append((run, checkpoint, completion))
    if not candidates:
        raise RuntimeError(
            "no completed V4.7 best checkpoint found; finish V4.7 first"
        )
    return candidates[-1]


def formal_training_started(run_root: Path = RUN_ROOT):
    """Distinguish an authorized formal run from a bounded dry-run."""
    for run in sorted(path for path in run_root.glob("*") if path.is_dir()):
        state_path = run / "run_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if bool(state.get("long_training_started")):
            return True
    return False


def validate_dataset_identity(
    source_root: Path = SOURCE,
    derived_root: Path = DERIVED,
):
    """Validate that V4.8 points at the existing frozen V4.7 dataset."""
    source_manifest = source_root / "manifests/dataset_manifest.json"
    derived_manifest = derived_root / "manifests/dataset_manifest.json"
    if not source_manifest.is_file() or not derived_manifest.is_file():
        raise FileNotFoundError("the frozen V4.7 dataset must exist before V4.8")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    derived = json.loads(derived_manifest.read_text(encoding="utf-8"))
    if source.get("dataset_version") != SOURCE_VERSION:
        raise RuntimeError("V4.7 raw dataset identity mismatch")
    if derived.get("dataset_version") != DERIVED_VERSION:
        raise RuntimeError("V4.7 derived dataset identity mismatch")
    if derived.get("status") != "COMPLETE_FROZEN":
        raise RuntimeError("V4.7 derived dataset is not frozen/complete")
    map_counts = derived.get("map_type_sample_counts", {})
    for split in ("train", "validation"):
        counts = map_counts.get(split, {})
        if set(counts) != EXPECTED_TYPES:
            raise RuntimeError(
                f"V4.7 {split} must contain exactly four supported map types"
            )
        if any(int(counts[name]) <= 0 for name in EXPECTED_TYPES):
            raise RuntimeError(f"V4.7 {split} contains an empty map type")
    return source_manifest, derived_manifest, derived


def build_config(
    parent,
    parent_run,
    checkpoint,
    completion,
    source_manifest,
    derived_manifest,
):
    """Build the V4.8 shakedown contract without mutating the V4.7 input."""
    parent_run = Path(parent_run).resolve()
    checkpoint = Path(checkpoint).resolve()
    expected_checkpoint = parent_run / "checkpoints/best.pth"
    if checkpoint != expected_checkpoint:
        raise ValueError("V4.8 must initialize from the V4.7 run's best.pth")
    config = copy.deepcopy(parent)
    config["contract_version"] = TRAINING_CONTRACT
    config["training_implementation_hash"] = training_implementation_hash()
    config["experiment_role"] = EXPERIMENT_ROLE
    config["parent_v4_7_config_hash"] = sha256_file(PARENT)
    config["parent_v4_7_run"] = str(parent_run)
    config["parent_v4_7_best_epoch"] = int(completion["best_epoch"])
    config["source_dataset_root"] = str(SOURCE)
    config["derived_dataset_root"] = str(DERIVED)
    config["source_dataset_manifest_hash"] = sha256_file(source_manifest)
    config["derived_dataset_manifest_hash"] = sha256_file(derived_manifest)
    config["output_root"] = str(RUN_ROOT)
    config["model"].update({
        "head_variant": "independent",
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "strict_load": True,
        "allow_unified_to_split": False,
    })
    config["observation"] = {
        "contract": OBSERVATION_CONTRACT,
        # V4.7's trainer used training.seed=84710 for the original-YOPO
        # sampler.  Preserve those ordinary observations byte-for-byte; the
        # V4.8 hash contract alone selects/replaces the 20% recovery subset.
        "seed": OBSERVATION_SEED,
    }
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 2.0e-6,
        "backbone_learning_rate": 5.0e-8,
        "candidate_head_learning_rate": 1.0e-6,
        "score_head_learning_rate": 2.0e-6,
        "score_only_learning_rate": 2.0e-6,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 2,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.25e-8,
            "candidate_head": 2.5e-7,
            "score_head": 5.0e-7,
        },
    }
    config["training"].update({
        "max_epochs": 5,
        "seed": 84801,
        "score_only_warmup_epochs": 0,
    })
    config["loader"]["sampling_strategy"] = "map_type_balanced"
    config["loader"].pop("map_type_filter", None)
    config["static_yopo_v4_5_10"]["enabled"] = False
    config["static_yopo_v4_8"] = {
        **copy.deepcopy(parent["static_yopo_v4_5_10"]),
        "enabled": True,
        "safe_sector_weight": 0.05,
        "safe_sector_ray_start_m": 0.5,
        "safe_sector_ray_end_m": 4.0,
        "safe_sector_ray_samples": 12,
        "safe_sector_openness_center_m": 0.45,
        "safe_sector_openness_softness_m": 0.10,
        "safe_sector_target_length_m": 3.0,
        "safe_sector_shortfall_softness_m": 0.35,
    }
    config["validation"].update({
        "contract_version": VALIDATION_CONTRACT,
        "minimum_epoch": 5,
        "patience": 5,
    })
    predefined = list(config["validation"].get("predefined_metrics", []))
    for name in (
        "safe_sector_coverage_loss",
        "recovery_sample_fraction",
        "recovery_open_sector_soft_count",
        "recovery_mean_endpoint_distance",
        "recovery_max_endpoint_distance",
    ):
        if name not in predefined:
            predefined.append(name)
    config["validation"]["predefined_metrics"] = predefined
    config.pop("route_a_v4_7", None)
    config["route_a_v4_8"] = {
        "enabled": True,
        "dataset_reused_from": "route_a_v4_7_static_yopo",
        "supported_map_types": sorted(EXPECTED_TYPES),
        "map_type_balanced": True,
        "maximum_epochs": 5,
        "qualification_gate_count": 0,
    }
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_8_initialize_from_latest_completed_v47_best_v1",
        "route_a_v4_8_reuse_frozen_v47_dataset_v1",
        "route_a_v4_8_recovery_observation_mixture_v1",
        "route_a_v4_8_safe_sector_coverage_v1",
        "route_a_v4_8_five_epoch_low_lr_shakedown_v1",
        "route_a_v4_8_no_new_qualification_gate_v1",
    ]
    return config


def main():
    source_manifest, derived_manifest, derived = validate_dataset_identity()
    parent_run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    parent = yaml.load(PARENT)
    config = build_config(
        parent,
        parent_run,
        checkpoint,
        completion,
        source_manifest,
        derived_manifest,
    )
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as stream:
            existing = yaml.load(stream)
        if existing != config and formal_training_started(RUN_ROOT):
            raise RuntimeError("refusing to change V4.8 config after training started")
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "dataset_reused_from": str(DERIVED),
        "samples": derived["split_counts"],
        "initial_checkpoint": str(checkpoint),
        "parent_best_epoch": int(completion["best_epoch"]),
        "epochs": 5,
        "observation_contract": OBSERVATION_CONTRACT,
        "observation_seed": OBSERVATION_SEED,
        "validation_contract": VALIDATION_CONTRACT,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
