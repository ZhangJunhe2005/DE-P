#!/usr/bin/env python3
"""Prepare the bounded V4.7 fine-tune from the completed V4.6 best.

V4.7 changes the map distribution, not the learning objective.  This entry
therefore preserves the V4.6 loss/validation contract, uses balanced sampling
over the same four map types, and only permits a small, bounded fine-tune.
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


PARENT = ROOT / "configs/route_a_v4_6_four_scene_finetune.yaml"
PARENT_RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_6_four_scene_finetune"
SOURCE = ROOT / "data/route_a_v4_7_raw_static"
DERIVED = ROOT / "data/route_a_v4_7_static_yopo"
OUTPUT = ROOT / "configs/route_a_v4_7_original_density_finetune.yaml"
RUN_ROOT = ROOT / "runs/route_a_static_yopo_v4_7_original_density_finetune"

EXPECTED_TYPES = {"cave", "forest", "pillar", "wall"}
SOURCE_VERSION = "route_a_v4_7_raw_static_v1"
DERIVED_VERSION = "route_a_v4_7_static_yopo"
PARENT_COMPLETION_STATUS = "TRAINING_COMPLETE_PENDING_CLOSED_LOOP"


def latest_parent_checkpoint(run_root: Path = PARENT_RUN_ROOT):
    """Return the newest identity-complete V4.6 best checkpoint."""
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
            "no completed V4.6 best checkpoint found; finish V4.6 first"
        )
    return candidates[-1]


def validate_dataset_identity(
    source_root: Path = SOURCE,
    derived_root: Path = DERIVED,
):
    """Validate the frozen V4.7 identities without reading sample payloads."""
    source_manifest = source_root / "manifests/dataset_manifest.json"
    derived_manifest = derived_root / "manifests/dataset_manifest.json"
    if not source_manifest.is_file() or not derived_manifest.is_file():
        raise FileNotFoundError(
            "generate and validate the V4.7 dataset before training"
        )
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


def build_config(parent, parent_run, checkpoint, completion,
                 source_manifest, derived_manifest):
    """Build a V4.7 config while leaving the inherited loss untouched."""
    config = copy.deepcopy(parent)
    config["contract_version"] = (
        "route_a_static_yopo_training_v4_7_original_density_finetune_v1"
    )
    config["training_implementation_hash"] = training_implementation_hash()
    config["parent_v4_6_config_hash"] = sha256_file(PARENT)
    config["parent_v4_6_run"] = str(parent_run)
    config["parent_v4_6_best_epoch"] = int(completion["best_epoch"])
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

    # A small distribution-adaptation step.  Proposal learning remains an
    # order of magnitude slower than score learning and the backbone slower
    # again, limiting drift from the closed-loop-tested V4.6 parent.
    config["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1.0e-5,
        "backbone_learning_rate": 5.0e-8,
        "candidate_head_learning_rate": 5.0e-7,
        "score_head_learning_rate": 1.0e-5,
        "score_only_learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    config["scheduler"] = {
        "name": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 2,
        "threshold": 1.0e-4,
        "minimum_learning_rate": {
            "backbone": 1.25e-8,
            "candidate_head": 1.25e-7,
            "score_head": 1.25e-6,
        },
    }
    config["training"].update({
        "max_epochs": 20,
        "seed": 84710,
        "score_only_warmup_epochs": 0,
    })
    config["loader"]["sampling_strategy"] = "map_type_balanced"
    config["loader"].pop("map_type_filter", None)
    config["validation"].update({
        # Keep the existing scalar and loss semantics.  This is early stopping,
        # not a new safety qualification Gate.
        "contract_version": "route_a_v4_5_10_tail_aware_safety_v1",
        "primary_metric": "macro_map_type_total_static_loss",
        "minimum_epoch": 6,
        "patience": 6,
    })
    config.pop("experiment_role", None)
    config.pop("route_a_v4_6", None)
    config["route_a_v4_7"] = {
        "enabled": True,
        "supported_map_types": sorted(EXPECTED_TYPES),
        "excluded_map_types": ["room"],
        "room_sample_count": 0,
        "map_type_balanced": True,
        "loss_contract_changed": False,
        "qualification_gate_count": 0,
        "maximum_epochs": 20,
    }
    config["implementation_hotfixes"] = list(
        config.get("implementation_hotfixes", [])
    ) + [
        "route_a_v4_7_initialize_from_v4_6_best_v1",
        "route_a_v4_7_four_type_balanced_sampling_v1",
        "route_a_v4_7_preserve_existing_loss_contract_v1",
        "route_a_v4_7_twenty_epoch_low_lr_bounded_finetune_v1",
        "route_a_v4_7_no_new_qualification_gate_v1",
    ]
    return config


def main():
    source_manifest, derived_manifest, derived = validate_dataset_identity()
    parent_run, checkpoint, completion = latest_parent_checkpoint()
    yaml = YAML()
    parent = yaml.load(PARENT)
    config = build_config(
        parent, parent_run, checkpoint, completion,
        source_manifest, derived_manifest,
    )
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as stream:
            existing = yaml.load(stream)
        if existing != config and any(RUN_ROOT.glob("*")):
            raise RuntimeError(
                "refusing to change V4.7 config after training started"
            )
    with OUTPUT.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)
    print(json.dumps({
        "status": "PASS",
        "config": str(OUTPUT),
        "config_sha256": sha256_file(OUTPUT),
        "source_dataset": str(SOURCE),
        "derived_dataset": str(DERIVED),
        "samples": derived["split_counts"],
        "map_types": sorted(EXPECTED_TYPES),
        "initial_checkpoint": str(checkpoint),
        "parent_best_epoch": int(completion["best_epoch"]),
        "maximum_epochs": 20,
        "earliest_early_stop_epoch": 6,
        "early_stop_patience": 6,
        "learning_rates": {
            "backbone": 5.0e-8,
            "candidate_head": 5.0e-7,
            "score_head": 1.0e-5,
        },
        "map_type_balanced": True,
        "loss_contract_changed": False,
        "qualification_gate_count": 0,
        "training_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
