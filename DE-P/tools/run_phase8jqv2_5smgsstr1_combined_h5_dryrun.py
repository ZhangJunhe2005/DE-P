#!/usr/bin/env python3
"""Bounded Route-A integration-entry dry-run; never a formal H5 claim."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_5smgsstr1_"
CONFIG = ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3.yaml"

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from policy.dynamic.deterministic_guard_strict_history_only_v1 import (
    DeterministicDynamicGuardInputV1,
    DeterministicDynamicGuardStrictHistoryOnlyV1,
)
from policy.static_yopo_training_v1 import MixedSceneStaticYOPOV1
from tools.train_mixed_static_yopo_v1 import load_contract


def write(name: str, value: object) -> None:
    path = REPORTS / f"{PREFIX}{name}.json"
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def must_reject(guard, **kwargs) -> bool:
    try:
        guard.evaluate(DeterministicDynamicGuardInputV1(
            "STRICT_MEASUREMENT", True, True, **kwargs
        ))
    except ValueError:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("only --dry-run is allowed in SMGSS-TR1")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required for combined-H5 entry dry-run")

    _, config, derived, manifest, identities = load_contract(CONFIG)
    h5 = YAML(typ="safe").load(
        ROOT / "configs/runtime_environment_contract_v1_candidate.yaml"
    )["experiments"]["H5"]
    historical = json.load(open(
        REPORTS / "phase8jqv2_4brir1_closed_loop_safety.json"
    ))
    if historical["dynamic_collision_proxy"] != 34:
        raise RuntimeError("BRIR1 historical blocker corpus was relabeled")

    dataset = StaticYOPODatasetV1(derived, "train")
    sample = dataset[0]
    model = MixedSceneStaticYOPOV1(config["model"]["initial_checkpoint"]).cuda().eval()
    with torch.inference_mode():
        endstate, score = model(
            sample["depth"].unsqueeze(0).cuda(),
            sample["observation"].unsqueeze(0).cuda(),
        )
    finite = bool(torch.isfinite(endstate).all() and torch.isfinite(score).all())

    guard = DeterministicDynamicGuardStrictHistoryOnlyV1()
    weak = guard.evaluate(DeterministicDynamicGuardInputV1(
        "WEAK", True, False
    ))
    strict = guard.evaluate(DeterministicDynamicGuardInputV1(
        "STRICT_MEASUREMENT", True, True
    ))
    history = guard.evaluate(DeterministicDynamicGuardInputV1(
        "HISTORY_BACKED_STATE", True, False, history_legal=True
    ))
    forbidden = {
        "runtime_gt": must_reject(guard, runtime_gt_present=True),
        "owner_map": must_reject(guard, owner_map_present=True),
        "future_actor_state": must_reject(guard, future_actor_state_present=True),
    }
    passed = (
        finite and not weak["active_risk"] and strict["active_risk"]
        and history["active_risk"] and all(forbidden.values())
    )
    result = {
        "status": "ENTRY_READY_NOT_RUN_FORMAL" if passed else "FAIL",
        "mode": "BOUNDED_DRY_RUN",
        "h5_environment_contract_present": bool(h5),
        "cuda_device": torch.cuda.get_device_name(0),
        "checkpoint_strict_identity_preflight": True,
        "dataset_manifest_hash": identities["dataset_manifest_hash"],
        "forward": {
            "finite": finite,
            "endstate_shape": list(endstate.shape),
            "score_shape": list(score.shape),
        },
        "guard": {
            "weak_inactive": not weak["active_risk"],
            "strict_accepted": strict["active_risk"],
            "history_accepted": history["active_risk"],
            "forbidden_inputs_rejected": forbidden,
        },
        "brir1_historical_blockers_preserved": 34,
        "runtime_gt_used": False,
        "optimizer_constructed": False,
        "training_started": False,
        "formal_h5_started": False,
        "formal_h5_pass_claimed": False,
        "production_qualified": False,
        "production_activation_authorized": False,
    }
    write("combined_h5_entry", result)
    write("combined_h5_dryrun", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
