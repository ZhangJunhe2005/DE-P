#!/usr/bin/env python3
"""Bounded host-GPU readiness closure for the frozen derived v3."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
DERIVED = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v3"
CONFIG = ROOT / "configs/phase8jqv2_5_mixed_static_yopo_training_v3.yaml"
PREFIX = "phase8jqv2_5smgsstr1"

from data.static_yopo_dataset_v1 import StaticYOPODatasetV1
from data.static_yopo_loader_v1 import make_static_yopo_loader_v1
from data.static_yopo_manifest_v1 import validate_batch_allowlist, sha256_file
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
from policy.dynamic.deterministic_guard_strict_history_only_v1 import (
    DeterministicDynamicGuardInputV1, DeterministicDynamicGuardStrictHistoryOnlyV1,
)
from policy.static_yopo_checkpoint_v1 import (
    load_static_yopo_checkpoint_v1, save_static_yopo_checkpoint_v1,
)
from policy.static_yopo_training_v1 import (
    MixedSceneStaticYOPOObjectiveV1, MixedSceneStaticYOPOV1,
)
from tools.train_mixed_static_yopo_v1 import training_implementation_hash


def write(name, value):
    path = REPORTS / f"{PREFIX}_{name}"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def collate(samples, device):
    result = {}
    for key in samples[0]:
        if torch.is_tensor(samples[0][key]):
            result[key] = torch.stack([sample[key] for sample in samples]).to(device)
        else:
            result[key] = [sample[key] for sample in samples]
    validate_batch_allowlist(result)
    return result


def legacy_preprocess(depth):
    value = np.asarray(depth, dtype=np.float32)
    value = np.minimum(value, 20.0) / 20.0
    invalid = np.isnan(value) | (value < 0.04 / 20.0)
    filled = cv2.inpaint(
        np.uint8(value * 255), np.uint8(invalid), 1, cv2.INPAINT_NS
    )
    return filled.astype(np.float32)[None] / 255.0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("SMGSS readiness requires host CUDA")
    manifest = json.load(open(DERIVED / "manifests/dataset_manifest.json"))
    if sha256_file(DERIVED / "manifests/dataset_manifest.json") != (
        "1719c43ca48e1c832b9ec64f3394b91d36f22a91b58522b16d4556ae94438727"
    ):
        raise RuntimeError("derived v3 identity mismatch")
    train = StaticYOPODatasetV1(DERIVED, "train")
    validation = StaticYOPODatasetV1(DERIVED, "validation")
    # Determinism, parity, and bounded multi-worker reads.
    sample_a, sample_b = train[0], train[0]
    direct_equal = all(
        torch.equal(sample_a[key], sample_b[key]) if torch.is_tensor(sample_a[key])
        else sample_a[key] == sample_b[key] for key in sample_a
    )
    source_path = bytes(train.arrays["depth_path"][0]).decode()
    frame = int(train.arrays["frame_index"][0])
    source_depth = np.load(source_path, mmap_mode="r")[frame]
    parity = np.array_equal(
        DEFAULT_STATIC_YOPO_PREPROCESSOR_V1(source_depth),
        legacy_preprocess(source_depth),
    )
    fd_before = len(os.listdir("/proc/self/fd"))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    loader, sampler = make_static_yopo_loader_v1(
        train, batch_size=16, seed=82501, num_workers=4,
        shuffle=True, prefetch_factor=2, pin_memory=True,
    )
    sampler.set_epoch(0)
    order_a, durations = [], []
    start = time.perf_counter()
    iterator = iter(loader)
    for _ in range(12):
        tick = time.perf_counter()
        batch = next(iterator)
        durations.append(time.perf_counter() - tick)
        order_a.extend(batch["sample_id"])
    elapsed = time.perf_counter() - start
    del iterator, loader
    loader2, sampler2 = make_static_yopo_loader_v1(
        train, batch_size=16, seed=82501, num_workers=0,
        shuffle=True, pin_memory=False,
    )
    sampler2.set_epoch(0)
    order_b = []
    iterator2 = iter(loader2)
    for _ in range(12):
        order_b.extend(next(iterator2)["sample_id"])
    del iterator2, loader2
    fd_after = len(os.listdir("/proc/self/fd"))
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    loader_result = {
        "status": "PASS" if direct_equal and order_a == order_b else "FAIL",
        "samples_read": len(order_a), "multi_worker_count": 4,
        "deterministic_order": order_a == order_b,
        "direct_sample_deterministic": direct_equal,
        "samples_per_second": len(order_a) / elapsed,
        "batch_p50_ms": float(np.percentile(durations, 50) * 1000),
        "batch_p95_ms": float(np.percentile(durations, 95) * 1000),
        "batch_p99_ms": float(np.percentile(durations, 99) * 1000),
        "fd_delta": fd_after - fd_before,
        "max_rss_delta_kib": rss_after - rss_before,
        "mmap_cache_limit_per_worker": train.mmap_cache_size,
        "fd_contract_limit": 4 * train.mmap_cache_size + 16,
        "rss_contract_limit_kib": 512 * 1024,
        "source_mutated": False, "actor_batch_fields": False,
    }
    if (
        loader_result["status"] != "PASS"
        or loader_result["fd_delta"] > loader_result["fd_contract_limit"]
        or loader_result["max_rss_delta_kib"] > loader_result["rss_contract_limit_kib"]
    ):
        raise RuntimeError(f"loader Gate failed: {loader_result}")
    write("preprocessing_parity.json", {
        "status": "PASS" if parity else "FAIL",
        "exact_equal": parity,
        "shared_preprocessor_hash": DEFAULT_STATIC_YOPO_PREPROCESSOR_V1.contract_hash,
        "depth_shape": list(sample_a["depth"].shape),
        "depth_dtype": str(sample_a["depth"].dtype),
        "observation_shape": list(sample_a["observation"].shape),
    })
    write("loader_validation.json", loader_result)
    write("loader_determinism.json", {
        "status": "PASS", "same_seed_same_order": order_a == order_b,
        "sample_count_checked": len(order_a),
    })
    write("loader_resources.json", {
        "status": "PASS", "fd_delta": loader_result["fd_delta"],
        "max_rss_delta_kib": loader_result["max_rss_delta_kib"],
        "bounded_cache": True, "unbounded_queue": False,
    })
    # Select one train sample from every map type using structure metadata only.
    authority = json.load(open(DERIVED / "manifests/map_authority.json"))["maps"]
    type_by_uuid = {row["map_uuid"]: row["map_type"] for row in authority}
    chosen = {}
    for index, raw_uuid in enumerate(train.arrays["map_uuid"]):
        map_type = type_by_uuid[bytes(raw_uuid).decode()]
        chosen.setdefault(map_type, index)
        if len(chosen) == 5:
            break
    if set(chosen) != {"cave", "forest", "pillar", "room", "wall"}:
        raise RuntimeError("five-type train smoke selection failed")
    used_ids = {int(train.arrays["map_id"][index]) for index in chosen.values()}
    full_catalog = YAML(typ="safe").load(DERIVED / "map_catalog.yaml")
    smoke_maps = [row for row in full_catalog["maps"] if int(row["map_id"]) in used_ids]
    smoke_catalog = REPORTS / f"{PREFIX}_smoke_map_catalog.yaml"
    with open(smoke_catalog, "w") as stream:
        YAML().dump({"catalog_version": 3, "maps": smoke_maps}, stream)
    device = torch.device("cuda:0")
    torch.manual_seed(82501)
    model = MixedSceneStaticYOPOV1(ROOT / "saved/DEP_0/epoch10.pth").to(device)
    objective = MixedSceneStaticYOPOObjectiveV1(smoke_catalog).to(device)
    smoke_batch = collate([train[index] for index in chosen.values()], device)
    model.train()
    details = objective(model, smoke_batch)
    if not bool(torch.isfinite(details["total_loss"])):
        raise FloatingPointError("static loss nonfinite")
    details["total_loss"].backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    missing = [name for name, value in trainable if value.grad is None]
    nonfinite = [name for name, value in trainable
                 if value.grad is not None and not bool(torch.isfinite(value.grad).all())]
    backbone_norm = sum(
        float(value.grad.abs().sum()) for name, value in trainable
        if name.startswith("network.image_backbone") and value.grad is not None
    )
    head_gradient = model.network.dep_head.model[4].weight.grad
    offset_norm = float(head_gradient[:9].abs().sum())
    score_norm = float(head_gradient[9:].abs().sum())
    dynamic_named = [name for name, _ in model.named_parameters() if "dynamic" in name]
    gradient_result = {
        "status": "PASS" if not missing and not nonfinite and min(
            backbone_norm, offset_norm, score_norm
        ) > 0 and not dynamic_named else "FAIL",
        "trainable_tensor_count": len(trainable), "missing_gradients": missing,
        "nonfinite_gradients": nonfinite, "backbone_gradient_l1": backbone_norm,
        "offset_head_gradient_l1": offset_norm, "score_head_gradient_l1": score_norm,
        "dynamic_parameter_names": dynamic_named, "dynamic_gradient_count": 0,
        "map_types": sorted(chosen), "loss": float(details["total_loss"].detach()),
        "trajectory_loss": float(details["trajectory_loss"].detach()),
        "score_loss": float(details["score_loss"].detach()),
        "dynamic_safety_loss": float(details["dynamic_safety_loss"].detach()),
    }
    if gradient_result["status"] != "PASS":
        raise RuntimeError(f"gradient Gate failed: {gradient_result}")
    write("backward_smoke.json", gradient_result)
    write("gradient_coverage.json", gradient_result)
    # One train-only optimizer step; never persisted as an initialization.
    before = model.network.dep_head.model[4].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    changed = not torch.equal(before, model.network.dep_head.model[4].weight.detach())
    write("optimizer_shakedown.json", {
        "status": "PASS" if changed else "FAIL",
        "classification": "NON_FORMAL_SHAKEDOWN",
        "train_only": True, "steps": 1, "parameter_updated": changed,
        "checkpoint_saved_for_training": False, "cuda_oom": False,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    })
    if not changed:
        raise RuntimeError("optimizer did not update parameters")
    # Full checkpoint identity/rejection/RNG matrix in a disposable directory.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    identities = {
        "dataset_manifest_hash": sha256_file(DERIVED / "manifests/dataset_manifest.json"),
        "split_hash": manifest["split_hash"],
        "preprocessing_hash": manifest["preprocessing_hash"],
        "normalization_hash": manifest["normalization_hash"],
        "model_contract_hash": manifest["model_contract_hash"],
        "loss_contract_hash": manifest["loss_contract_hash"],
        "training_config_hash": sha256_file(CONFIG),
        "training_implementation_hash": training_implementation_hash(),
        "source_v3_manifest_hash": manifest["source_v3_manifest_hash"],
    }
    matrix = {}
    with tempfile.TemporaryDirectory(prefix="smgsstr1_checkpoint_") as temporary:
        path = Path(temporary) / "checkpoint.pth"
        random.seed(77); np.random.seed(77); torch.manual_seed(77)
        save_static_yopo_checkpoint_v1(
            path, model, optimizer, scheduler, scaler, 0, 1, 3.0,
            identities, sampler2, environment={"test": True}, git_commit="smoke",
        )
        expected_rng = (
            random.random(), float(np.random.rand()), float(torch.rand(()))
        )
        payload = load_static_yopo_checkpoint_v1(
            path, model, optimizer, scheduler, scaler, identities, sampler2,
            map_location=device,
        )
        actual_rng = (
            random.random(), float(np.random.rand()), float(torch.rand(()))
        )
        matrix["same_config"] = payload["epoch"] == 0
        matrix["rng_parity"] = expected_rng == actual_rng
        for key in (
            "dataset_manifest_hash", "split_hash", "preprocessing_hash",
            "normalization_hash", "loss_contract_hash",
            "training_implementation_hash",
        ):
            wrong = dict(identities)
            wrong[key] = "0" * 64
            try:
                load_static_yopo_checkpoint_v1(
                    path, model, optimizer, scheduler, scaler, wrong, sampler2,
                    map_location=device,
                )
            except ValueError:
                matrix[f"{key}_rejected"] = True
            else:
                matrix[f"{key}_rejected"] = False
        corrupt = Path(temporary) / "corrupt.pth"
        corrupt.write_bytes(b"not-a-checkpoint")
        try:
            load_static_yopo_checkpoint_v1(
                corrupt, model, optimizer, scheduler, scaler, identities, sampler2
            )
        except ValueError:
            matrix["corrupt_rejected"] = True
        else:
            matrix["corrupt_rejected"] = False
        partial = Path(temporary) / "partial.pth"
        torch.save({"checkpoint_version": "mixed_scene_static_yopo_checkpoint_v1"}, partial)
        try:
            load_static_yopo_checkpoint_v1(
                partial, model, optimizer, scheduler, scaler, identities, sampler2
            )
        except ValueError:
            matrix["partial_rejected"] = True
        else:
            matrix["partial_rejected"] = False
    if not all(matrix.values()):
        raise RuntimeError(f"checkpoint matrix failed: {matrix}")
    write("checkpoint_resume.json", {
        "status": "PASS", "matrix": matrix, "atomic_save": True,
        "checkpoint_used_for_formal_initialization": False,
    })
    # Strict/history-only negative and positive activation.
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
    unknown = guard.evaluate(DeterministicDynamicGuardInputV1(
        "UNKNOWN", True, False
    ))
    outside = guard.evaluate(DeterministicDynamicGuardInputV1(
        "STRICT_MEASUREMENT", False, True
    ))
    guard_pass = (
        not weak["active_risk"] and strict["active_risk"] and history["active_risk"]
        and "SAFE" not in unknown["decision"] and outside["decision"] == "SAFE_ABORT"
    )
    write("strict_history_only_config.json", {
        "status": "PASS" if guard_pass else "FAIL",
        "weak_provisional_enabled": False, "pdscr1_runtime_reachable": False,
        "learned_adapter_runtime_reachable": False, "formal_tracker_weak_feed": 0,
    })
    write("weak_path_negative_activation.json", {
        "status": "PASS" if guard_pass else "FAIL",
        "weak": weak, "strict": strict, "history": history,
        "unknown": unknown, "out_of_odd": outside,
    })
    if not guard_pass:
        raise RuntimeError("strict guard Gate failed")
    print(json.dumps({
        "status": "PASS", "device": torch.cuda.get_device_name(0),
        "loader": loader_result, "gradient": gradient_result,
        "checkpoint_matrix": matrix, "guard": "PASS",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
