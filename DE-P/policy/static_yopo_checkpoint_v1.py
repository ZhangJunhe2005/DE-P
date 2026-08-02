"""Atomic, identity-strict formal checkpoint and deterministic resume."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_VERSION = "mixed_scene_static_yopo_checkpoint_v1"
REQUIRED_IDENTITIES = (
    "dataset_manifest_hash", "split_hash", "preprocessing_hash",
    "normalization_hash", "model_contract_hash", "loss_contract_hash",
    "training_config_hash", "training_implementation_hash",
    "source_v3_manifest_hash",
)


def capture_rng_state(sampler=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.state_dict() if sampler is not None else None,
    }


def restore_rng_state(state, sampler=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``map_location="cuda"`` is valid for model/optimizer restoration but it
    # also relocates RNG tensors.  PyTorch's RNG setters require CPU
    # ByteTensors regardless of the checkpoint map location.
    cpu_state = torch.as_tensor(
        state["torch_cpu"], dtype=torch.uint8, device="cpu"
    ).contiguous()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state["torch_cuda"]:
        cuda_states = [
            torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
            for value in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    if sampler is not None and state["sampler"] is not None:
        sampler.load_state_dict(state["sampler"])


def _validate_identities(identities):
    missing = [key for key in REQUIRED_IDENTITIES if not identities.get(key)]
    if missing:
        raise ValueError(f"checkpoint identity fields missing: {missing}")


def save_static_yopo_checkpoint_v1(
    path, model, optimizer, scheduler, scaler, epoch, global_step, best_metric,
    identities, sampler=None, environment=None, git_commit="unknown",
):
    _validate_identities(identities)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch), "global_step": int(global_step),
        "best_validation_metric": float(best_metric),
        "rng": capture_rng_state(sampler),
        "identities": dict(identities),
        "environment": dict(environment or {}),
        "git_commit": str(git_commit),
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_static_yopo_checkpoint_v1(
    path, model, optimizer, scheduler, scaler, expected_identities, sampler=None,
    map_location="cpu",
):
    _validate_identities(expected_identities)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise ValueError(f"corrupt checkpoint: {exc}") from exc
    required = {
        "checkpoint_version", "model", "optimizer", "scheduler", "scaler",
        "epoch", "global_step", "best_validation_metric", "rng", "identities",
        "environment", "git_commit",
    }
    missing = required - set(payload) if isinstance(payload, dict) else required
    if missing:
        raise ValueError(f"partial checkpoint missing fields: {sorted(missing)}")
    if payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint schema mismatch")
    # Required v1 fields preserve old checkpoints.  Any additional identity
    # supplied by a newer training contract is equally strict on resume.
    for key in expected_identities:
        if payload["identities"].get(key) != expected_identities.get(key):
            raise ValueError(f"checkpoint identity mismatch: {key}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint scheduler state missing")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint scaler state missing")
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"], sampler)
    return payload
