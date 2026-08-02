#!/usr/bin/env python3
"""Freeze the exact epoch-boundary resume evidence for one shakedown run."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    schedule = payload["metadata"]["training_schedule"]
    expected_steps = int(schedule["steps_per_epoch"])
    evidence = {
        "status": "PASS" if payload["epoch"] == 1 and payload["global_step"] == expected_steps else "FAIL",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "resume_boundary": "completed_epoch_1",
        "saved_epoch": int(payload["epoch"]),
        "saved_global_step": int(payload["global_step"]),
        "expected_global_step": expected_steps,
        "static_sampler_state": payload.get("static_sampler_state"),
        "dynamic_sampler_generator_state_present": payload.get("dynamic_sampler_generator_state") is not None,
        "rng_state_present": payload.get("torch_rng_state") is not None,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
