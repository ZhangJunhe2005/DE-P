#!/usr/bin/env python3
"""Validate legacy and corrected DEP backbones on a CUDA-enabled host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from tests.baseline_helpers import reset_lattice_singleton, seed_everything


def fail(message, code=2):
    print(f"HOST BACKBONE VALIDATION FAILED: {message}", file=sys.stderr)
    return code


def benchmark(model, depth, observation, warmup=20, iterations=100):
    model.eval()
    times = []
    torch.cuda.reset_peak_memory_stats(depth.device)
    with torch.inference_mode():
        for _ in range(warmup):
            model.inference(depth, observation.clone())
        torch.cuda.synchronize(depth.device)
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.inference(depth, observation.clone())
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return {
        "inference_average_ms": statistics.fmean(times),
        "inference_p95_ms": float(np.percentile(times, 95)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(depth.device)),
    }


def train_step(model, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    depth = torch.randn(2, 1, 96, 160, device=device)
    observation = torch.randn(2, 9, 3, 5, device=device)
    optimizer.zero_grad(set_to_none=True)
    endstate, score = model(depth, observation)
    loss = endstate.square().mean() + score.mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing, NaN, or Inf gradient")
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "optimizer_step": "PASS",
        "gradient_tensor_count": len(gradients),
        "loss": float(loss.detach().cpu()),
    }


def shape_and_finite(model, depth, observation):
    model.eval()
    with torch.inference_mode():
        feature = model.image_backbone(depth)
        endstate, score = model.inference(depth, observation.clone())
    expected = ((1, 64, 3, 5), (1, 9, 3, 5), (1, 3, 5))
    actual = (tuple(feature.shape), tuple(endstate.shape), tuple(score.shape))
    if actual != expected or not all(torch.isfinite(value).all() for value in (feature, endstate, score)):
        raise RuntimeError(f"invalid shape or non-finite output: {actual}")
    return feature, endstate, score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-checkpoint", type=Path,
                        default=ROOT / "saved/DEP_0/epoch10.pth")
    parser.add_argument("--corrected-checkpoint", type=Path,
                        default=ROOT / "saved/DEP_corrected_init/epoch10_converted.pth")
    parser.add_argument("--cpu-golden-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def cpu_golden_result(checkpoint):
    """Run the authoritative same-device regression against the CPU fixture."""
    fixture = torch.load(
        ROOT / "tests/fixtures/legacy_forward_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != fixture["checkpoint_sha256"]:
        raise RuntimeError(
            f"legacy checkpoint SHA256 mismatch: actual={checkpoint_hash}, "
            f"fixture={fixture['checkpoint_sha256']}"
        )
    reset_lattice_singleton()
    model = DepNetwork(backbone_variant="legacy").cpu().eval()
    load_dep_checkpoint(model, checkpoint, "legacy")
    with torch.inference_mode():
        endstate, score = model.inference(fixture["depth"], fixture["obs"].clone())
    result = {
        "device": "cpu",
        "checkpoint_sha256": checkpoint_hash,
        "endstate_max_abs_diff": float(torch.max(torch.abs(endstate - fixture["endstate"]))),
        "score_max_abs_diff": float(torch.max(torch.abs(score - fixture["score"]))),
        "tolerance": 1e-7,
    }
    result["status"] = (
        "PASS" if max(result["endstate_max_abs_diff"], result["score_max_abs_diff"])
        <= result["tolerance"] else "FAIL"
    )
    return result


def run_isolated_cpu_golden(checkpoint):
    """Hide CUDA so LatticePrimitive and every tensor are created on CPU."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cpu-golden-only",
        "--legacy-checkpoint",
        str(checkpoint),
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"isolated CPU golden regression exited {completed.returncode}")
    marker = "LEGACY_CPU_GOLDEN_RESULT "
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("isolated CPU golden regression did not emit one structured result")
    return json.loads(lines[0][len(marker):])


def main():
    args = parse_args()
    if args.cpu_golden_only:
        result = cpu_golden_result(args.legacy_checkpoint)
        print("LEGACY_CPU_GOLDEN_RESULT " + json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 2

    print("torch_version:", torch.__version__)
    print("torch_cuda_build:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return fail("NOT MEASURED IN CODEX SANDBOX: CUDA device is not visible")
    if not args.corrected_checkpoint.is_file():
        return fail(
            f"converted checkpoint not found: {args.corrected_checkpoint}; "
            "run tools/convert_legacy_to_corrected.py first"
        )

    try:
        legacy_cpu_golden = run_isolated_cpu_golden(args.legacy_checkpoint)
    except Exception as exc:
        return fail(f"authoritative CPU golden regression failed: {exc}")

    seed_everything(314159)
    device = torch.device("cuda:0")
    reset_lattice_singleton()
    fixture = torch.load(
        ROOT / "tests/fixtures/legacy_forward_reference.pt",
        map_location="cpu",
        weights_only=True,
    )
    depth = fixture["depth"].to(device)
    observation = fixture["obs"].to(device)

    legacy = DepNetwork(backbone_variant="legacy").to(device)
    load_dep_checkpoint(legacy, args.legacy_checkpoint, "legacy")
    legacy_feature, legacy_end, legacy_score = shape_and_finite(legacy, depth, observation)
    legacy_cpu_cuda_difference = {
        "endstate_max_abs_diff": float(torch.max(torch.abs(
            legacy_end.cpu() - fixture["endstate"]
        ))),
        "score_max_abs_diff": float(torch.max(torch.abs(
            legacy_score.cpu() - fixture["score"]
        ))),
        "role": "diagnostic_only_not_a_regression_gate",
        "note": "CPU and CUDA/cuDNN use different numerical kernels and accumulation order",
    }

    seed_everything(314159)
    corrected_random = DepNetwork(backbone_variant="corrected").to(device)
    _, random_end, random_score = shape_and_finite(corrected_random, depth, observation)
    corrected_converted = DepNetwork(backbone_variant="corrected").to(device)
    load_dep_checkpoint(corrected_converted, args.corrected_checkpoint, "corrected")
    _, converted_end, converted_score = shape_and_finite(corrected_converted, depth, observation)

    comparison = {
        "endstate_mae": float(torch.mean(torch.abs(legacy_end - converted_end))),
        "endstate_max_abs_diff": float(torch.max(torch.abs(legacy_end - converted_end))),
        "endstate_cosine_similarity": float(F.cosine_similarity(
            legacy_end.flatten().unsqueeze(0), converted_end.flatten().unsqueeze(0)
        )),
        "score_mae": float(torch.mean(torch.abs(legacy_score - converted_score))),
        "score_rank_positions_changed": int(torch.sum(
            torch.argsort(legacy_score.flatten()) != torch.argsort(converted_score.flatten())
        )),
        "best_primitive_same": bool(
            torch.argmin(legacy_score).item() == torch.argmin(converted_score).item()
        ),
    }

    random_train = train_step(corrected_random, device)
    converted_train = train_step(corrected_converted, device)
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "legacy": {
            "checkpoint": str(args.legacy_checkpoint.resolve()),
            "strict_load": True,
            "golden_regression_cpu": legacy_cpu_golden,
            "cpu_cuda_numerical_difference": legacy_cpu_cuda_difference,
            "feature_shape": list(legacy_feature.shape),
            "endstate_shape": list(legacy_end.shape),
            "score_shape": list(legacy_score.shape),
            "parameter_count": sum(p.numel() for p in legacy.parameters()),
            **benchmark(legacy, depth, observation),
        },
        "corrected_random": {
            "endstate_shape": list(random_end.shape),
            "score_shape": list(random_score.shape),
            "parameter_count": sum(p.numel() for p in corrected_random.parameters()),
            **random_train,
            **benchmark(corrected_random, depth, observation),
        },
        "corrected_converted": {
            "checkpoint": str(args.corrected_checkpoint.resolve()),
            "strict_load": True,
            "endstate_shape": list(converted_end.shape),
            "score_shape": list(converted_score.shape),
            "parameter_count": sum(p.numel() for p in corrected_converted.parameters()),
            **converted_train,
            **benchmark(corrected_converted, depth, observation),
        },
        "comparison_before_optimizer_step": comparison,
    }
    for section in ("legacy", "corrected_random", "corrected_converted"):
        if result[section]["peak_gpu_memory_bytes"] <= 0:
            return fail(f"non-positive GPU memory for {section}")
    print("HOST_BACKBONE_VARIANT_VALIDATION_RESULT")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
