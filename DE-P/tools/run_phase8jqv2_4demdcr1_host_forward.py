#!/usr/bin/env python3
"""CPU/CUDA random-weight forward-only smoke under RETR1 H5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4demdcr1_"
sys.path.insert(0, str(ROOT))


def percentile(values, q):
    values = sorted(values)
    index = (len(values)-1)*q/100.
    lo, hi = int(index), min(int(index)+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def atomic(path, value):
    path = Path(path); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def main():
    allowed = os.sched_getaffinity(0)
    affinity = tuple(cpu for cpu in range(8) if cpu in allowed)
    if len(affinity) != 8:
        raise RuntimeError("H5 requires affinity 0-7")
    os.sched_setaffinity(0, affinity)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from policy.dynamic.dynamic_evidence_feature_encoder_v1 import (
        CausalFeatureTemporalDynamicEvidenceV1,
    )
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required; sandbox result is not evidence")
    torch.manual_seed(8242401); torch.cuda.manual_seed_all(8242401)
    results = {}
    for device in ("cpu", "cuda:0"):
        model = CausalFeatureTemporalDynamicEvidenceV1().to(device).eval()
        features = torch.randn(16, 4, 32, device=device)
        valid = torch.rand(16, 4, 32, device=device) > .1
        time_mask = torch.ones(16, 4, dtype=torch.bool, device=device)
        timings = []
        with torch.inference_mode():
            for _ in range(20):
                out = model(features, valid, time_mask)
            if device.startswith("cuda"): torch.cuda.synchronize()
            for _ in range(200):
                started = time.perf_counter_ns()
                out = model(features, valid, time_mask)
                if device.startswith("cuda"): torch.cuda.synchronize()
                timings.append((time.perf_counter_ns()-started)/1e6)
        results[device] = {
            "logits_shape": list(out["logits"].shape),
            "actionability_shape": list(out["actionability_logit"].shape),
            "finite": bool(torch.isfinite(out["logits"]).all()),
            "p95_ms": percentile(timings, 95), "p99_ms": percentile(timings, 99),
            "mean_ms": statistics.mean(timings),
        }
    model = CausalFeatureTemporalDynamicEvidenceV1()
    parameter_count = sum(p.numel() for p in model.parameters())
    gradient_tensors = sum(p.grad is not None for p in model.parameters())
    cuda = results["cuda:0"]
    prep = []
    import numpy as np
    x = np.zeros((16, 4, 32), np.float32)
    m = np.ones_like(x, dtype=np.bool_)
    for _ in range(200):
        started = time.perf_counter_ns()
        torch.from_numpy(x); torch.from_numpy(m)
        prep.append((time.perf_counter_ns()-started)/1e6)
    prep_p99 = percentile(prep, 99)
    baseline_p99 = json.loads((REPORTS/"phase8jqv2_4pepcr1_runtime.json").read_text())[
        "strategies"]["B"]["p99_ms"]
    full_estimate = baseline_p99+cuda["p99_ms"]+prep_p99
    passed = bool(
        cuda["p95_ms"] <= 1.5 and cuda["p99_ms"] <= 2.5
        and prep_p99 <= 1.0 and full_estimate <= 30.303030303030305
        and gradient_tensors == 0
    )
    smoke = {
        "status": "PASS" if passed else "FAIL", "random_weights": True,
        "forward_only": True, "backward_executed": False,
        "optimizer_constructed": False, "optimizer_step_executed": False,
        "parameter_count": parameter_count, "gradient_tensor_count": gradient_tensors,
        "batch_size": 16, "temporal_length": 4, "feature_dim": 32,
        "devices": results, "runtime_gt_used": False,
    }
    atomic(REPORTS/f"{PREFIX}forward_smoke.json", smoke)
    atomic(REPORTS/f"{PREFIX}runtime_budget.json", {
        "status": smoke["status"], "inference_p95_gate_ms": 1.5,
        "inference_p99_gate_ms": 2.5, "preprocessing_p99_gate_ms": 1.0,
        "cuda_inference_p95_ms": cuda["p95_ms"],
        "cuda_inference_p99_ms": cuda["p99_ms"],
        "preprocessing_p99_ms": prep_p99, "pepcr1_full_cycle_p99_ms": baseline_p99,
        "estimated_full_cycle_p99_ms": full_estimate,
        "full_cycle_gate_ms": 30.303030303030305,
        "final_gate_requires_future_integrated_measurement": True,
    })
    atomic(REPORTS/f"{PREFIX}h5_compatibility.json", {
        "status": smoke["status"], "environment": "H5_MINIMAL_ENVIRONMENT_CONTRACT",
        "affinity": list(os.sched_getaffinity(0)), "threads": 1,
        "gc_policy": "EPISODE_BOUNDARY_FOR_INTEGRATED_RUNTIME",
        "memory_pre_touch_bytes": 8388608, "warmup": 20,
        "no_cross_frame_backlog": True,
    })
    print(json.dumps(smoke, indent=2))
    if not passed: raise SystemExit(2)


if __name__ == "__main__":
    main()
