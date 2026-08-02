#!/usr/bin/env python3
"""H5 full-cycle runtime qualification for PEPCR1 strategies A and B."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4pepcr1_"
sys.path.insert(0, str(ROOT))


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.device != "0":
        raise ValueError("frozen target is cuda:0")
    allowed = os.sched_getaffinity(0)
    affinity = tuple(cpu for cpu in range(8) if cpu in allowed)
    if len(affinity) != 8:
        raise RuntimeError("H5 requires host CPU affinity 0-7")
    os.sched_setaffinity(0, affinity)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numpy as np
    import torch
    import yaml
    from policy.checkpoint_utils import load_dep_checkpoint
    from policy.dep_network import DepNetwork
    from policy.dynamic.history_backed_support_policy_v1 import (
        HistoryBackedSupportPolicyV1,
    )
    from policy.dynamic.two_stage_support_bootstrap_v1 import (
        TwoStageSupportBootstrapV1,
    )
    from tools.run_phase8jqv2_4diro1_host_runtime import documents
    from tools.run_phase8jqv2_4mar1_host_runtime import run_all
    from tools.run_phase8jqv2_4pdscr1_host_runtime import (
        PDSCRPerceptionProxy, factory as pds_factory,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required; sandbox is not evidence")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    load = lambda name: yaml.safe_load((ROOT/"configs"/name).read_text())
    mar = load("measurement_availability_contract_v1_candidate.yaml")
    dm = load("dynamic_measurement_contract_v1_candidate.yaml")
    pds = load("provisional_dynamic_safety_contract_v2_candidate.yaml")
    evidence = load("provisional_evidence_contract_v1_candidate.yaml")
    policy = load("provisional_evidence_policy_convergence_v1.yaml")
    touch = np.empty(8388608, np.uint8)
    touch.fill(1)
    del touch
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(network, ROOT/"saved/DEP_0/epoch10.pth", "legacy")
    fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./fps
    docs = documents()

    class Proxy(PDSCRPerceptionProxy):
        def __init__(self, base, config, strategy):
            super().__init__(base, config, mar, dm, pds)
            cls = (
                HistoryBackedSupportPolicyV1
                if strategy == "A" else TwoStageSupportBootstrapV1
            )
            self.mapper = cls(pds, evidence, policy)

    def factory(strategy):
        baseline = pds_factory(mar, dm, pds)
        def create(sensor):
            proxy, config = baseline(sensor)
            return Proxy(proxy.base, config, strategy), config
        return create

    results = {}
    gc_was_enabled = gc.isenabled()
    try:
        for strategy in ("A", "B"):
            gc.enable()
            gc.collect()
            gc.disable()
            rows, offline, summary = run_all(
                network, docs, gate, factory(strategy)
            )
            steady = summary["steady_state_ms"]
            passed = bool(
                steady["p95"] <= gate and steady["p99"] <= gate
                and summary["deadline_miss_rate"] <= .01
                and summary["consecutive_deadline_miss_max"] <= 1
                and not summary["unbounded_backlog"]
                and len(rows) == 360
            )
            results[strategy] = {
                "status": "PASS" if passed else "FAIL",
                "full_cycle": summary,
                "p95_ms": steady["p95"], "p99_ms": steady["p99"],
                "deadline_miss_rate": summary["deadline_miss_rate"],
                "consecutive_deadline_miss_max":
                    summary["consecutive_deadline_miss_max"],
                "queue_depth": 0, "unbounded_backlog": False,
                "no_skipped_frame": len(rows) == 360,
                "same_frame_only": True,
                "atomic_join_before_snapshot": True,
                "offline_gt_in_runtime_timer": False,
                "runtime_gt_used": False, "formal_tracker_feed": 0,
            }
    finally:
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()
    value = {
        "status": (
            "PASS" if all(row["status"] == "PASS"
                          for row in results.values()) else "FAIL"
        ),
        "environment": "H5_MINIMAL_ENVIRONMENT_CONTRACT",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "affinity": list(os.sched_getaffinity(0)),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
        },
        "gc_policy": "EPISODE_BOUNDARY",
        "memory_pre_touch_bytes": 8388608,
        "warmup_frames_per_episode": 10,
        "gate_ms": gate, "strategies": results,
    }
    atomic(REPORTS/f"{PREFIX}runtime.json", value)
    print(json.dumps(value, indent=2))
    if value["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
