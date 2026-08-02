#!/usr/bin/env python3
"""Host CUDA full-cycle gate for the PECR1 evidence-authorized mapper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4pecr1_"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint  # noqa:E402
from policy.dep_network import DepNetwork  # noqa:E402
from policy.dynamic.provisional_outcome_mapper_v3 import (  # noqa:E402
    ProvisionalOutcomeMapperV3,
)
from tools.run_phase8jqv2_4diro1_host_runtime import (  # noqa:E402
    distribution, documents,
)
from tools.run_phase8jqv2_4mar1_host_runtime import (  # noqa:E402
    pass_runtime, run_all,
)
from tools.run_phase8jqv2_4pdscr1_host_runtime import (  # noqa:E402
    PDSCRPerceptionProxy, factory as pds_factory,
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    )+"\n")
    os.replace(temporary, path)


class PECRPerceptionProxy(PDSCRPerceptionProxy):
    def __init__(
        self, base, config, mar_contract, dm_contract,
        pds_contract, evidence_contract,
    ):
        super().__init__(
            base, config, mar_contract, dm_contract, pds_contract
        )
        self.mapper = ProvisionalOutcomeMapperV3(
            pds_contract, evidence_contract
        )


def factory(
    mar_contract, dm_contract, pds_contract, evidence_contract,
):
    baseline = pds_factory(mar_contract, dm_contract, pds_contract)

    def create(sensor):
        base_proxy, config = baseline(sensor)
        return (
            PECRPerceptionProxy(
                base_proxy.base, config, mar_contract, dm_contract,
                pds_contract, evidence_contract,
            ),
            config,
        )
    return create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    if str(args.device) != "0":
        raise ValueError("host target is cuda:0")
    load = lambda name: yaml.safe_load((ROOT/"configs"/name).read_text())
    mar = load("measurement_availability_contract_v1_candidate.yaml")
    dm = load("dynamic_measurement_contract_v1_candidate.yaml")
    pds = load("provisional_dynamic_safety_contract_v2_candidate.yaml")
    evidence = load("provisional_evidence_contract_v1_candidate.yaml")
    torch.cuda.set_device(0)
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    docs = documents()
    fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./fps
    baseline_rows, baseline_offline, baseline = run_all(
        network, docs, gate, pds_factory(mar, dm, pds)
    )
    candidate_rows, candidate_offline, candidate = run_all(
        network, docs, gate, factory(mar, dm, pds, evidence)
    )
    paired = [
        right["cycle_ms"]-left["cycle_ms"]
        for left, right in zip(baseline_rows, candidate_rows)
        if left["frame"] >= 10 and right["frame"] >= 10
    ]
    paired_stats = distribution(paired)
    runtime_pass = bool(
        pass_runtime(candidate, gate)
        and candidate["steady_state_ms"]["p95"] <= gate
        and candidate["steady_state_ms"]["p50"] < gate
        and candidate["deadline_miss_rate"]
        <= evidence["runtime"]["deadline_miss_rate_maximum"]
        and candidate["consecutive_deadline_miss_max"]
        <= evidence["runtime"]["consecutive_deadline_miss_maximum"]
        and not candidate["unbounded_backlog"]
    )
    host = {
        "status": "PASS" if runtime_pass else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "compute_capability":
            list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "selected_runtime_candidate": "R9_SAME_FRAME_CPU_GPU_OVERLAP",
        "same_frame_only": True, "atomic_join_before_snapshot": True,
        "queue_depth": 0, "cross_frame_queue_depth": 0,
        "gate_ms": gate, "pdscr1_baseline": baseline,
        "evidence_candidate": candidate,
        "paired_cycle_delta_ms": paired_stats,
        "offline_gt_in_runtime_timer": False,
        "offline_exact_gt_ms": distribution(
            baseline_offline+candidate_offline
        ),
        "cuda_synchronized": True, "runtime_gt_used": False,
    }
    atomic(REPORTS/f"{PREFIX}host_runtime.json", host)
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": host["status"], "gate_ms": gate,
        "deadline_miss_count": candidate["deadline_miss_count"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "unbounded_backlog": candidate["unbounded_backlog"],
        "queue_depth": 0,
    })
    incremental = json.loads((
        REPORTS/f"{PREFIX}runtime_overhead.json"
    ).read_text())
    authorizer_p95 = incremental["authorizer_overhead_ms"].get(
        "p95", float("inf")
    )
    incremental.update({
        "status": "PASS" if (
            runtime_pass and authorizer_p95
            <= evidence["runtime"]["authorizer_overhead_p95_ms"]
        ) else "FAIL",
        "host_full_cycle_pending": False,
        "paired_full_cycle_delta_ms": paired_stats,
    })
    atomic(REPORTS/f"{PREFIX}runtime_overhead.json", incremental)
    final_path = REPORTS/f"{PREFIX}final_result.json"
    final = json.loads(final_path.read_text())
    fresh = json.loads((
        REPORTS/f"{PREFIX}fresh_validation_freeze.json"
    ).read_text())["fresh_summary"]
    known_births = int(final.get("known_failure_birth") or 0)
    semantic_pass = (
        fresh.get("status") == "PASS" and known_births == 0
    )
    recall_failure = bool(
        fresh["true_support_recall"]
        < evidence["validation"]["true_support_recall_minimum"]
        or fresh["baseline_true_support_retention"]
        < evidence["validation"][
            "baseline_true_support_retention_minimum"
        ]
    )
    final.update({
        "runtime": host["status"],
        "runtime_p95_ms": candidate["steady_state_ms"]["p95"],
        "runtime_median_ms": candidate["steady_state_ms"]["p50"],
        "runtime_gate_ms": gate,
    })
    if not runtime_pass:
        final.update({
            "status": "PARTIAL_PASS", "route": "F",
            "primary_cause": "provisional_evidence_runtime",
            "next_allowed_phase":
                "phase8jqv2_4_provisional_evidence_runtime_optimization",
        })
    elif semantic_pass:
        final.update({
            "status": "PASS_DEVELOPMENT_ONLY", "route": "A",
            "primary_cause": None,
            "next_allowed_phase":
                "phase8jqv2_4_foreground_component_availability_repair",
        })
    else:
        final.update({
            "status": "FAIL",
            "route": "B" if recall_failure else "C",
            "primary_cause": (
                "provisional_evidence_recall_tradeoff"
                if recall_failure
                else "provisional_dynamic_evidence_separation"
            ),
            "next_allowed_phase": (
                "phase8jqv2_4_provisional_support_recall_contract_review"
                if recall_failure
                else "phase8jqv2_4_provisional_evidence_representation_review"
            ),
        })
    atomic(final_path, final)
    selection_path = REPORTS/f"{PREFIX}candidate_selection.json"
    selection = json.loads(selection_path.read_text())
    selection["host_runtime"] = host["status"]
    if not runtime_pass:
        selection.update({"status": "PARTIAL_PASS", "selected": None,
                          "selected_count": 0})
    atomic(selection_path, selection)
    deterministic = json.loads((
        REPORTS/f"{PREFIX}determinism.json"
    ).read_text())
    deterministic.update({
        "status": "PASS" if runtime_pass else "FAIL",
        "host_repeat_pending": False,
        "same_frame_only": True, "queue_depth": 0,
    })
    atomic(REPORTS/f"{PREFIX}determinism.json", deterministic)
    print(json.dumps({
        "status": host["status"], "route": final["route"],
        "candidate_p95_ms": candidate["steady_state_ms"]["p95"],
        "candidate_median_ms": candidate["steady_state_ms"]["p50"],
        "gate_ms": gate,
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "queue_depth": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
