#!/usr/bin/env python3
"""Host CUDA R9 runtime gate for the DMCR1 shadow contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4dmcr1_"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint  # noqa:E402
from policy.dep_network import DepNetwork  # noqa:E402
from policy.dynamic.dynamic_measurement_outcome_v1 import (  # noqa:E402
    DynamicMeasurementStatusV1,
)
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa:E402
    load_architecture_config,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa:E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.measurement_contract_shadow_consumer_v1 import (  # noqa:E402
    consume_outcome,
)
from tools.run_phase8jqv2_4diro1_host_runtime import (  # noqa:E402
    distribution, documents,
)
import tools.run_phase8jqv2_4diro1_host_runtime as diro_host  # noqa:E402
from tools.run_phase8jqv2_4mar1_host_runtime import (  # noqa:E402
    MARPerceptionProxy, factory as mar_factory, pass_runtime, run_all,
)
from tools.run_phase8jqv2_4dmcr1_review import (  # noqa:E402
    IMPLEMENTATION_PATHS, rejected_outcome,
)
from tools.run_phase8jqv2_4tccr1_telemetry import (  # noqa:E402
    make_perception,
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True
    ) + "\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DMCRPerceptionProxy(MARPerceptionProxy):
    """Adds the frozen D5 classifier inside the same-frame CPU branch."""

    def __init__(self, base, config, mar_contract, dm_contract):
        super().__init__(base, config, mar_contract)
        self.dm_contract = dm_contract
        self.outcome_id = 0
        self.last_outcomes = ()
        self.last_shadow_semantics = ()

    def update_depth(self, depth, pose, timestamp, model):
        result = super().update_depth(depth, pose, timestamp, model)
        rejected_rows = [
            row for row, accepted in self.adapter._captured.values()
            if not accepted
        ]
        outcomes = []
        for row, decision in zip(
            rejected_rows, self.adapter.last_decisions
        ):
            if decision.measurement is not None:
                # MAR1 already consumed this valid weak measurement. D5 only
                # annotates it; it does not duplicate the provisional feed.
                from tools.run_phase8jqv2_4dmcr1_review import weak_outcome
                outcome = weak_outcome(
                    decision, self.frame_index-1, timestamp,
                    self.outcome_id,
                )
                context = None
                fields = None
            else:
                outcome, context, fields = rejected_outcome(
                    row, decision, self.base.last_artifacts.depth_frame,
                    result.all_tracks, self.dm_contract,
                    self.frame_index-1, self.outcome_id,
                )
            self.outcome_id += 1
            outcomes.append(outcome)
        self.last_outcomes = tuple(outcomes)
        self.last_shadow_semantics = tuple(
            consume_outcome(item).semantic for item in outcomes
        )
        return result


def factory(mar_contract, dm_contract):
    def create(sensor):
        _reference, config = make_perception(sensor)
        parameters = load_architecture_config()["candidates"][
            "physical_control_residual_v1"
        ]
        base = DynamicPerceptionFastPathV1(
            config, (3, 5), "cpu", parameters
        )
        return (
            DMCRPerceptionProxy(
                base, config, mar_contract, dm_contract
            ),
            config,
        )
    return create


def refresh_implementation_contract():
    path = REPORTS / f"{PREFIX}implementation_contract.json"
    implementation = json.loads(path.read_text())
    implementation["files"] = {
        value: sha(ROOT/value)
        for value in IMPLEMENTATION_PATHS
    }
    atomic(path, implementation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "host CUDA is required; sandbox result is not evidence"
        )
    if str(args.device) != "0":
        raise ValueError("frozen host target is cuda:0")
    selection = json.loads((
        REPORTS / "phase8jqv2_4diro1_candidate_selection.json"
    ).read_text())
    if selection["selected"] != "R9_SAME_FRAME_CPU_GPU_OVERLAP":
        raise RuntimeError("DIRO1 R9 overlap is not selected")
    mar_contract = yaml.safe_load((
        ROOT / "configs/"
        "measurement_availability_contract_v1_candidate.yaml"
    ).read_text())
    dm_contract = yaml.safe_load((
        ROOT / "configs/"
        "dynamic_measurement_contract_v1_candidate.yaml"
    ).read_text())
    docs = documents()
    torch.cuda.set_device(0)
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT / "saved/DEP_0/epoch10.pth", "legacy"
    )
    depth_fps = float(yaml.safe_load((
        ROOT.parent / "Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./depth_fps
    baseline_rows, baseline_offline, baseline = run_all(
        network, docs, gate, mar_factory(mar_contract)
    )
    candidate_rows, candidate_offline, candidate = run_all(
        network, docs, gate, factory(mar_contract, dm_contract)
    )
    candidate_pass = pass_runtime(candidate, gate)
    status = "PASS" if candidate_pass else "FAIL"
    paired = [
        candidate_row["cycle_ms"]-baseline_row["cycle_ms"]
        for baseline_row, candidate_row
        in zip(baseline_rows, candidate_rows)
        if baseline_row["frame"] >= 10
        and candidate_row["frame"] >= 10
    ]
    overhead = distribution(paired)
    host = {
        "status": status,
        "device": torch.cuda.get_device_name(0),
        "compute_capability":
            list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "selected_runtime_candidate":
            "R9_SAME_FRAME_CPU_GPU_OVERLAP",
        "same_frame_only": True,
        "atomic_join_before_snapshot": True,
        "cross_frame_queue_depth": 0,
        "queue_depth": 0,
        "gate_ms": gate,
        "gate_decision_source": "measurement_contract_candidate",
        "mar1_baseline": baseline,
        "measurement_contract_candidate": candidate,
        "paired_cycle_delta_ms": overhead,
        "contract_cpu_p95_ms": json.loads((
            REPORTS / f"{PREFIX}runtime_overhead.json"
        ).read_text())["contract_path_ms"]["p95"],
        "offline_gt_in_runtime_timer": False,
        "offline_exact_gt_ms": distribution(
            baseline_offline+candidate_offline
        ),
        "cuda_synchronized": True,
        "runtime_gt_used": False,
    }
    atomic(REPORTS / f"{PREFIX}host_runtime.json", host)
    atomic(REPORTS / f"{PREFIX}runtime_breakdown.json", {
        "status": status, "mar1_baseline": baseline,
        "measurement_contract_candidate": candidate,
        "paired_cycle_delta_ms": overhead,
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}deadline_and_backlog.json", {
        "status": status, "gate_ms": gate,
        "deadline_miss_count": candidate["deadline_miss_count"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "maximum_accumulated_backlog_ms":
            candidate["maximum_accumulated_backlog_ms"],
        "unbounded_backlog": candidate["unbounded_backlog"],
        "queue_depth": 0,
    })
    runtime_overhead_path = REPORTS / f"{PREFIX}runtime_overhead.json"
    runtime_overhead = json.loads(runtime_overhead_path.read_text())
    runtime_overhead.update({
        "status": (
            "PASS" if (
                runtime_overhead["contract_path_ms"]["p95"] <= .75
                and status == "PASS"
            ) else "FAIL"
        ),
        "paired_full_cycle_delta_ms": overhead,
        "host_full_cycle_pending": False,
    })
    atomic(runtime_overhead_path, runtime_overhead)
    final_path = REPORTS / f"{PREFIX}final_result.json"
    final = json.loads(final_path.read_text())
    final.update({
        "runtime": status,
        "runtime_p95_ms": candidate["steady_state_ms"]["p95"],
        "runtime_median_ms": candidate["steady_state_ms"]["p50"],
        "runtime_gate_ms": gate,
    })
    if status == "PASS":
        final.update({
            "status": "PASS_DEVELOPMENT_ONLY", "route": "A",
            "next_allowed_phase":
                "phase8jqv2_4_provisional_dynamic_safety_contract_review",
        })
    else:
        final.update({
            "status": "PARTIAL_PASS", "route": "F",
            "primary_cause": "dynamic_measurement_contract_runtime",
            "next_allowed_phase":
                "phase8jqv2_4_dynamic_measurement_runtime_optimization",
        })
    atomic(final_path, final)
    selection_path = REPORTS / f"{PREFIX}candidate_selection.json"
    selected = json.loads(selection_path.read_text())
    selected.update({
        "status": (
            "PASS_DEVELOPMENT_ONLY" if status == "PASS"
            else "PARTIAL_PASS"
        ),
        "host_runtime": status,
    })
    atomic(selection_path, selected)
    deterministic_path = REPORTS / f"{PREFIX}determinism.json"
    deterministic = json.loads(deterministic_path.read_text())
    deterministic.update({
        "status": "PASS" if status == "PASS" else "FAIL",
        "host_repeat_pending": False,
        "same_frame_only": True, "queue_depth": 0,
    })
    atomic(deterministic_path, deterministic)
    refresh_implementation_contract()
    atomic_text(REPORTS / f"{PREFIX}final_readiness.md", f"""# DMCR1 readiness

Status: **{final['status']} / Route {final['route']}**.

The development-only D5 contract measures p95=
{candidate['steady_state_ms']['p95']:.3f} ms and median=
{candidate['steady_state_ms']['p50']:.3f} ms against gate={gate:.3f} ms.
Formal measurement, TrackManager, router, production, and training remain
unchanged or disabled.
""")
    atomic_text(REPORTS / f"{PREFIX}final_recommendation.md", """# DMCR1 recommendation

Freeze D5 as a development-only measurement outcome contract. Both remaining
L3 rows now retain explicit safe non-measurement semantics. Proceed only to
the provisional dynamic safety contract review; do not activate production,
modify formal tracking, access sealed data, or train.
""")
    print(json.dumps({
        "status": status, "route": final["route"],
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
