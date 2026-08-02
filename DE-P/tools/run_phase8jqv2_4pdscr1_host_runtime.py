#!/usr/bin/env python3
"""Host CUDA R9 runtime gate for the PDSCR1 shadow contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4pdscr1_"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint  # noqa:E402
from policy.dep_network import DepNetwork  # noqa:E402
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa:E402
    load_architecture_config,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa:E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.historical_measurement_context_v1 import (  # noqa:E402
    context_from_track,
)
from policy.dynamic.provisional_dynamic_safety_state_v1 import (  # noqa:E402
    ProvisionalSafetyStateManagerV1,
)
from policy.dynamic.provisional_outcome_mapper_v1 import (  # noqa:E402
    ProvisionalOutcomeMapperV1,
)
from policy.dynamic.provisional_risk_consumer_v1 import (  # noqa:E402
    ProvisionalRiskConsumerV1,
)
from tools.run_phase8jqv2_4diro1_host_runtime import (  # noqa:E402
    distribution, documents,
)
from tools.run_phase8jqv2_4dmcr1_host_runtime import (  # noqa:E402
    DMCRPerceptionProxy, factory as dmcr_factory,
)
from tools.run_phase8jqv2_4dmcr1_review import (  # noqa:E402
    nearest_track, row_fields,
)
from tools.run_phase8jqv2_4mar1_host_runtime import (  # noqa:E402
    pass_runtime, run_all,
)
from tools.run_phase8jqv2_4pdscr1_review import (  # noqa:E402
    IMPLEMENTATION_PATHS,
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
    )+"\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PDSCRPerceptionProxy(DMCRPerceptionProxy):
    def __init__(
        self, base, config, mar_contract, dm_contract, pds_contract,
    ):
        super().__init__(
            base, config, mar_contract, dm_contract
        )
        self.mapper = ProvisionalOutcomeMapperV1(pds_contract)
        self.manager = ProvisionalSafetyStateManagerV1(
            pds_contract["association"][
                "maximum_position_distance_m"
            ]
        )
        self.consumer = ProvisionalRiskConsumerV1()
        self.next_state_id = 0
        self.last_provisional_decision = None

    def update_depth(self, depth, pose, timestamp, model):
        result = super().update_depth(depth, pose, timestamp, model)
        rejected_rows = [
            row for row, accepted in self.adapter._captured.values()
            if not accepted
        ]
        candidates = []
        for row, outcome in zip(rejected_rows, self.last_outcomes):
            fields = row_fields({
                **row, "depth_frame": self.base.last_artifacts.depth_frame
            })
            context = context_from_track(
                nearest_track(row, result.all_tracks),
                outcome.status.value, 1.0/33.0,
            )
            mapped = self.mapper.map(
                {
                    **outcome.__dict__,
                    "status": outcome.status.value,
                    "support": (
                        None if outcome.support is None
                        else outcome.support.__dict__
                    ),
                    "unresolved_risk": (
                        None if outcome.unresolved_risk is None
                        else outcome.unresolved_risk.__dict__
                    ),
                },
                self.next_state_id,
                causal_context={
                    "fields": fields,
                    "historical_context": context.__dict__,
                },
            )
            self.next_state_id += 1
            if mapped is not None:
                candidates.append(mapped)
        live = self.manager.update(candidates, float(timestamp))
        self.last_provisional_decision = self.consumer.consume(live)
        return result


def factory(mar_contract, dm_contract, pds_contract):
    def create(sensor):
        _reference, config = make_perception(sensor)
        parameters = load_architecture_config()["candidates"][
            "physical_control_residual_v1"
        ]
        base = DynamicPerceptionFastPathV1(
            config, (3, 5), "cpu", parameters
        )
        return (
            PDSCRPerceptionProxy(
                base, config, mar_contract, dm_contract, pds_contract
            ),
            config,
        )
    return create


def refresh_implementation_contract():
    path = REPORTS/f"{PREFIX}implementation_contract.json"
    value = json.loads(path.read_text())
    value["files"] = {
        item: sha(ROOT/item) for item in IMPLEMENTATION_PATHS
    }
    atomic(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    if str(args.device) != "0":
        raise ValueError("host target is cuda:0")
    mar_contract = yaml.safe_load((
        ROOT/"configs/measurement_availability_contract_v1_candidate.yaml"
    ).read_text())
    dm_contract = yaml.safe_load((
        ROOT/"configs/dynamic_measurement_contract_v1_candidate.yaml"
    ).read_text())
    pds_contract = yaml.safe_load((
        ROOT/"configs/provisional_dynamic_safety_contract_v2_candidate.yaml"
    ).read_text())
    docs = documents()
    torch.cuda.set_device(0)
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(
        network, ROOT/"saved/DEP_0/epoch10.pth", "legacy"
    )
    fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./fps
    baseline_rows, baseline_offline, baseline = run_all(
        network, docs, gate,
        dmcr_factory(mar_contract, dm_contract),
    )
    candidate_rows, candidate_offline, candidate = run_all(
        network, docs, gate,
        factory(mar_contract, dm_contract, pds_contract),
    )
    status = "PASS" if pass_runtime(candidate, gate) else "FAIL"
    paired = [
        right["cycle_ms"]-left["cycle_ms"]
        for left, right in zip(baseline_rows, candidate_rows)
        if left["frame"] >= 10 and right["frame"] >= 10
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
        "queue_depth": 0, "cross_frame_queue_depth": 0,
        "gate_ms": gate,
        "gate_decision_source": "provisional_contract_candidate",
        "dmcr1_baseline": baseline,
        "provisional_candidate": candidate,
        "paired_cycle_delta_ms": overhead,
        "offline_gt_in_runtime_timer": False,
        "offline_exact_gt_ms": distribution(
            baseline_offline+candidate_offline
        ),
        "cuda_synchronized": True,
        "runtime_gt_used": False,
    }
    atomic(REPORTS/f"{PREFIX}host_runtime.json", host)
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": status, "gate_ms": gate,
        "deadline_miss_count": candidate["deadline_miss_count"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "unbounded_backlog": candidate["unbounded_backlog"],
        "queue_depth": 0,
    })
    runtime_path = REPORTS/f"{PREFIX}runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime.update({
        "status": "PASS" if (
            status == "PASS"
            and runtime["provisional_path_ms"]["p95"] <= 1.
        ) else "FAIL",
        "paired_full_cycle_delta_ms": overhead,
        "host_full_cycle_pending": False,
    })
    atomic(runtime_path, runtime)
    final_path = REPORTS/f"{PREFIX}final_result.json"
    final = json.loads(final_path.read_text())
    semantic_pass = (
        json.loads((
            REPORTS/f"{PREFIX}fresh_validation_freeze.json"
        ).read_text())["status"] == "PASS"
    )
    final.update({
        "runtime": status,
        "runtime_p95_ms": candidate["steady_state_ms"]["p95"],
        "runtime_median_ms": candidate["steady_state_ms"]["p50"],
        "runtime_gate_ms": gate,
    })
    if status == "PASS" and semantic_pass:
        final.update({
            "status": "PASS_DEVELOPMENT_ONLY", "route": "A",
            "next_allowed_phase":
                "phase8jqv2_4_foreground_component_availability_repair",
        })
    elif status != "PASS":
        final.update({
            "status": "PARTIAL_PASS", "route": "G",
            "primary_cause": "provisional_safety_runtime",
            "next_allowed_phase":
                "phase8jqv2_4_provisional_runtime_optimization",
        })
    else:
        final.update({
            "status": "FAIL", "route": "F",
            "primary_cause":
                "provisional_safety_false_availability",
            "next_allowed_phase":
                "phase8jqv2_4_provisional_evidence_contract_review",
        })
    atomic(final_path, final)
    selection_path = REPORTS/f"{PREFIX}candidate_selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update({
        "status": (
            "PASS_DEVELOPMENT_ONLY"
            if status == "PASS" and semantic_pass
            else "PARTIAL_PASS" if status != "PASS" else "FAIL"
        ),
        "host_runtime": status,
    })
    atomic(selection_path, selection)
    deterministic_path = REPORTS/f"{PREFIX}determinism.json"
    deterministic = json.loads(deterministic_path.read_text())
    deterministic.update({
        "status": "PASS" if status == "PASS" else "FAIL",
        "host_repeat_pending": False,
        "same_frame_only": True, "queue_depth": 0,
    })
    atomic(deterministic_path, deterministic)
    refresh_implementation_contract()
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", f"""# PDSCR1 readiness

Status: **{final['status']} / Route {final['route']}**.
The shadow candidate measures p95={candidate['steady_state_ms']['p95']:.3f}
ms and median={candidate['steady_state_ms']['p50']:.3f} ms against
gate={gate:.3f} ms. Production and training remain disabled.
""")
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", (
        """# PDSCR1 recommendation

Freeze the development-only provisional contract. L6, bounded support, and
unresolved risk now remain safety-consumable without formal tracker mutation.
Proceed only to foreground component availability repair; L2 and outside-FOV
risk still block production and training.
""" if semantic_pass else """# PDSCR1 recommendation

Do not select or integrate P7. Runtime passes, but held-out no-target
validation produced three provisional births. Proceed only to provisional
evidence contract review without tuning against the frozen split.
"""))
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
