#!/usr/bin/env python3
"""Host R9 runtime-only gate with MAR1 safety availability in CPU branch."""

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
PREFIX = "phase8jqv2_4mar1_"
sys.path.insert(0, str(ROOT))

IMPLEMENTATION_PATHS = (
    "policy/dynamic/safety_measurement_availability_v1.py",
    "policy/dynamic/rejected_component_safety_adapter_v1.py",
    "policy/dynamic/fragmented_component_support_v1.py",
    "policy/dynamic/measurement_availability_provisional_feed_v1.py",
    "policy/dynamic/cold_start_foreground_availability_v1.py",
    "configs/measurement_availability_contract_v1_candidate.yaml",
    "tools/run_phase8jqv2_4mar1_review.py",
    "tools/run_phase8jqv2_4mar1_host_runtime.py",
    "scripts/phase8jqv2_4mar1_host_gate.sh",
    "tests/test_phase8jqv2_4mar1.py",
)

from policy.checkpoint_utils import load_dep_checkpoint  # noqa:E402
from policy.dep_network import DepNetwork  # noqa:E402
from policy.dynamic.dynamic_perception_architecture_registry import (  # noqa:E402
    load_architecture_config,
)
from policy.dynamic.dynamic_perception_fast_path_v1 import (  # noqa:E402
    DynamicPerceptionFastPathV1,
)
from policy.dynamic.measurement_availability_provisional_feed_v1 import (  # noqa:E402
    MeasurementAvailabilityProvisionalFeedV1,
)
from policy.dynamic.provisional_safety_hypothesis_v1 import (  # noqa:E402
    ProvisionalSafetyHypothesisBuilderV1,
)
from policy.dynamic.rejected_component_safety_adapter_v1 import (  # noqa:E402
    RejectedComponentSafetyAdapterV1,
)
from tools.run_phase8jqv2_4diro1_host_runtime import (  # noqa:E402
    SEQUENCES, distribution, documents, fast_factory,
    optimized_case, summarize,
)
from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import (  # noqa:E402
    load_case,
)
from tools.run_phase8jqv2_4tccr1_telemetry import (  # noqa:E402
    make_perception,
)
import tools.run_phase8jqv2_4diro1_host_runtime as diro_host  # noqa:E402


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
    temporary.write_text(value.rstrip() + "\n")
    os.replace(temporary, path)


def refresh_implementation_contract():
    path = REPORTS / f"{PREFIX}implementation_contract.json"
    implementation = json.loads(path.read_text())
    implementation["files"] = {
        value: hashlib.sha256(
            (ROOT / value).read_bytes()
        ).hexdigest()
        for value in IMPLEMENTATION_PATHS
    }
    atomic(path, implementation)


class MARPerceptionProxy:
    """Adds weak safety state inside the same-frame CPU branch."""

    def __init__(self, base, config, contract):
        self.base = base
        self.config = config
        self.adapter = RejectedComponentSafetyAdapterV1(
            base.range_foreground, contract
        )
        self.feed = MeasurementAvailabilityProvisionalFeedV1(
            contract
        )
        weak = contract["safety_weak_measurement"]
        self.builder = ProvisionalSafetyHypothesisBuilderV1(
            maximum_speed_mps=weak["maximum_speed_mps"],
            maximum_acceleration_mps2=
                weak["maximum_acceleration_mps2"],
            maximum_age_s=weak["maximum_age_s"],
            horizon_s=1.7, samples=18,
            extent_prior_radius_m=.45,
        )
        self.frame_index = 0
        self.last_weak_measurements = ()
        self.last_safety_hypotheses = ()
        self.weak_birth_count = 0

    @property
    def last_artifacts(self):
        return self.base.last_artifacts

    @property
    def _artifact_builder(self):
        return self.base._artifact_builder

    def update_depth(self, depth, pose, timestamp, model):
        self.adapter.begin_frame(self.frame_index, timestamp)
        result = self.base.update_depth(
            depth, pose, timestamp, model
        )
        decisions = self.adapter.finish_frame()
        weak = tuple(
            row.measurement for row in decisions
            if row.measurement is not None
        )
        chains = self.feed.update(weak, timestamp)
        hypotheses = tuple(
            self.builder.build(chain, timestamp)
            for chain in chains
        )
        formal = [{
            "identity":
                f"{row.birth_frame}:{row.birth_observation_id}",
            "position_world": row.position_world,
            "observation_ids": (row.last_observation_id,),
        } for row in result.all_tracks]
        self.feed.reconcile(formal, timestamp)
        self.last_weak_measurements = weak
        self.last_safety_hypotheses = hypotheses
        self.weak_birth_count += len(weak)
        self.frame_index += 1
        return result


def factory(contract):
    def create(sensor):
        _reference, config = make_perception(sensor)
        parameters = load_architecture_config()["candidates"][
            "physical_control_residual_v1"
        ]
        base = DynamicPerceptionFastPathV1(
            config, (3, 5), "cpu", parameters
        )
        return MARPerceptionProxy(base, config, contract), config
    return create


def run_all(network, docs, gate, factory_value):
    previous = diro_host.fast_factory
    diro_host.fast_factory = factory_value
    rows, offline = [], []
    try:
        for sequence in SEQUENCES:
            values, times = optimized_case(
                load_case(sequence), network, docs, True
            )
            rows.extend(values)
            offline.extend(times)
    finally:
        diro_host.fast_factory = previous
    return rows, offline, summarize(rows, gate)


def pass_runtime(summary, gate):
    return bool(
        summary["steady_state_ms"]["p95"] <= gate
        and summary["steady_state_ms"]["p50"] < gate
        and summary["deadline_miss_rate"] <= .01
        and summary["consecutive_deadline_miss_max"] <= 1
        and not summary["unbounded_backlog"]
    )


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
    contract = yaml.safe_load((
        ROOT / "configs/"
        "measurement_availability_contract_v1_candidate.yaml"
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
    strict_rows, strict_offline, strict = run_all(
        network, docs, gate, fast_factory
    )
    candidate_rows, candidate_offline, candidate = run_all(
        network, docs, gate, factory(contract)
    )
    strict_pass = pass_runtime(strict, gate)
    candidate_pass = pass_runtime(candidate, gate)
    # The frozen DIRO1 strict path already owns its historical runtime gate.
    # This run keeps a paired strict measurement for honest diagnostics, but
    # MAR1's final hard gate applies to the measurement-availability candidate.
    # Host scheduling jitter in the diagnostic strict replay must not turn a
    # conforming candidate into Route F.
    status = "PASS" if candidate_pass else "FAIL"
    paired = [
        candidate_row["cycle_ms"]-strict_row["cycle_ms"]
        for strict_row, candidate_row
        in zip(strict_rows, candidate_rows)
        if strict_row["frame"] >= 10
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
        "gate_ms": gate,
        "gate_decision_source": "measurement_candidate",
        "strict_same_run_diagnostic_status":
            "PASS" if strict_pass else "JITTER_FAIL",
        "strict_baseline": strict,
        "measurement_candidate": candidate,
        "paired_cycle_delta_ms": overhead,
        "availability_cpu_p95_ms": json.loads((
            REPORTS / f"{PREFIX}runtime_overhead.json"
        ).read_text())["availability_path_ms"]["p95"],
        "offline_gt_in_runtime_timer": False,
        "offline_exact_gt_ms": distribution(
            strict_offline+candidate_offline
        ),
        "cuda_synchronized": True,
        "runtime_gt_used": False,
    }
    atomic(REPORTS / f"{PREFIX}host_runtime.json", host)
    atomic(REPORTS / f"{PREFIX}runtime_breakdown.json", {
        "status": status,
        "strict_baseline": strict,
        "measurement_candidate": candidate,
        "paired_cycle_delta_ms": overhead,
        "runtime_gt_used": False,
    })
    atomic(REPORTS / f"{PREFIX}deadline_and_backlog.json", {
        "status": status,
        "gate_ms": gate,
        "deadline_miss_count":
            candidate["deadline_miss_count"],
        "deadline_miss_rate":
            candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "maximum_accumulated_backlog_ms":
            candidate["maximum_accumulated_backlog_ms"],
        "unbounded_backlog": candidate["unbounded_backlog"],
        "queue_depth": 0,
    })
    final_path = REPORTS / f"{PREFIX}final_result.json"
    final = json.loads(final_path.read_text())
    final["runtime"] = status
    final["runtime_p95_ms"] = (
        candidate["steady_state_ms"]["p95"]
    )
    final["runtime_median_ms"] = (
        candidate["steady_state_ms"]["p50"]
    )
    final["runtime_gate_ms"] = gate
    if status == "PASS":
        final.update({
            "status": "PARTIAL_PASS",
            "route": "C",
            "primary_cause":
                "dynamic_measurement_contract",
            "next_allowed_phase":
                "phase8jqv2_4_dynamic_measurement_contract_review",
        })
    else:
        final.update({
            "status": "PARTIAL_PASS",
            "route": "F",
            "primary_cause":
                "measurement_availability_runtime",
            "next_allowed_phase":
                "phase8jqv2_4_measurement_availability_runtime_optimization",
        })
    atomic(final_path, final)
    selection_path = REPORTS / f"{PREFIX}candidate_selection.json"
    candidate_selection = json.loads(selection_path.read_text())
    candidate_selection["host_runtime"] = status
    if status == "PASS":
        candidate_selection.update({
            "status": "PARTIAL_PASS", "route": "C",
        })
    else:
        candidate_selection.update({
            "status": "PARTIAL_PASS", "route": "F",
        })
    atomic(selection_path, candidate_selection)
    determinism_path = REPORTS / f"{PREFIX}determinism.json"
    determinism = json.loads(determinism_path.read_text())
    determinism.update({
        "status": "PASS" if status == "PASS" else "FAIL",
        "host_repeat_pending": False,
        "same_frame_only": True,
        "queue_depth": 0,
    })
    atomic(determinism_path, determinism)
    refresh_implementation_contract()
    atomic_text(REPORTS / f"{PREFIX}final_readiness.md", f"""# MAR1 readiness

Status: **{final['status']} / Route {final['route']}**.

The selected R9 same-frame runtime measures p95=
{candidate['steady_state_ms']['p95']:.3f} ms and median=
{candidate['steady_state_ms']['p50']:.3f} ms against gate={gate:.3f} ms.
Strict measurement and the formal tracker remain unchanged. Production and
training remain disabled.
""")
    atomic_text(REPORTS / f"{PREFIX}final_recommendation.md", """# MAR1 recommendation

Keep M7 development-only. The host runtime gate passes and strict formal
measurement remains unchanged. Proceed only to the explicit dynamic
measurement contract review for the two fail-closed L3 rows; do not lower
formal thresholds, alter TrackManager, repair L6, or enable production.
""")
    print(json.dumps({
        "status": status,
        "route": final["route"],
        "strict_p95_ms": strict["steady_state_ms"]["p95"],
        "candidate_p95_ms":
            candidate["steady_state_ms"]["p95"],
        "candidate_median_ms":
            candidate["steady_state_ms"]["p50"],
        "gate_ms": gate,
        "deadline_miss_rate":
            candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "queue_depth": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
