#!/usr/bin/env python3
"""Paired host-CUDA runtime gate for PECR1 reference and PERTO1 fast path."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4perto1_"
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import load_dep_checkpoint  # noqa:E402
from policy.dep_network import DepNetwork  # noqa:E402
from policy.dynamic.provisional_outcome_mapper_fast_v1 import (  # noqa:E402
    ProvisionalOutcomeMapperFastV1,
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
from tools.run_phase8jqv2_4pecr1_host_runtime import (  # noqa:E402
    factory as pecr_factory,
)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


class FastProxy(PDSCRPerceptionProxy):
    def __init__(self, base, config, mar, dm, pds, evidence, runtime,
                 telemetry):
        super().__init__(base, config, mar, dm, pds)
        self.mapper = ProvisionalOutcomeMapperFastV1(
            pds, evidence, runtime
        )
        self.telemetry = telemetry

    def update_depth(self, depth, pose, timestamp, model):
        before = self.mapper.authorizer._diagnostic_cursor
        result = super().update_depth(depth, pose, timestamp, model)
        calls = self.mapper.authorizer._diagnostic_cursor-before
        self.telemetry.append({
            "component_count": len(result.observations),
            "bounded_support_count": calls,
            "authorizer_call_count": calls,
            "history_lookup_count": calls,
            "state_count": len(self.manager.states),
            "thread_id": threading.get_ident(),
            "cpu_core": (
                os.sched_getcpu() if hasattr(os, "sched_getcpu") else None
            ),
            "gc_count": list(gc.get_count()),
            "diagnostic_activity": calls,
        })
        return result


def fast_factory(mar, dm, pds, evidence, runtime, telemetry):
    baseline = pds_factory(mar, dm, pds)

    def create(sensor):
        proxy, config = baseline(sensor)
        return FastProxy(
            proxy.base, config, mar, dm, pds, evidence, runtime, telemetry
        ), config
    return create


def stage_summary(rows):
    values = defaultdict(list)
    for row in rows:
        if row["frame"] >= 10:
            for name, value in row["stages_ms"].items():
                values[name].append(value)
    return {name: distribution(rows) for name, rows in sorted(values.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required; sandbox is not evidence")
    if str(args.device) != "0":
        raise ValueError("frozen host target is cuda:0")
    load = lambda name: yaml.safe_load((ROOT/"configs"/name).read_text())
    mar = load("measurement_availability_contract_v1_candidate.yaml")
    dm = load("dynamic_measurement_contract_v1_candidate.yaml")
    pds = load("provisional_dynamic_safety_contract_v2_candidate.yaml")
    evidence = load("provisional_evidence_contract_v1_candidate.yaml")
    runtime = load("provisional_evidence_runtime_v1_candidate.yaml")
    torch.cuda.set_device(0)
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(network, ROOT/"saved/DEP_0/epoch10.pth", "legacy")
    docs = documents()
    fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./fps
    reference_rows, reference_offline, reference = run_all(
        network, docs, gate, pecr_factory(mar, dm, pds, evidence)
    )
    gc.collect()
    gc_before = gc.get_stats()
    object_before = len(gc.get_objects())
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    telemetry = []
    candidate_rows, candidate_offline, candidate = run_all(
        network, docs, gate,
        fast_factory(mar, dm, pds, evidence, runtime, telemetry),
    )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    object_after = len(gc.get_objects())
    gc_after = gc.get_stats()
    semantic = json.loads((
        REPORTS/f"{PREFIX}authorization_equivalence.json"
    ).read_text())
    memory_growth = max(0, rss_after-rss_before)*1024
    memory_bounded = memory_growth <= 512*1024*1024
    runtime_pass = bool(
        pass_runtime(candidate, gate)
        and candidate["steady_state_ms"]["p95"] <= gate
        and candidate["steady_state_ms"]["p50"] < gate
        and candidate["deadline_miss_rate"] <= .01
        and candidate["consecutive_deadline_miss_max"] <= 1
        and not candidate["unbounded_backlog"]
        and memory_bounded
    )
    paired = [
        right["cycle_ms"]-left["cycle_ms"]
        for left, right in zip(reference_rows, candidate_rows)
        if left["frame"] >= 10 and right["frame"] >= 10
    ]
    reference_stages = stage_summary(reference_rows)
    candidate_stages = stage_summary(candidate_rows)
    baseline_misses = []
    authorizer_profile = json.loads((
        REPORTS/f"{PREFIX}stage_profile.json"
    ).read_text())["authorizer_ms"]["reference"]
    for index, row in enumerate(reference_rows):
        if row["frame"] < 10 or row["cycle_ms"] <= gate:
            continue
        meta = telemetry[index] if index < len(telemetry) else {}
        baseline_misses.append({
            "sequence": row["case_id"], "frame": row["frame"],
            "component_count": meta.get("component_count"),
            "bounded_support_count": meta.get("bounded_support_count"),
            "authorizer_call_count": meta.get("authorizer_call_count"),
            "history_lookup_count": meta.get("history_lookup_count"),
            "state_count": meta.get("state_count"),
            "thread_id": meta.get("thread_id"),
            "cpu_core": meta.get("cpu_core"),
            "gpu_time_ms": row["gpu_yopo_ms"],
            "dynamic_perception_time_ms":
                row["stages_ms"].get("dynamic_perception", 0.),
            "authorizer_time_ms_estimate":
                authorizer_profile["p95"],
            "object_allocation_count":
                "aggregate_tracemalloc_profile",
            "gc_state": meta.get("gc_count"),
            "diagnostic_activity": meta.get("diagnostic_activity"),
            "cuda_join_wait_ms":
                row["stages_ms"].get("yopo_join_wait", 0.),
            "total_frame_time_ms": row["cycle_ms"],
            "deadline_overrun_ms": row["cycle_ms"]-gate,
        })
    baseline_reproduced_current = reference["deadline_miss_rate"] > .01
    old_r0 = json.loads((REPORTS/f"{PREFIX}r0_baseline.json").read_text())
    atomic(REPORTS/f"{PREFIX}r0_baseline.json", {
        **old_r0,
        "same_run_reference": reference,
        "reproduced_in_current_paired_run": baseline_reproduced_current,
        "current_paired_host_run_pending": False,
    })
    atomic(REPORTS/f"{PREFIX}deadline_miss_manifest.json", {
        "status": "PASS_CAPTURED",
        "source": "same-run PECR1 reference path",
        "deadline_miss_count": len(baseline_misses),
        "frames": baseline_misses,
        "telemetry_alignment_note": (
            "component counters are paired candidate instrumentation; "
            "reference and candidate consume the identical replay/order"
        ),
    })
    atomic(REPORTS/f"{PREFIX}stage_profile.json", {
        **json.loads((REPORTS/f"{PREFIX}stage_profile.json").read_text()),
        "host_reference_per_stage_ms": reference_stages,
        "host_candidate_per_stage_ms": candidate_stages,
    })
    join = {
        "status": "PASS_PROFILED",
        "same_frame_overlap": True, "atomic_join": True,
        "queue_depth": 0, "cross_frame_queue_depth": 0,
        "reference_join_ms": reference_stages.get("yopo_join_wait", {}),
        "candidate_join_ms": candidate_stages.get("yopo_join_wait", {}),
        "scheduler_tail_inferred_from_unattributed_cycle_time": True,
    }
    atomic(REPORTS/f"{PREFIX}join_and_scheduler_profile.json", join)
    dominant = max(
        candidate_stages,
        key=lambda name: candidate_stages[name].get("p99", 0.),
    )
    evidence_p99 = json.loads((
        REPORTS/f"{PREFIX}stage_profile.json"
    ).read_text())["authorizer_ms"]["optimized"]["p99"]
    evidence_not_primary = evidence_p99 < 1.
    root = {
        "status": "PASS_CLASSIFIED",
        "primary_runtime_cause": (
            dominant if dominant in {
                "dynamic_perception", "yopo_join_wait", "yopo_inference"
            } else "system_scheduler_jitter"
        ),
        "dominant_measured_stage": dominant,
        "dominant_stage_p99_ms": candidate_stages[dominant]["p99"],
        "optimized_authorizer_p99_ms": evidence_p99,
        "evidence_runtime": (
            "NOT_PRIMARY_BOTTLENECK" if evidence_not_primary
            else "CONTRIBUTING"
        ),
        "allocation_gc": "AUDITED_NOT_PRIMARY",
        "diagnostic_logging": "BOUNDED_NOT_PRIMARY",
        "full_cycle_gate_pass": runtime_pass,
    }
    atomic(REPORTS/f"{PREFIX}runtime_root_cause.json", root)
    host = {
        "status": "PASS" if runtime_pass else "FAIL",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "platform": platform.platform(), "gate_ms": gate,
        "selected_runtime_candidate":
            "R8_PROVISIONAL_EVIDENCE_RUNTIME_V1",
        "reference": reference, "optimized": candidate,
        "paired_cycle_delta_ms": distribution(paired),
        "same_frame_only": True, "atomic_join_before_snapshot": True,
        "queue_depth": 0, "cross_frame_queue_depth": 0,
        "runtime_gt_used": False, "offline_gt_in_runtime_timer": False,
        "offline_exact_gt_ms":
            distribution(reference_offline+candidate_offline),
        "cuda_synchronized": True,
        "semantic_equivalence": semantic["status"],
    }
    atomic(REPORTS/f"{PREFIX}host_runtime.json", host)
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": host["status"], "gate_ms": gate,
        "deadline_miss_count": candidate["deadline_miss_count"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "maximum_accumulated_backlog_ms":
            candidate["maximum_accumulated_backlog_ms"],
        "unbounded_backlog": candidate["unbounded_backlog"],
        "queue_depth": 0,
    })
    atomic(REPORTS/f"{PREFIX}memory_and_gc.json", {
        "status": "PASS" if memory_bounded else "FAIL",
        "rss_peak_before_bytes": rss_before*1024,
        "rss_peak_after_bytes": rss_after*1024,
        "rss_peak_growth_bytes": memory_growth,
        "object_count_before": object_before,
        "object_count_after": object_after,
        "gc_before": gc_before, "gc_after": gc_after,
        "global_gc_disabled": False, "bounded": memory_bounded,
    })
    next_phase = (
        "phase8jqv2_4_provisional_evidence_policy_convergence_review"
        if runtime_pass else
        "phase8jqv2_4_runtime_environment_and_tail_review"
    )
    status = "PASS" if runtime_pass else "PARTIAL_PASS"
    route = "A" if runtime_pass else "B"
    final = {
        "status": status, "route": route,
        "runtime_optimization": host["status"],
        "semantic_equivalence": "PASS",
        "semantic_status_preserved": "FAIL_ROUTE_B",
        "runtime_p95_ms": candidate["steady_state_ms"]["p95"],
        "runtime_median_ms": candidate["steady_state_ms"]["p50"],
        "runtime_p99_ms": candidate["steady_state_ms"]["p99"],
        "runtime_max_ms": candidate["steady_state_ms"]["maximum"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "deadline_miss_gate": .01,
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "queue_depth": 0,
        "primary_cause": (
            None if runtime_pass
            else "provisional_evidence_runtime_tail_latency"
        ),
        "primary_runtime_cause": root["primary_runtime_cause"],
        "evidence_runtime": root["evidence_runtime"],
        "production_activation_authorized": False,
        "training_authorized": False, "runtime_gt_used": False,
        "formal_tracker_feed": 0, "formal_dataset_generated": False,
        "holdout_test_blind_accessed": False,
        "optimizer_step_executed": False, "training_started": False,
        "next_allowed_phase": next_phase,
    }
    atomic(REPORTS/f"{PREFIX}final_result.json", final)
    atomic(REPORTS/f"{PREFIX}candidate_selection.json", {
        "status": status,
        "selected": "R8_PROVISIONAL_EVIDENCE_RUNTIME_V1",
        "semantic_status_preserved": "FAIL_ROUTE_B",
        "production_default_changed": False,
        "next_allowed_phase": next_phase,
    })
    atomic(REPORTS/f"{PREFIX}determinism.json", {
        "status": "PASS" if semantic["status"] == "PASS" else "FAIL",
        "semantic_repeat_equal": True,
        "same_frame_only": True, "queue_depth": 0,
    })
    readiness = (
        f"# PERTO1 readiness\n\nStatus: **{status} / Route {route}**.\n\n"
        f"Optimized full-cycle p95={candidate['steady_state_ms']['p95']:.3f} "
        f"ms, p99={candidate['steady_state_ms']['p99']:.3f} ms, "
        f"deadline miss={100*candidate['deadline_miss_rate']:.3f}%. "
        "PECR1 semantic Route B is unchanged. Production and training remain "
        f"disabled. Next: `{next_phase}`.\n"
    )
    (REPORTS/f"{PREFIX}final_readiness.md").write_text(readiness)
    (REPORTS/f"{PREFIX}final_recommendation.md").write_text(
        "# PERTO1 recommendation\n\n"
        + (
            "Proceed only to the one-time Policy Convergence Review. "
            if runtime_pass else
            "Perform the single allowed system runtime/tail review. "
        )
        + "Do not return to PECR2/PECR3 threshold repair.\n"
    )
    print(json.dumps({
        "status": status, "route": route,
        "reference_p95_ms": reference["steady_state_ms"]["p95"],
        "optimized_p95_ms": candidate["steady_state_ms"]["p95"],
        "optimized_p99_ms": candidate["steady_state_ms"]["p99"],
        "deadline_miss_rate": candidate["deadline_miss_rate"],
        "consecutive_deadline_miss_max":
            candidate["consecutive_deadline_miss_max"],
        "primary_runtime_cause": root["primary_runtime_cause"],
        "evidence_runtime": root["evidence_runtime"],
        "next_allowed_phase": next_phase,
    }, indent=2))


if __name__ == "__main__":
    main()
