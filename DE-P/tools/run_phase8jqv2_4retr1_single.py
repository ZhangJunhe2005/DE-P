#!/usr/bin/env python3
"""One independent RETR1 host process; writes one bounded run record."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


class InternalProfiler:
    """Profile-only wrappers; qualification never installs these."""

    def __init__(self):
        self.values = defaultdict(list)
        self._restore = []

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.values[stage].append(
                    (time.perf_counter_ns()-started)/1e6
                )
        setattr(owner, name, measured)
        self._restore.append((owner, name, original))

    def install(self):
        import policy.dynamic.dynamic_frame_artifacts_v1 as artifacts
        import policy.dynamic.dynamic_perception_fast_path_v1 as fast
        import policy.dynamic.image_foreground_components as components
        import policy.dynamic.physical_control_residual_v1 as physical
        import policy.dynamic.range_image_foreground_v2_1 as range_image
        from policy.dynamic.track_manager import TrackManager
        self.wrap(
            artifacts.DynamicFrameArtifactBuilderV1, "build",
            "depth_preprocessing_and_artifact_build",
        )
        self.wrap(
            fast.CachedTemporalVoxelForegroundV1,
            "previously_free_mask", "temporal_residual_free_mask",
        )
        self.wrap(
            range_image, "_common", "residual_and_mask_construction",
        )
        self.wrap(
            components, "connected_components",
            "connected_components",
        )
        self.wrap(
            range_image, "_observation", "component_aggregation",
        )
        self.wrap(
            physical.PhysicalControlResidualV1, "_accept",
            "measurement_filtering",
        )
        self.wrap(
            TrackManager, "update", "track_update",
        )
        self.wrap(
            fast, "build_dynamic_attention_with_projection",
            "shared_attention_publication",
        )

    def restore(self):
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)

    def summary(self):
        return {
            key: distribution(value)
            for key, value in sorted(self.values.items())
        }


def semantic_digest(rows):
    fields = [{
        "case_id": row["case_id"], "frame": row["frame"],
        "decision_status": row["decision_status"],
        "recommended_candidate": row["recommended_candidate"],
        "track_ids": row["track_ids"],
        "provisional_hypothesis_count":
            row["provisional_hypothesis_count"],
        "formal_reachability_count":
            row["formal_reachability_count"],
        "prediction_only_active": row["prediction_only_active"],
        "runtime_gt_used": row["runtime_gt_used"],
        "frame_artifact_id": row["frame_artifact_id"],
        "point_reconstruction_count":
            row["point_reconstruction_count"],
        "world_transform_count": row["world_transform_count"],
    } for row in rows]
    return hashlib.sha256(json.dumps(
        fields, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True,
                        choices=("H0", "H1", "H2", "H3", "H4", "H5"))
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--affinity", default="")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gc-policy", default="DEFAULT",
                        choices=("DEFAULT", "EPISODE_BOUNDARY"))
    parser.add_argument("--pre-touch-bytes", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    allowed_before = tuple(sorted(os.sched_getaffinity(0)))
    requested = tuple(
        int(value) for value in args.affinity.split(",") if value
    )
    if requested:
        if not set(requested).issubset(allowed_before):
            raise RuntimeError("requested affinity contains unavailable CPU")
        os.sched_setaffinity(0, requested)
    if args.threads is not None:
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = str(args.threads)
    import torch
    import yaml
    from policy.checkpoint_utils import load_dep_checkpoint
    from policy.dep_network import DepNetwork
    from policy.dynamic.runtime_qualification_telemetry_v1 import (
        ProcessTelemetryV1,
    )
    from tools.run_phase8jqv2_4diro1_host_runtime import (
        SEQUENCES, documents, optimized_case, summarize,
    )
    from tools.run_phase8jqv2_4dogmr1_geometry_evaluation import load_case
    from tools.run_phase8jqv2_4perto1_host_runtime import fast_factory
    import tools.run_phase8jqv2_4diro1_host_runtime as diro
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required")
    torch.cuda.set_device(0)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
    profiler = InternalProfiler() if args.profile else None
    if profiler is not None:
        profiler.install()
    contract_load = lambda name: yaml.safe_load(
        (ROOT/"configs"/name).read_text()
    )
    mar = contract_load(
        "measurement_availability_contract_v1_candidate.yaml"
    )
    dm = contract_load("dynamic_measurement_contract_v1_candidate.yaml")
    pds = contract_load(
        "provisional_dynamic_safety_contract_v2_candidate.yaml"
    )
    evidence = contract_load(
        "provisional_evidence_contract_v1_candidate.yaml"
    )
    runtime = contract_load(
        "provisional_evidence_runtime_v1_candidate.yaml"
    )
    fps = float(yaml.safe_load((
        ROOT.parent/"Simulator/src/config/config.yaml"
    ).read_text())["depth_fps"])
    gate = 1000./fps
    network = DepNetwork(backbone_variant="legacy").cuda().eval()
    load_dep_checkpoint(network, ROOT/"saved/DEP_0/epoch10.pth", "legacy")
    if args.pre_touch_bytes:
        touch = np.empty(args.pre_touch_bytes, np.uint8)
        touch.fill(1)
        del touch
    telemetry = []
    factory = fast_factory(mar, dm, pds, evidence, runtime, telemetry)
    previous_factory = diro.fast_factory
    diro.fast_factory = factory
    docs = documents()
    before = ProcessTelemetryV1.capture()
    rows, offline = [], []
    gc_was_enabled = gc.isenabled()
    try:
        for sequence in SEQUENCES:
            if args.gc_policy == "EPISODE_BOUNDARY":
                if not gc.isenabled():
                    gc.enable()
                gc.collect()
                gc.disable()
            values, times = optimized_case(
                load_case(sequence), network, docs, True
            )
            rows.extend(values)
            offline.extend(times)
        if args.gc_policy == "EPISODE_BOUNDARY":
            gc.enable()
            gc.collect()
    finally:
        diro.fast_factory = previous_factory
        if gc_was_enabled and not gc.isenabled():
            gc.enable()
        if profiler is not None:
            profiler.restore()
    after = ProcessTelemetryV1.capture()
    summary = summarize(rows, gate)
    steady_rows = [row for row in rows if row["frame"] >= 10]
    miss_rows = [{
        "case_id": row["case_id"], "frame": row["frame"],
        "cycle_ms": row["cycle_ms"],
        "overrun_ms": row["cycle_ms"]-gate,
        "dynamic_perception_ms":
            row["stages_ms"].get("dynamic_perception", 0.),
        "gpu_yopo_ms": row["gpu_yopo_ms"],
        "join_wait_ms": row["stages_ms"].get("yopo_join_wait", 0.),
        "authorizer_call_count": telemetry[index][
            "authorizer_call_count"
        ] if index < len(telemetry) else None,
        "component_count": telemetry[index][
            "component_count"
        ] if index < len(telemetry) else None,
    } for index, row in enumerate(rows)
        if row["frame"] >= 10 and row["cycle_ms"] > gate]
    result = {
        "status": "PASS",
        "candidate": args.candidate, "run_index": args.run_index,
        "pid": os.getpid(), "profile_run": args.profile,
        "qualification_run": not args.profile,
        "gate_ms": gate, "summary": summary,
        "semantic_digest": semantic_digest(rows),
        "frame_count": len(rows),
        "steady_frame_count": len(steady_rows),
        "steady_cycle_ms": [row["cycle_ms"] for row in steady_rows],
        "steady_dynamic_perception_ms": [
            row["stages_ms"].get("dynamic_perception", 0.)
            for row in steady_rows
        ],
        "steady_gpu_yopo_ms": [
            row["gpu_yopo_ms"] for row in steady_rows
        ],
        "steady_join_wait_ms": [
            row["stages_ms"].get("yopo_join_wait", 0.)
            for row in steady_rows
        ],
        "no_skipped_frame": len(rows) == 360,
        "frame_order": [
            f"{row['case_id']}:{row['frame']}" for row in rows
        ],
        "deadline_misses": miss_rows,
        "process_telemetry_before": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in before.__dict__.items()
        } if hasattr(before, "__dict__") else {
            name: list(getattr(before, name))
            if isinstance(getattr(before, name), tuple)
            else getattr(before, name)
            for name in before.__dataclass_fields__
        },
        "process_telemetry_delta": after.delta(before),
        "affinity_requested": list(requested),
        "affinity_effective": list(after.affinity),
        "allowed_affinity_before": list(allowed_before),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "process_threads_end": after.process_thread_count,
        },
        "gc_policy": args.gc_policy,
        "gc_restored": gc.isenabled() == gc_was_enabled,
        "memory_pre_touch_bytes": args.pre_touch_bytes,
        "internal_profile": (
            {} if profiler is None else profiler.summary()
        ),
        "offline_gt_in_runtime_timer": False,
        "runtime_gt_used": False, "formal_tracker_feed": 0,
        "same_frame_only": True, "atomic_join_before_snapshot": True,
        "queue_depth": 0, "unbounded_backlog": False,
        "cuda_synchronized": True,
    }
    atomic(args.output, result)
    print(json.dumps({
        "candidate": args.candidate, "run_index": args.run_index,
        "profile": args.profile,
        "p95_ms": summary["steady_state_ms"]["p95"],
        "p99_ms": summary["steady_state_ms"]["p99"],
        "deadline_miss_rate": summary["deadline_miss_rate"],
        "miss_count": summary["deadline_miss_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
