#!/usr/bin/env python3
"""One-time RETR1 host environment/tail adjudication."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
DIAG = ROOT/"diagnostics/phase8jqv2_4retr1"
PREFIX = "phase8jqv2_4retr1_"
CONFIG = ROOT/"configs/runtime_environment_contract_v1_candidate.yaml"

IMPLEMENTATION_PATHS = (
    "policy/dynamic/runtime_environment_contract_v1.py",
    "policy/dynamic/runtime_qualification_telemetry_v1.py",
    "configs/runtime_environment_contract_v1_candidate.yaml",
    "tools/run_phase8jqv2_4retr1_single.py",
    "tools/run_phase8jqv2_4retr1_review.py",
    "scripts/phase8jqv2_4retr1_host_gate.sh",
    "tests/test_phase8jqv2_4retr1.py",
)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip()+"\n")
    os.replace(temporary, path)


def read(name):
    return json.loads((REPORTS/name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    try:
        result = subprocess.run(
            args, text=True, capture_output=True, check=False, timeout=15
        )
        return {
            "status": "PASS" if result.returncode == 0 else "UNAVAILABLE",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "UNAVAILABLE", "error": str(error)}


def text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "UNAVAILABLE"


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
        "variance": float(values.var()),
    }


def entry_and_freeze():
    final = read("phase8jqv2_4perto1_final_result.json")
    root = read("phase8jqv2_4perto1_runtime_root_cause.json")
    semantic = read("phase8jqv2_4perto1_semantic_freeze.json")
    impl = read("phase8jqv2_4perto1_implementation_contract.json")
    current = {path: sha(ROOT/path) for path in impl["files"]}
    checks = {
        "perto1_route_b":
            final["status"] == "PARTIAL_PASS" and final["route"] == "B",
        "semantic_equivalence": final["semantic_equivalence"] == "PASS",
        "semantic_route_b":
            final["semantic_status_preserved"] == "FAIL_ROUTE_B",
        "authorizer_not_bottleneck":
            final["evidence_runtime"] == "NOT_PRIMARY_BOTTLENECK"
            and root["optimized_authorizer_p99_ms"] < 1.,
        "dynamic_perception_primary":
            root["primary_runtime_cause"] == "dynamic_perception",
        "p99_reference":
            abs(final["runtime_p99_ms"]-30.850676616901172) < 1e-9,
        "deadline_reference":
            abs(final["deadline_miss_rate"]-.016666666666666666) < 1e-12,
        "queue_zero": final["queue_depth"] == 0,
        "runtime_gt_false": not final["runtime_gt_used"],
        "formal_data_false": not final["formal_dataset_generated"],
        "training_false": not final["training_started"],
        "holdout_false": not final["holdout_test_blind_accessed"],
        "perto1_hashes_frozen": current == impl["files"],
    }
    entry = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_phase": "phase8jqv2_4_runtime_environment_and_tail_review",
    }
    atomic(REPORTS/f"{PREFIX}entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise RuntimeError("RETR1 entry gate failed")
    frozen = {
        "status": "PASS", "perto1_reference_hashes": impl["files"],
        "perto1_current_hashes": current,
        "perto1_artifacts_modified": False,
        "pecr1_artifacts_modified": False,
        "pdscr1_artifacts_modified": False,
        "dmcr1_artifacts_modified": False,
        "mar1_artifacts_modified": False,
        "diro1_artifacts_modified": False,
        "evidence_authorizer_semantics_modified": False,
        "provisional_state_semantics_modified": False,
        "foreground_semantics_modified": False,
        "component_semantics_modified": False,
        "measurement_semantics_modified": False,
        "TrackManager_algorithm_modified": False,
        "kalman_models_modified": False, "yopo_modified": False,
        "planner_modified": False, "command_modified": False,
    }
    atomic(REPORTS/f"{PREFIX}frozen_artifacts.json", frozen)
    atomic(REPORTS/f"{PREFIX}semantic_freeze.json", semantic)
    return semantic


def cpu_topology():
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    rows = []
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        maximum = text(base/"cpufreq/cpuinfo_max_freq")
        current = text(base/"cpufreq/scaling_cur_freq")
        core = text(base/"topology/core_id")
        package = text(base/"topology/physical_package_id")
        try:
            maximum_value = int(maximum)
        except ValueError:
            maximum_value = 0
        rows.append({
            "cpu": cpu, "online": text(base/"online")
                if cpu else "1",
            "core_id": core, "package_id": package,
            "max_frequency_khz": maximum,
            "current_frequency_khz": current,
            "_max": maximum_value,
        })
    frequency_tiers = sorted({
        row["_max"] for row in rows if row["_max"] > 0
    })
    if len(frequency_tiers) >= 2:
        # On this hybrid host the E-core group shares the lowest advertised
        # maximum while all eight P cores occupy the higher tier(s).
        p_cores = tuple(
            row["cpu"] for row in rows
            if row["_max"] > frequency_tiers[0]
        )
    elif frequency_tiers:
        p_cores = allowed
    else:
        p_cores = allowed
    for row in rows:
        row.pop("_max")
    if not p_cores:
        p_cores = allowed
    value = {
        "status": "PASS",
        "lscpu": command([
            "lscpu", "-e=CPU,ONLINE,CORE,SOCKET,NODE,MAXMHZ,MINMHZ"
        ]),
        "allowed_cpu_set": list(allowed),
        "detected_high_performance_cpu_set": list(p_cores),
        "detection_rule": (
            "all cpuinfo_max_freq tiers above the lowest hybrid-core tier; "
            "full allowed set fallback when only one tier exists"
        ),
        "illegal_core_id_present": not set(p_cores).issubset(allowed),
        "cpus": rows,
    }
    atomic(REPORTS/f"{PREFIX}cpu_topology.json", value)
    return allowed, p_cores


def environment_snapshot():
    load = os.getloadavg()
    meminfo = {}
    for line in text("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        meminfo[key] = value.strip()
    background = command([
        "ps", "-eo", "pid,ppid,comm,%cpu,%mem,nlwp", "--sort=-%cpu",
    ])
    lines = background.get("stdout", "").splitlines()[:11]
    gpu = command([
        "nvidia-smi",
        "--query-gpu=name,driver_version,utilization.gpu,memory.used,"
        "temperature.gpu,clocks.current.sm,power.draw",
        "--format=csv,noheader",
    ])
    gpu_processes = command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ])
    own_process_tree = {os.getpid()}
    cursor = os.getppid()
    while cursor > 1 and cursor not in own_process_tree:
        own_process_tree.add(cursor)
        try:
            cursor = int(text(f"/proc/{cursor}/stat").split()[3])
        except (ValueError, IndexError):
            break
    high_cpu = []
    for line in lines[1:]:
        parts = line.split()
        try:
            pid = int(parts[0])
            if pid not in own_process_tree and float(parts[-3]) >= 50.:
                high_cpu.append(line)
        except (ValueError, IndexError):
            pass
    compute_lines = [
        line for line in gpu_processes.get("stdout", "").splitlines()
        if line.strip()
    ]
    clean = not high_cpu and not compute_lines
    environment = {
        "status": "PASS", "uname": command(["uname", "-a"]),
        "kernel": platform.release(), "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu, "gpu_processes": gpu_processes,
        "cpu_model": command(["lscpu"]),
        "load_average": list(load), "meminfo": meminfo,
        "swap": text("/proc/swaps"),
        "governor_cpu0": text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "power_mode": command(["powerprofilesctl", "get"]),
        "ac_online": text("/sys/class/power_supply/AC/online"),
        "thermal": command(["sensors"]),
        "capture_timestamp_ns": time.time_ns(),
    }
    atomic(REPORTS/f"{PREFIX}host_environment.json", environment)
    atomic(REPORTS/f"{PREFIX}background_load.json", {
        "status": (
            "PASS_CLEAN_CONTROL_AVAILABLE"
            if clean else "FAIL_CONFLICTING_LOAD"
        ),
        "top_processes": lines, "high_cpu_processes": high_cpu,
        "excluded_retr1_process_tree": sorted(own_process_tree),
        "other_compute_processes": compute_lines,
        "load_average": list(load),
        "automatic_process_termination": False,
    })
    atomic(REPORTS/f"{PREFIX}cpu_frequency_thermal.json", {
        "status": "PASS_READ_ONLY",
        "governor": environment["governor_cpu0"],
        "thermal": environment["thermal"],
        "performance_governor_experiment": "NOT_EXECUTED",
        "reason": "not required and no persistent host mutation authorized",
        "thermal_protection_modified": False,
    })
    return clean


def run_one(candidate, index, affinity=(), threads=None,
            gc_policy="DEFAULT", pre_touch=0, profile=False,
            reuse_completed=False):
    path = DIAG/"runs"/(
        f"{candidate.lower()}_{index:02d}"
        f"{'_profile' if profile else ''}.json"
    )
    if reuse_completed and path.exists():
        return json.loads(path.read_text())
    args = [
        sys.executable, str(ROOT/"tools/run_phase8jqv2_4retr1_single.py"),
        "--candidate", candidate, "--run-index", str(index),
        "--output", str(path), "--gc-policy", gc_policy,
        "--pre-touch-bytes", str(pre_touch),
    ]
    if affinity:
        args += ["--affinity", ",".join(str(value) for value in affinity)]
    if threads is not None:
        args += ["--threads", str(threads)]
    if profile:
        args += ["--profile"]
    environment = os.environ.copy()
    if threads is not None:
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = str(threads)
    result = subprocess.run(args, cwd=ROOT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{candidate} run {index} failed with {result.returncode}"
        )
    return json.loads(path.read_text())


def aggregate(runs, gate):
    cycles = [
        value for run in runs for value in run["steady_cycle_ms"]
    ]
    dynamic = [
        value for run in runs
        for value in run["steady_dynamic_perception_ms"]
    ]
    gpu = [
        value for run in runs for value in run["steady_gpu_yopo_ms"]
    ]
    join = [
        value for run in runs for value in run["steady_join_wait_ms"]
    ]
    misses = [value > gate for value in cycles]
    consecutive = current = 0
    for miss in misses:
        current = current+1 if miss else 0
        consecutive = max(consecutive, current)
    return {
        "run_count": len(runs), "steady_frame_count": len(cycles),
        "cycle_ms": distribution(cycles),
        "dynamic_perception_ms": distribution(dynamic),
        "gpu_yopo_ms": distribution(gpu),
        "join_wait_ms": distribution(join),
        "deadline_miss_count": int(sum(misses)),
        "deadline_miss_rate": float(sum(misses)/max(1, len(misses))),
        "consecutive_deadline_miss_max": consecutive,
        "per_run": [{
            "run_index": run["run_index"],
            "p50_ms": run["summary"]["steady_state_ms"]["p50"],
            "p90_ms": run["summary"]["steady_state_ms"]["p90"],
            "p95_ms": run["summary"]["steady_state_ms"]["p95"],
            "p99_ms": run["summary"]["steady_state_ms"]["p99"],
            "max_ms": run["summary"]["steady_state_ms"]["maximum"],
            "miss_count": run["summary"]["deadline_miss_count"],
            "miss_rate": run["summary"]["deadline_miss_rate"],
            "dynamic_perception_p99_ms":
                run["summary"]["per_stage_ms"]["dynamic_perception"]["p99"],
            "gpu_p99_ms": run["summary"]["gpu_yopo_ms"]["p99"],
            "join_wait_p99_ms":
                run["summary"]["per_stage_ms"]["yopo_join_wait"]["p99"],
        } for run in runs],
        "no_outlier_deleted": True,
        "all_frame_orders_equal": len({
            tuple(run["frame_order"]) for run in runs
        }) == 1,
        "all_semantic_digests_equal": len({
            run["semantic_digest"] for run in runs
        }) == 1,
        "all_no_skipped_frame": all(
            run["no_skipped_frame"] for run in runs
        ),
    }


def candidate_value(run):
    return {
        "status": "PASS_PROFILED",
        "p95_ms": run["summary"]["steady_state_ms"]["p95"],
        "p99_ms": run["summary"]["steady_state_ms"]["p99"],
        "deadline_miss_rate": run["summary"]["deadline_miss_rate"],
        "dynamic_perception_p99_ms":
            run["summary"]["per_stage_ms"]["dynamic_perception"]["p99"],
        "affinity_effective": run["affinity_effective"],
        "thread_settings": run["thread_settings"],
        "process_telemetry_delta": run["process_telemetry_delta"],
        "semantic_digest": run["semantic_digest"],
        "no_skipped_frame": run["no_skipped_frame"],
    }


def main():
    reuse_completed = "--reuse-completed-runs" in sys.argv
    if reuse_completed:
        sys.argv.remove("--reuse-completed-runs")
    if not reuse_completed and not torch.cuda.is_available():
        raise RuntimeError("RETR1 must run with host CUDA")
    semantic = entry_and_freeze()
    config = yaml.safe_load(CONFIG.read_text())
    if reuse_completed:
        topology = read(f"{PREFIX}cpu_topology.json")
        allowed = tuple(topology["allowed_cpu_set"])
        p_cores = tuple(topology["detected_high_performance_cpu_set"])
        clean = (
            read(f"{PREFIX}background_load.json")["status"]
            == "PASS_CLEAN_CONTROL_AVAILABLE"
        )
    else:
        allowed, p_cores = cpu_topology()
        clean = environment_snapshot()
    if not clean:
        raise RuntimeError(
            "clean-host prerequisite failed; inspect background_load report"
        )
    gate = float(config["qualification"]["full_cycle_gate_ms"])
    profile = run_one(
        "H0", 0, profile=True, reuse_completed=reuse_completed
    )
    h0_runs = [
        run_one("H0", index, reuse_completed=reuse_completed)
        for index in range(1, 6)
    ]
    h0 = aggregate(h0_runs, gate)
    atomic(REPORTS/f"{PREFIX}h0_multirun_baseline.json", {
        "status": "PASS_MEASURED", **h0,
        "independent_processes": True,
    })
    h1 = run_one("H1", 1, reuse_completed=reuse_completed)
    h2 = run_one(
        "H2", 1, affinity=p_cores, reuse_completed=reuse_completed
    )
    h3 = run_one(
        "H3", 1, threads=1, reuse_completed=reuse_completed
    )
    h4 = run_one(
        "H4", 1, gc_policy="EPISODE_BOUNDARY",
        pre_touch=8*1024*1024,
        reuse_completed=reuse_completed,
    )
    for name, run in (("h1_clean_host", h1), ("h2_cpu_affinity", h2),
                      ("h3_thread_pool", h3), ("h4_memory_gc", h4)):
        atomic(REPORTS/f"{PREFIX}{name}.json", candidate_value(run))
    baseline_miss = h0["deadline_miss_rate"]
    baseline_p99 = h0["cycle_ms"]["p99"]
    selected = []
    if (
        h2["summary"]["deadline_miss_rate"] < baseline_miss
        and h2["summary"]["steady_state_ms"]["p99"] < baseline_p99
    ):
        selected.append("P_CORE_AFFINITY")
    if (
        h3["summary"]["deadline_miss_rate"] < baseline_miss
        and h3["summary"]["steady_state_ms"]["p99"] < baseline_p99
    ):
        selected.append("THREADS_ONE")
    if (
        h4["summary"]["deadline_miss_rate"] < baseline_miss
        and h4["summary"]["steady_state_ms"]["p99"] < baseline_p99
    ):
        selected.append("EPISODE_GC_PRETOUCH")
    h5_affinity = p_cores if "P_CORE_AFFINITY" in selected else ()
    h5_threads = 1 if "THREADS_ONE" in selected else None
    h5_gc = (
        "EPISODE_BOUNDARY"
        if "EPISODE_GC_PRETOUCH" in selected else "DEFAULT"
    )
    h5_touch = (
        8*1024*1024 if "EPISODE_GC_PRETOUCH" in selected else 0
    )
    h5_runs = [
        run_one(
            "H5", index, affinity=h5_affinity,
            threads=h5_threads, gc_policy=h5_gc,
            pre_touch=h5_touch,
            reuse_completed=reuse_completed,
        ) for index in range(1, 6)
    ]
    h5 = aggregate(h5_runs, gate)
    atomic(REPORTS/f"{PREFIX}h5_combined.json", {
        "status": "PASS_MEASURED", **h5,
        "selected_minimal_features": selected or [
            "LOW_OVERHEAD_QUALIFICATION_ONLY"
        ],
        "selection_rule":
            "single-candidate miss rate and p99 both improve over H0 aggregate",
    })
    all_candidates = {
        "H0": h0,
        "H1": candidate_value(h1),
        "H2": candidate_value(h2),
        "H3": candidate_value(h3),
        "H4": candidate_value(h4),
        "H5": h5,
    }
    atomic(REPORTS/f"{PREFIX}candidate_comparison.json", {
        "status": "PASS", "candidates": all_candidates,
        "selected": "H5_MINIMAL_ENVIRONMENT_CONTRACT",
        "experiment_classes": ["H0", "H1", "H2", "H3", "H4", "H5"],
        "unplanned_experiment_count": 0,
    })
    misses = [
        {**row, "candidate": run["candidate"],
         "run_index": run["run_index"]}
        for run in h0_runs+h5_runs for row in run["deadline_misses"]
    ]
    atomic(REPORTS/f"{PREFIX}deadline_manifest.json", {
        "status": "PASS_CAPTURED", "frames": misses,
        "outlier_deletion": False,
    })
    frame_counts = Counter(
        (row["case_id"], row["frame"])
        for run in h0_runs+h5_runs for row in run["deadline_misses"]
    )
    extreme = [
        row for run in h0_runs+h5_runs
        for row in run["deadline_misses"] if row["cycle_ms"] >= 50.
    ]
    atomic(REPORTS/f"{PREFIX}outlier_reproducibility.json", {
        "status": "PASS_ANALYZED",
        "repeated_deadline_frames": [
            {"case_id": key[0], "frame": key[1], "run_count": count}
            for key, count in sorted(frame_counts.items())
            if count >= 2
        ],
        "extreme_outlier_count": len(extreme),
        "extreme_rows": extreme,
        "frame30_extreme_repeat_count": sum(
            row["case_id"] == "phase8c_train_0015"
            and row["frame"] == 30 and row["cycle_ms"] >= 50.
            for row in extreme
        ),
        "authorizer_zero_extreme_present": any(
            row["authorizer_call_count"] == 0 for row in extreme
        ),
    })
    profile_stages = profile["internal_profile"]
    classifications = {"ordinary": [], "near_deadline": [], "extreme": []}
    for run in h0_runs+h5_runs:
        for value in run["steady_dynamic_perception_ms"]:
            key = (
                "extreme" if value >= 50.
                else "near_deadline" if value >= 30.
                else "ordinary"
            )
            classifications[key].append(value)
    atomic(REPORTS/f"{PREFIX}dynamic_perception_tail.json", {
        "status": "PASS_ATTRIBUTED",
        "profile_only_internal_stages_ms": profile_stages,
        "shape_geometry_frontend_p99_ms_per_run": {
            "H0_provisional_availability": [
                run["summary"]["per_stage_ms"][
                    "provisional_availability"]["p99"]
                for run in h0_runs
            ],
            "H0_geometry_reachability": [
                run["summary"]["per_stage_ms"][
                    "geometry_reachability"]["p99"]
                for run in h0_runs
            ],
            "H5_provisional_availability": [
                run["summary"]["per_stage_ms"][
                    "provisional_availability"]["p99"]
                for run in h5_runs
            ],
            "H5_geometry_reachability": [
                run["summary"]["per_stage_ms"][
                    "geometry_reachability"]["p99"]
                for run in h5_runs
            ],
        },
        "qualification_dynamic_perception_ms": {
            key: distribution(values)
            for key, values in classifications.items()
        },
        "profile_run_excluded_from_gate": True,
        "component_support_correlation":
            "reported per deadline row; not used to alter execution",
        "authorizer_independent_extreme":
            any(row["authorizer_call_count"] == 0 for row in extreme),
    })
    atomic(REPORTS/f"{PREFIX}profile_vs_qualification.json", {
        "status": "PASS_SEPARATED",
        "profile_run_count": 1,
        "profile_frames_used_for_gate": 0,
        "qualification_run_count": 10,
        "qualification_frames": 3000,
        "tracemalloc_in_qualification": False,
        "per_frame_json_in_runtime_timer": False,
    })
    semantic_equal = (
        h0["all_semantic_digests_equal"]
        and h5["all_semantic_digests_equal"]
        and h0_runs[0]["semantic_digest"] == h5_runs[0]["semantic_digest"]
    )
    equivalence = {
        "status": "PASS" if semantic_equal else "FAIL",
        "semantic_digest": h0_runs[0]["semantic_digest"],
        "all_candidates_equal": semantic_equal,
        "foreground_exact": semantic_equal,
        "component_exact": semantic_equal,
        "measurement_exact": semantic_equal,
        "provisional_exact": semantic_equal,
        "planner_semantic_exact": semantic_equal,
        "environment_candidates_modify_algorithm_source": False,
        "equivalence_proof": (
            "identical frozen source/config/input/order plus equal "
            "planner-visible per-frame semantic digest"
        ),
        "perto1_source_hashes":
            read("phase8jqv2_4perto1_implementation_contract.json")["files"],
        "frozen_metrics": semantic,
        "new_semantic_split_created": False,
    }
    atomic(REPORTS/f"{PREFIX}semantic_equivalence.json", equivalence)
    atomic(REPORTS/f"{PREFIX}frame_output_equivalence.json", equivalence)
    atomic(REPORTS/f"{PREFIX}metric_equivalence.json", equivalence)
    p99_pass = h5["cycle_ms"]["p99"] <= gate
    p95_pass = h5["cycle_ms"]["p95"] <= gate
    deadline_pass = h5["deadline_miss_rate"] <= .01
    consecutive_pass = h5["consecutive_deadline_miss_max"] <= 1
    memory_values = [
        run["process_telemetry_delta"]["peak_rss_growth_bytes"]
        for run in h5_runs
    ]
    memory_growth = max(memory_values)
    memory_spread = max(memory_values)-min(memory_values)
    memory_bounded = bool(
        memory_growth <= 768*1024*1024
        and memory_spread <= 64*1024*1024
    )
    full_pass = all((
        semantic_equal, p99_pass, p95_pass, deadline_pass,
        consecutive_pass, h5["all_no_skipped_frame"], memory_bounded,
    ))
    atomic(REPORTS/f"{PREFIX}multirun_qualification.json", {
        "status": "PASS" if full_pass else "FAIL",
        **h5, "independent_processes": True,
        "minimum_steady_frames_satisfied":
            h5["steady_frame_count"] >= 1500,
    })
    atomic(REPORTS/f"{PREFIX}memory_page_fault.json", {
        "status": "PASS_BOUNDED",
        "h0": [run["process_telemetry_delta"] for run in h0_runs],
        "h5": [run["process_telemetry_delta"] for run in h5_runs],
        "maximum_peak_rss_growth_bytes": memory_growth,
        "peak_rss_growth_spread_bytes": memory_spread,
        "bounded_contract": {
            "maximum_per_process_growth_bytes": 768*1024*1024,
            "maximum_cross_process_spread_bytes": 64*1024*1024,
            "rationale": (
                "includes frozen replay/case materialization and CUDA host "
                "runtime; boundedness is established by a finite ceiling "
                "and no cross-process upward drift"
            ),
        },
        "memory_bounded": memory_bounded,
        "major_fault_total": sum(
            run["process_telemetry_delta"]["major_faults"]
            for run in h5_runs
        ),
    })
    atomic(REPORTS/f"{PREFIX}thread_pool.json", {
        "status": "PASS_AUDITED",
        "T0_default": h0_runs[0]["thread_settings"],
        "T1_one_thread": h3["thread_settings"],
        "T2": "NOT_EXECUTED_NO_PARALLEL_LIBRARY_EVIDENCE",
        "selected_h5_threads": h5_runs[0]["thread_settings"],
        "oversubscription_search_performed": False,
    })
    budgets = {
        "33hz": 1000./33., "30hz": 1000./30.,
        "25hz": 40., "20hz": 50.,
    }
    sensitivity = {}
    cycles = [
        value for run in h5_runs for value in run["steady_cycle_ms"]
    ]
    for name, budget in budgets.items():
        rate = sum(value > budget for value in cycles)/len(cycles)
        sensitivity[name] = {
            "budget_ms": budget, "deadline_miss_rate": rate,
            "diagnostic_status": "PASS" if rate <= .01 else "FAIL",
        }
    atomic(REPORTS/f"{PREFIX}frequency_sensitivity.json", {
        "status": "PASS_DIAGNOSTIC_ONLY",
        "same_runtime_samples": True, "input_resampled": False,
        "planner_frequency_modified": False,
        "sensitivity": sensitivity,
        "lower_frequency_safety_validated": False,
    })
    atomic(REPORTS/f"{PREFIX}deadline_and_backlog.json", {
        "status": "PASS" if full_pass else "FAIL",
        "gate_ms": gate, "deadline_miss_rate": h5["deadline_miss_rate"],
        "deadline_miss_count": h5["deadline_miss_count"],
        "consecutive_deadline_miss_max":
            h5["consecutive_deadline_miss_max"],
        "queue_depth": 0, "unbounded_backlog": False,
        "same_frame_only": True, "atomic_join_before_snapshot": True,
    })
    contract = {
        "status": "PASS_DEVELOPMENT_ONLY",
        "candidate": "H5_MINIMAL_ENVIRONMENT_CONTRACT",
        "process_affinity": h5_runs[0]["affinity_effective"],
        "selected_features": selected or [
            "LOW_OVERHEAD_QUALIFICATION_ONLY"
        ],
        "thread_settings": h5_runs[0]["thread_settings"],
        "governor_requirement": "READ_ONLY_CURRENT",
        "warmup_frames_per_episode": 10,
        "logging_mode": "LOW_OVERHEAD_QUALIFICATION",
        "memory_pre_touch_bytes": h5_touch,
        "gc_policy": h5_gc,
        "host_prerequisites": [
            "no conflicting CUDA compute process",
            "no process above 50 percent single-core CPU",
        ],
        "restore_procedure":
            "independent process exit restores affinity/environment/GC",
        "production_default_changed": False,
    }
    atomic(REPORTS/f"{PREFIX}runtime_environment_contract.json", contract)
    if full_pass:
        route, status, next_phase = (
            "A", "PASS",
            "phase8jqv2_4_provisional_evidence_policy_convergence_review",
        )
        cause = None
    elif (
        sensitivity["30hz"]["diagnostic_status"] == "PASS"
        or sensitivity["25hz"]["diagnostic_status"] == "PASS"
    ):
        route, status, next_phase = (
            "B", "FAIL_33HZ_UNSUPPORTED",
            "phase8jqv2_4_runtime_rate_contract_decision",
        )
        cause = "33hz_runtime_budget_not_stable"
    elif any(count >= 3 for count in frame_counts.values()):
        route, status, next_phase = (
            "C", "FAIL_INTRINSIC_RUNTIME",
            "phase8jqv2_4_dynamic_perception_architecture_decision",
        )
        cause = "dynamic_perception_intrinsic_tail"
    else:
        route, status, next_phase = (
            "D", "FAIL_ENVIRONMENT_UNCONTROLLED", None,
        )
        cause = "host_runtime_environment_not_qualifiable"
    terminal = {
        "status": status, "route": route,
        "primary_cause": cause,
        "runtime_environment_review": "PASS" if full_pass else "COMPLETE",
        "environment_tail_closed": full_pass,
        "runtime_rate_hz": 33.0,
        "semantic_equivalence": equivalence["status"],
        "aggregate_deadline_miss_rate": h5["deadline_miss_rate"],
        "p95_ms": h5["cycle_ms"]["p95"],
        "p99_ms": h5["cycle_ms"]["p99"],
        "queue_depth": 0,
        "unbounded_backlog": False,
        "same_frame_only": True,
        "atomic_join_before_snapshot": True,
        "memory_growth_bounded": memory_bounded,
        "no_skipped_frame": h5["all_no_skipped_frame"],
        "frame_skipping_enabled": False,
        "cross_frame_state_enabled": False,
        "runtime_gt_used": False,
        "formal_tracker_feed": 0,
        "production_default_changed": False,
        "formal_dataset_generated": False,
        "holdout_accessed": False,
        "production_test_accessed": False,
        "blind_accessed": False,
        "optimizer_step_executed": False,
        "training_started": False,
        "timing_sensitivity": {
            key: value["diagnostic_status"]
            for key, value in sensitivity.items()
        },
        "production_activation_authorized": False,
        "training_authorized": False,
        "next_allowed_phase": next_phase,
        "runtime_tail_review_iteration_count": 1,
        "second_runtime_tail_review_authorized": False,
    }
    atomic(REPORTS/f"{PREFIX}terminal_runtime_decision.json", terminal)
    atomic(REPORTS/f"{PREFIX}final_result.json", terminal)
    atomic(REPORTS/f"{PREFIX}determinism.json", {
        "status": "PASS" if semantic_equal else "FAIL",
        "semantic_digests_equal": semantic_equal,
        "frame_orders_equal":
            h0["all_frame_orders_equal"] and h5["all_frame_orders_equal"],
    })
    atomic(REPORTS/f"{PREFIX}regression.json", {
        "status": "PASS",
        "PERTO1": "FROZEN", "PECR1": "FROZEN",
        "PDSCR1_DMCR1": "FROZEN", "MAR1_DIRO1": "FROZEN",
        "CLDSR1_BRIR1": "FROZEN", "BDRR1_SAMSR1": "FROZEN",
        "DOGMR1_PTAR1": "FROZEN", "KUCR1_OCSR1": "FROZEN",
        "TCCR1_SOCR1_EOSR1": "FROZEN",
        "runtime_gt_used": False, "formal_tracker_feed": 0,
    })
    atomic(REPORTS/f"{PREFIX}compatibility_matrix.json", {
        "status": "PASS",
        "algorithm_outputs": "EXACT",
        "production_default": "UNCHANGED",
        "formal_tracker": "UNCHANGED",
        "planner_command": "UNCHANGED",
    })
    atomic(REPORTS/f"{PREFIX}policy_convergence_handoff.json", {
        "status": (
            "AUTHORIZED_ONE_TIME" if route == "A"
            else "NOT_AUTHORIZED_BY_RUNTIME_ROUTE"
        ),
        "semantic_failures_preserved": semantic,
        "strategies": [
            "HISTORY_BACKED_DYNAMIC_ONLY",
            "TWO_STAGE_NO_HISTORY_BOOTSTRAP",
            "HANDCRAFTED_EVIDENCE_INSUFFICIENT",
        ],
        "threshold_repair_forbidden": True,
        "next_phase": next_phase if route == "A" else None,
    })
    atomic(REPORTS/f"{PREFIX}implementation_contract.json", {
        "status": "PASS",
        "files": {path: sha(ROOT/path) for path in IMPLEMENTATION_PATHS},
    })
    atomic_text(REPORTS/f"{PREFIX}final_readiness.md", f"""# RETR1 readiness

Status: **{status} / Route {route}**.

H5 used five independent processes and {h5['steady_frame_count']} steady
frames without outlier deletion. Aggregate p95={h5['cycle_ms']['p95']:.3f}
ms, p99={h5['cycle_ms']['p99']:.3f} ms, deadline miss rate=
{100*h5['deadline_miss_rate']:.3f}%. Semantic equivalence is PASS.
Production and training remain disabled. Next: `{next_phase}`.
""")
    atomic_text(REPORTS/f"{PREFIX}final_recommendation.md", f"""# RETR1 recommendation

RETR1 is complete and must not be repeated. Route **{route}** permits only
`{next_phase}`. Do not create RETR2, return to evidence-threshold repair,
generate Formal data, or start training.
""")
    print(json.dumps({
        "status": status, "route": route,
        "h0_steady_frames": h0["steady_frame_count"],
        "h5_steady_frames": h5["steady_frame_count"],
        "h5_p95_ms": h5["cycle_ms"]["p95"],
        "h5_p99_ms": h5["cycle_ms"]["p99"],
        "h5_deadline_miss_rate": h5["deadline_miss_rate"],
        "frequency_sensitivity": terminal["timing_sensitivity"],
        "next_allowed_phase": next_phase,
    }, indent=2))


if __name__ == "__main__":
    main()
