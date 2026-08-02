#!/usr/bin/env python3
"""Finite, recoverable orchestration for the approved 216 depth sequences."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "Simulator"


def start(command, log, env=None):
    stream = log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True, env=env)
    process._phase8c_log = stream
    return process


def stop(process):
    if process is None:
        return
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    stream = getattr(process, "_phase8c_log", None)
    if stream is not None and not stream.closed:
        stream.close()


def worker_environment(staging, worker_id, port):
    env = os.environ.copy()
    ros_home = staging / "ros_home" / f"worker_{worker_id}"
    ros_home.mkdir(parents=True, exist_ok=True)
    env.update({
        "ROS_MASTER_URI": f"http://127.0.0.1:{port}",
        "ROS_HOSTNAME": "127.0.0.1",
        "ROS_HOME": str(ros_home),
        "ROS_LOG_DIR": str(ros_home / "log"),
    })
    return env


def wait_for_ros_master(env, roscore):
    for _ in range(80):
        if roscore.poll() is not None:
            raise RuntimeError("roscore exited before becoming ready")
        ready = subprocess.run(
            ["rosnode", "list"], capture_output=True, env=env
        ).returncode == 0
        if ready:
            return
        time.sleep(0.25)
    raise RuntimeError("roscore did not become ready")


def wait_for_simulator_subscription(env, simulator, sequence):
    """Wait until map loading is complete and the odometry subscriber exists."""
    for _ in range(300):
        if simulator.poll() is not None:
            log_stream = getattr(simulator, "_phase8c_log", None)
            log_path = Path(log_stream.name) if log_stream is not None else None
            if log_stream is not None:
                log_stream.flush()
            tail = ""
            if log_path is not None and log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(errors="replace").splitlines()[-12:]
                )
            raise RuntimeError(
                f"simulator exited while loading map: {sequence}; "
                f"returncode={simulator.returncode}; log={log_path}\n{tail}"
            )
        try:
            node = subprocess.run(
                ["rosnode", "info", "/sensor_simulator_node"],
                capture_output=True, text=True, env=env, timeout=2,
            )
        except subprocess.TimeoutExpired:
            node = None
        if node is not None and node.returncode == 0 and "/sim/odom" in node.stdout:
            return
        time.sleep(0.1)
    raise TimeoutError(f"simulator did not finish map loading: {sequence}")


def record_one(staging, logs, row, env, cancel_event):
    sequence = row["sequence_id"]
    current_sim = current_odom = current_recorder = None
    try:
        goal = [str(value) for value in json.loads(row["goal"])]
        command = [
            sys.executable, str(ROOT / "tools/record_dynamic_sequences.py"),
            "--output", str(staging), "--sequence-id", sequence,
            "--scenario-file", row["scenario_file"], "--scenario-id", sequence,
            "--scenario-type", row["scenario_type"], "--seed", row["actor_seed"],
            "--frames", "60", "--rate", "10", "--sync-slop", "0.002",
            "--timeout", "35", "--goal", *goal, "--map-id", row["map_id"],
            "--static-map-sha256", row["static_map_sha256"],
        ]
        if row.get("record_instance", "").lower() in {"1", "true", "yes"}:
            command.extend(["--instance-topic", "/dynamic_actor_instance"])
        # Subscribe before the first odometry message so actor time starts at
        # the same point as the captured sequence.
        current_recorder = start(
            command, logs / f"{sequence}_recorder.log", env=env
        )
        recorder_prefix = f"/record_{sequence}"
        for _ in range(120):
            nodes = subprocess.run(
                ["rosnode", "list"], capture_output=True, text=True, env=env
            )
            if nodes.returncode == 0 and any(
                    node.startswith(recorder_prefix)
                    for node in nodes.stdout.splitlines()):
                break
            if current_recorder.poll() is not None:
                raise RuntimeError(f"recorder exited before subscribing: {sequence}")
            time.sleep(0.1)
        else:
            raise TimeoutError(f"recorder did not become ready: {sequence}")
        current_sim = start([
            "rosrun", "sensor_simulator", "sensor_simulator_cuda",
            f"_dynamic_scenario_file:={row['scenario_file']}", "_random_map:=false",
            f"_ply_file:={row['static_ply']}", "_render_lidar:=false",
            "_render_depth:=true",
        ], logs / f"{sequence}_simulator.log", env=env)
        # Loading a multi-million-point PLY can take seconds. Starting odometry
        # before the subscriber exists silently advances the path clock and
        # makes path-relative actors miss the camera's recorded interval.
        wait_for_simulator_subscription(env, current_sim, sequence)
        current_odom = start([
            "/usr/bin/python3", str(ROOT / "tools/publish_reachable_path_odom.py"),
            "--reachability", row["reachability_json"],
            "--map-id", row["local_map_id"], "--speed", "2.0",
        ], logs / f"{sequence}_odom.log", env=env)
        deadline = time.monotonic() + 55.0
        while current_recorder.poll() is None:
            if cancel_event.is_set():
                raise InterruptedError(f"recording cancelled: {sequence}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"recording timed out: {sequence}")
            time.sleep(0.2)
        return_code = current_recorder.returncode
        if return_code != 0:
            raise RuntimeError(
                f"recorder failed for {sequence}; see {sequence}_recorder.log"
            )
        stop(current_recorder)
        current_recorder = None
    finally:
        stop(current_recorder)
        stop(current_sim)
        stop(current_odom)


def run_worker(worker_id, port, rows, staging, logs, progress, progress_lock,
               cancel_event):
    env = worker_environment(staging, worker_id, port)
    roscore = None
    try:
        roscore = start(
            ["roscore", "-p", str(port)],
            logs / f"worker_{worker_id}_roscore.log", env=env,
        )
        wait_for_ros_master(env, roscore)
        for row in rows:
            if cancel_event.is_set():
                return
            with progress_lock:
                progress[0] += 1
                number = progress[0]
            print(
                f"[{number}/{progress[1]}] worker={worker_id} recording "
                f"{row['sequence_id']} {row['scenario_type']}", flush=True,
            )
            record_one(staging, logs, row, env, cancel_event)
            free_gib = shutil.disk_usage(staging).free / 2**30
            if free_gib < 10:
                raise OSError("formal recording stopped: less than 10 GiB free")
    except BaseException:
        cancel_event.set()
        raise
    finally:
        stop(roscore)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-catalog", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenario-type")
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ros-master-port-base", type=int, default=12100)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing formal dataset: {output}")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    logs = staging / "logs"; logs.mkdir()
    scenarios = staging / "scenario_configs"
    subprocess.run([
        sys.executable, str(ROOT / "tools/generate_phase8c_scenario_matrix.py"),
        "--catalog", str(args.static_catalog), "--output", str(scenarios),
    ], check=True)
    rows = list(csv.DictReader((scenarios / "matrix.csv").open(newline="", encoding="utf-8")))
    if args.scenario_type is not None:
        rows = [row for row in rows if row["scenario_type"] == args.scenario_type]
        if not rows:
            raise ValueError(f"scenario type not present in matrix: {args.scenario_type}")
    if args.sequence_id:
        requested = set(args.sequence_id)
        rows = [row for row in rows if row["sequence_id"] in requested]
        found = {row["sequence_id"] for row in rows}
        if found != requested:
            raise ValueError(f"sequence IDs not present in matrix: {sorted(requested - found)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no sequences selected")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be in 1..8")
    worker_count = min(args.workers, len(rows))
    if not 1024 <= args.ros_master_port_base <= 65535 - worker_count:
        raise ValueError("ROS master port range is invalid")
    is_smoke = bool(args.limit is not None or args.scenario_type or args.sequence_id)
    print(json.dumps({
        "status": "STARTING", "staging": str(staging),
        "requested_output": str(output), "sequences": len(rows),
        "workers": worker_count, "smoke": is_smoke,
    }, indent=2), flush=True)
    try:
        buckets = [rows[index::worker_count] for index in range(worker_count)]
        progress = [0, len(rows)]
        progress_lock = threading.Lock()
        cancel_event = threading.Event()
        executor = ThreadPoolExecutor(max_workers=worker_count)
        try:
            futures = [
                executor.submit(
                    run_worker, worker_id,
                    args.ros_master_port_base + worker_id, bucket,
                    staging, logs, progress, progress_lock, cancel_event,
                )
                for worker_id, bucket in enumerate(buckets)
            ]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            cancel_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        if is_smoke:
            from finalize_phase8c_dynamic_dataset import load_yaml, validate_sequence

            sequence_stats = {}
            for row in rows:
                sequence = row["sequence_id"]
                metadata = load_yaml(
                    staging / "sequences" / sequence / "metadata.yaml"
                )
                sequence_stats[sequence] = validate_sequence(staging, row, metadata)
            (staging / "smoke_manifest.json").write_text(json.dumps({
                "status": "PASS", "scope": "pre_formal_recording_smoke",
                "sequences": len(rows), "workers": worker_count,
                "formal_manifest": False, "sequence_stats": sequence_stats,
            }, indent=2) + "\n", encoding="utf-8")
            os.replace(staging, output)
            print(json.dumps({"status": "PASS", "scope": "smoke", "sequences": len(rows),
                              "output": str(output)}, indent=2))
            return
        subprocess.run([
            sys.executable, str(ROOT / "tools/finalize_phase8c_dynamic_dataset.py"),
            "--dataset", str(staging), "--matrix", str(scenarios / "matrix.csv"),
            "--map-catalog", str(args.static_catalog),
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "tools/validate_dynamic_dataset.py"), str(staging),
        ], check=True)
        os.replace(staging, output)
        print(json.dumps({"status": "PASS", "sequences": len(rows),
                          "output": str(output)}, indent=2))
    except BaseException:
        (staging / "INCOMPLETE").write_text("formal recording did not commit\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
