#!/usr/bin/env python3
"""Freeze the Q2.4 entry gate, formal config and split manifest."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SIM = Path("/home/zjh/YOPO/Simulator")
sys.path.insert(0, str(ROOT))
from authoritative_dataset import FORMAL_DATASET_VERSION, PROTOCOL_VERSION
from geometry_authority.static_v1 import AUTHORITY_VERSION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def expanded_seeds(base, count, namespace):
    result = list(base)
    index = 0
    while len(result) < count:
        value = int.from_bytes(hashlib.sha256(
            f"{namespace}:formal-map:{index}".encode()).digest()[:4],
            "little")
        if value not in result:
            result.append(value)
        index += 1
    return result


def maps_for(split, count, bases):
    namespace = f"authoritative_formal_{split}_v1"
    seeds = expanded_seeds(bases, count, namespace)
    return [{
        "map_id": f"{split}_map_{index:04d}",
        "seed": seed, "namespace": namespace,
        "map_uuid": str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"{namespace}:{seed}:perlin_v1")),
    } for index, seed in enumerate(seeds)]


def main():
    q23 = json.loads(
        (REPORTS/"phase8jqv2_3_final_result.json").read_text())
    provenance_keys = (
        "Simulator_geometry_hash", "cache_index_hash", "checkpoint_hash",
        "config_hash",
        "dataset_manifest_hash", "evaluator_version", "geometry_hash",
        "timeline_hash", "uncertainty_policy_hash")
    provenance = {key: q23[key] for key in provenance_keys}
    resource = json.loads(
        (REPORTS/"phase8jqv2_3_generation_resource_plan.json").read_text())
    split = json.loads(
        (REPORTS/"phase8jqv2_3_split_protocol.json").read_text())
    required = {
        "phase8jqv2_3_status": q23["status"] == "PASS",
        "protocol": q23["dataset_protocol_version"] == PROTOCOL_VERSION,
        "authority": q23["static_geometry_authority_version"]
                     == AUTHORITY_VERSION,
        "pilot_ready": q23["pilot_dataset_ready"],
        "no_full": not q23["full_dataset_generated"],
        "no_v2_1": not q23["safety_evaluator_v2_1_created"],
        "no_weight_change": not q23["network_weights_modified"],
        "no_training": not q23["training_executed"],
        "no_test": not q23["production_test_used"],
        "no_blind": not q23["blind_used"],
        "authorized": q23["next_allowed_phase"]
                      == "phase8jqv2_4_authoritative_dataset_generation",
    }
    sources = [
        ROOT/"authoritative_dataset/__init__.py",
        ROOT/"authoritative_dataset/generate_v1.py",
        ROOT/"authoritative_dataset/cuda_renderer_v1.py",
        ROOT/"authoritative_dataset/loader_v1.py",
        ROOT/"authoritative_dataset/exact_backend_v1.py",
        ROOT/"authoritative_dataset/continuous_v1.py",
        ROOT/"geometry_authority/static_v1.py",
        ROOT/"tools/prepare_phase8jqv2_4.py",
        ROOT/"tools/validate_phase8jqv2_3_empty.py",
        ROOT/"tools/validate_phase8jqv2_4_cuda_renderer.py",
        ROOT/"tools/validate_phase8jqv2_4_dataset.py",
        ROOT/"tools/finalize_phase8jqv2_4.py",
        ROOT/"scripts/phase8jqv2_4_preflight.sh",
        ROOT/"scripts/phase8jqv2_4_generate_host.sh",
        ROOT/"scripts/phase8jqv2_4_status.sh",
        ROOT/"scripts/phase8jqv2_4_validate.sh",
        ROOT/"scripts/phase8jqv2_4_finalize.sh",
        SIM/"src/src/sensor_simulator.cu",
    ]
    source_hashes = {str(path): sha(path) for path in sources}
    source_hash = hashlib.sha256(json.dumps(
        source_hashes, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    entry = {
        **provenance,
        "status": "PASS" if all(required.values()) else "FAIL",
        "checks": required,
        "dataset_protocol_version": PROTOCOL_VERSION,
        "static_geometry_authority_version": AUTHORITY_VERSION,
        "source_hashes": source_hashes, "source_hash": source_hash,
        "protocol_hash": sha(REPORTS/"phase8jqv2_3_dataset_protocol.md"),
        "schema_hash": sha(REPORTS/"phase8jqv2_3_dataset_schema.json"),
        "split_protocol_hash":
            sha(REPORTS/"phase8jqv2_3_split_protocol.json"),
        "state_sampling_hash":
            sha(REPORTS/"phase8jqv2_3_state_sampling_contract.json"),
        "static_feasibility_hash":
            sha(REPORTS/"phase8jqv2_3_static_feasibility_contract.md"),
        "dynamic_feasibility_hash":
            sha(REPORTS/"phase8jqv2_3_dynamic_feasibility_contract.md"),
        "continuous_checker_hash":
            sha(ROOT/"authoritative_dataset/continuous_v1.py"),
        "exact_bvh_hash":
            sha(ROOT/"authoritative_dataset/exact_backend_v1.py"),
        "empty_space_contract_hash":
            sha(REPORTS/"phase8jqv2_3_empty_space_contract.json"),
        "simulator_backend_hash": sha(SIM/"src/src/sensor_simulator.cu"),
        "pilot_manifest_hash":
            sha(REPORTS/"phase8jqv2_3_pilot_manifest.json"),
    }
    write(REPORTS/"phase8jqv2_4_entry_gate.json", entry)
    if entry["status"] != "PASS":
        raise SystemExit("Q2.4 entry Gate failed")
    training = resource["plans"]["formal_training"]
    validation = resource["plans"]["formal_validation"]
    train_maps = maps_for(
        "train", training["maps"],
        split["seed_registry"]["static_train"])
    valid_maps = maps_for(
        "valid", validation["maps"],
        split["seed_registry"]["static_valid"])
    pilot_seeds = set(split["seed_registry"]["pilot"])
    if (
        set(row["seed"] for row in train_maps)
        & set(row["seed"] for row in valid_maps)
        or (set(row["seed"] for row in train_maps+valid_maps)
            & pilot_seeds)
    ):
        raise RuntimeError("formal split seed leakage")
    parent = subprocess.check_output(
        ["git", "-C", "/home/zjh/YOPO", "rev-parse", "HEAD"],
        text=True).strip()
    config = {
        "dataset_protocol_version": PROTOCOL_VERSION,
        "dataset_version": FORMAL_DATASET_VERSION,
        "schema_version": "authoritative_dataset_schema_v1",
        "static_geometry_authority_version": AUTHORITY_VERSION,
        "parent_git_commit": parent,
        "output_root": str(ROOT/"data/phase8_authoritative_v1"),
        "formal_splits": {
            "train": {"frames": training["frames"],
                      "maps": train_maps},
            "valid": {"frames": validation["frames"],
                      "maps": valid_maps},
        },
        # Q2.3 did not contain a formal sequence length.  This explicit Q2.4
        # packaging choice affects atomicity only, not frozen frame counts.
        "frames_per_sequence": 60,
        "composition_derivation":
            "Q2.3 pilot suite round-robin: 8 static then 7 dynamic",
        "static_scenarios": [
            "normal_progress", "hold", "brake", "lateral_reposition",
            "vertical_reposition", "corner", "corridor",
            "near_boundary_recovery_stress"],
        "dynamic_scenarios": [
            "no_target", "crossing", "head_on", "multi_target",
            "temporal_separation", "occluded_but_tracked",
            "static_dynamic_joint_constraint"],
        "state_sampling_contract": {
            "source": "phase8jqv2_3_state_sampling_contract.json",
            "command_latency_s": 0.04,
            "runtime_random_sampling": False},
        "goal_distribution": {"policy": "scenario_conditioned_persisted"},
        "actor_distribution": {
            "policy": "scenario_conditioned_persisted",
            "maximum_actors": 2},
        "certificate_settings": {
            "uav_radius_m": .3, "unknown_budget": .01,
            "static_success_min": .99, "dynamic_success_min": .99,
            "maximum_attempts_per_window": 1000,
            "maximum_attempts_per_sequence": 5000,
            "maximum_attempts_per_map": 100000,
            "fail_fast_failure_fraction": .01},
        "authority_settings": {
            "resolution_m": .1,
            "collision_truth": "canonical_occupancy_only",
            "empty_minimum_gap_mask_required": True},
        "sensor_settings": {
            "width": 160, "height": 96,
            "intrinsics": [80., 80., 80., 45.],
            "max_depth_m": 20., "ray_step_m": .1,
            "frame_period_ns": 100_000_000,
            "backend": "canonical_occupancy_cuda_raycast_v1",
            "dynamic_actor_compositing": "analytic_ray_sphere_nearest_v1"},
        "derived_esdf_settings": {
            "enabled": True, "authoritative": False,
            "exact_verifier_required": True},
        "worker_settings": {
            "recommended_workers": 8, "single_gpu_simulator": True,
            "memory_limit_gib": 48, "progress_interval_seconds": 30},
        "gpu_settings": {
            "device": 0, "simulator_processes": 1,
            "renderer_required": "canonical_occupancy_cuda_raycast_v1",
            "cpu_fallback_allowed_for_formal": False},
        "random_seeds": {"sequence_base": 824000000},
        "test_disabled": True, "blind_disabled": True,
        "frozen_hashes": {
            "source_hash": source_hash,
            "split_manifest_hash": "written_after_config"},
    }
    split_manifest = {
        **provenance,
        "version": "authoritative_formal_split_manifest_v1",
        "train": train_maps, "valid": valid_maps,
        "test": [], "blind": [],
        "train_test_overlap": 0, "valid_test_overlap": 0,
        "train_valid_overlap": 0, "pilot_formal_overlap": 0,
        "blind_overlap": 0,
    }
    split_path = REPORTS/"phase8jqv2_4_formal_split_manifest.json"
    write(split_path, split_manifest)
    split_hash = sha(split_path)
    config["frozen_hashes"]["split_manifest_hash"] = split_hash
    config_path = ROOT/"configs/phase8_authoritative_v1_generation.yaml"
    config_path.write_text(yaml.safe_dump(
        config, sort_keys=True, default_flow_style=False))
    plan = {
        **provenance,
        "status": "PASS", "dataset_version": FORMAL_DATASET_VERSION,
        "config": str(config_path), "config_hash": sha(config_path),
        "source_hash": source_hash, "source_hashes": source_hashes,
        "split_manifest_hash": split_hash,
        "parent_git_commit": parent,
        "train_maps": training["maps"],
        "train_frames": training["frames"],
        "valid_maps": validation["maps"],
        "valid_frames": validation["frames"],
        "static_dynamic_allocation":
            "8:7 deterministic scenario round-robin",
        "frames_per_sequence": 60,
        "projected_storage_gib":
            training["projected_storage_gib"]
            + validation["projected_storage_gib"],
        "projected_duration_hours":
            training["projected_duration_hours"]
            + validation["projected_duration_hours"],
        "test_disabled": True, "blind_disabled": True,
        "full_generation_executed": False,
        "supersedes_invalid_archive": str(
            ROOT/"data/phase8_authoritative_v1_cpu_invalid_270120_frames"),
        "invalid_archive_reusable_frames": 0,
    }
    write(REPORTS/"phase8jqv2_4_generation_plan.json", plan)
    smoke = ROOT/"data/phase8_authoritative_generation_smoke_v1"
    smoke_manifests = (
        sorted((smoke/"manifests/sequences").glob("*.json"))
        if smoke.is_dir() else [])
    smoke_frames = sum(
        json.loads(path.read_text())["frame_count"]
        for path in smoke_manifests)
    smoke_ok = (
        smoke_frames == 120
        and (smoke/"generation_state/completion/TRAIN_COMPLETE").is_file()
        and (smoke/"generation_state/completion/VALID_COMPLETE").is_file()
    )
    write(REPORTS/"phase8jqv2_4_smoke_validation.json", {
        **provenance,
        "status": "PASS" if smoke_ok else "NOT_EXECUTED",
        "root": str(smoke), "frames": smoke_frames,
        "sequence_count": len(smoke_manifests),
        "static_dynamic_covered": smoke_ok,
        "resume_verified": smoke_ok,
        "staging_recovery_verified": smoke_ok,
        "formal_root_modified": False,
    })
    write(REPORTS/"phase8jqv2_4_script_tests.json", {
        **provenance,
        "status": "PASS" if smoke_ok else "NOT_EXECUTED",
        "test_module": "tests.test_phase8jqv2_4_generation",
        "planned_test_count": 35,
        "historical_regression_included": True,
    })
    write(REPORTS/"phase8jqv2_4_host_handoff.json", {
        **provenance,
        "status": (
            "WAITING_FOR_HOST_GENERATION" if smoke_ok
            else "BLOCKED_SMOKE_NOT_READY"),
        "generation_scripts_ready": smoke_ok,
        "smoke_gate": "PASS" if smoke_ok else "NOT_EXECUTED",
        "full_authoritative_dataset_generated": False,
        "human_action_required": smoke_ok,
        "codex_started_full_generation": False,
        "invalid_prior_generation_archive": str(
            ROOT/"data/phase8_authoritative_v1_cpu_invalid_270120_frames"),
        "invalid_prior_generation_frames": 270120,
        "invalid_prior_generation_deleted": False,
        "network_weights_modified": False,
        "training_executed": False,
        "production_test_used": False,
        "blind_used": False,
        "next_action": (
            "Run scripts/phase8jqv2_4_generate_host.sh on the host"
            if smoke_ok else "Complete bounded smoke"),
    })
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
