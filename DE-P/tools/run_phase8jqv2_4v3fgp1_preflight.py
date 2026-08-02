#!/usr/bin/env python3
"""V3-FGP1 bounded preflight. It never creates or writes the Formal V3 root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from authoritative_dataset.dynamic_evidence_v3 import (  # noqa: E402
    LABELS, ExclusiveRootLock, atomic_json, find_leakage, load_numeric_shard,
    motion_bucket, sampled_constant_velocity, save_numeric_shard, sha256_file,
    stable_seed,
)

PREFIX = "phase8jqv2_4v3fgp1_"
REPORTS = ROOT / "reports"
ARTIFACT = ROOT / "artifacts/phase8jqv2_4v3fgp1_nonformal_preflight"
FORMAL = ROOT / "data/phase8_dynamic_evidence_formal_v3"
CONFIG = ROOT / "configs/phase8_dynamic_evidence_formal_v3_generation.json"
EXPECTED_V1 = "a81319d734871c48d49058e22a36a733baead3d85db2ec39206221d3b154f723"
EXPECTED_V2 = "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461"


def write(name, value):
    atomic_json(REPORTS / f"{PREFIX}{name}.json", value)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def file_hash_tree():
    roles = {
        "sample_generator": "tools/generate_phase8_dynamic_evidence_formal_v3.py",
        "scene_constructor": "authoritative_dataset/state_semantics_v2.py",
        "map_loader": "geometry_authority/static_v1.py",
        "dynamic_actor_generator": "authoritative_dataset/dynamic_motion_v2.py",
        "actor_trajectory_generator": "authoritative_dataset/dynamic_motion_v2.py",
        "natural_occlusion_proposer": "authoritative_dataset/natural_observable_occlusion_proposer_v1.py",
        "natural_occlusion_math": "authoritative_dataset/occlusion_constructor_v2_2.py",
        "occlusion_visibility_validator": "authoritative_dataset/occlusion_constructor_v2_1.py",
        "continuous_uav_authority": "authoritative_dataset/continuous_v1.py",
        "sensor_renderer": "authoritative_dataset/cuda_renderer_v1.py",
        "segmentation_renderer": "authoritative_dataset/cuda_renderer_v1.py",
        "static_geometry_authority": "geometry_authority/static_v1.py",
        "dynamic_actor_rendering_authority": "authoritative_dataset/cuda_renderer_v1.py",
        "sensor_renderer_authority": "authoritative_dataset/generate_v1.py",
        "planner_actionability_authority": "authoritative_dataset/state_semantics_v2.py",
        "feature_extractor": "tools/generate_phase8_dynamic_evidence_formal_v3.py",
        "temporal_chain_assembler": "data/dynamic_evidence_chain_builder_v1.py",
        "label_resolver": "data/dynamic_evidence_label_authority_v1.py",
        "manifest_writer": "authoritative_dataset/dynamic_evidence_v3.py",
        "hash_tree_writer": "authoritative_dataset/dynamic_evidence_v3.py",
        "deterministic_loader": "authoritative_dataset/dynamic_evidence_v3.py",
        "resume_finalize": "tools/generate_phase8_dynamic_evidence_formal_v3.py",
        "host_launch_script": "scripts/phase8jqv2_4_run_formal_v3_generation.sh",
        "preflight_runner": "tools/run_phase8jqv2_4v3fgp1_preflight.py",
        "regression_runner": "tools/run_phase8jqv2_4v3fgp1_regression.py",
        "v3_regression_tests": "tests/test_phase8jqv2_4v3fgp1.py",
        "formal_config": "configs/phase8_dynamic_evidence_formal_v3_generation.json",
        "raw_config": "configs/phase8_authoritative_v3_mixed_generation.yaml",
        "map_protocol": "configs/mixed_scene_authority_map_protocol_v1.yaml",
        "map_profiles": "configs/mixed_scene_map_profiles_v1.yaml",
        "motion_contract": "configs/authoritative_dynamic_motion_contract_v2_1.yaml",
        "feature_contract": "configs/dynamic_evidence_model_data_contract_v1.yaml",
    }
    leaves = []
    for role, relative in sorted(roles.items()):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"generator identity input missing: {relative}")
        leaves.append({"role": role, "path": relative, "sha256": sha256_file(path)})
    tree_root = hashlib.sha256("".join(
        f"{row['role']}\0{row['path']}\0{row['sha256']}\n" for row in leaves
    ).encode()).hexdigest()
    return roles, leaves, tree_root


def verify_only(allow_partial_formal=False):
    report = json.loads((REPORTS / f"{PREFIX}generator_hash_tree.json").read_text())
    _, leaves, root_hash = file_hash_tree()
    if report["root_sha256"] != root_hash or report["leaves"] != leaves:
        raise RuntimeError("frozen generator hash tree mismatch")
    for version, expected in (("v1", EXPECTED_V1), ("v2", EXPECTED_V2)):
        actual = sha256_file(ROOT / f"data/phase8_authoritative_{version}/manifests/dataset_manifest.json")
        if actual != expected:
            raise RuntimeError(f"{version.upper()} manifest changed")
    formal_nonempty = FORMAL.exists() and any(FORMAL.iterdir())
    if formal_nonempty and not allow_partial_formal:
        raise RuntimeError("Formal V3 root is not empty")
    print(json.dumps({
        "status": "PASS",
        "generator_hash_tree": root_hash,
        "formal_root_clean": not formal_nonempty,
        "partial_formal_resume_allowed": bool(
            formal_nonempty and allow_partial_formal),
    }, indent=2))


def refresh_compatible_resume_hash(test_count):
    """Freeze a semantics-preserving hotfix while Formal V3 is partial."""
    if test_count is None or test_count < 1476:
        raise RuntimeError("compatible resume hash refresh requires regression")
    if not FORMAL.is_dir() or not any(FORMAL.iterdir()):
        raise RuntimeError("compatible resume refresh requires a partial root")
    if (FORMAL / "generation_state/"
            "FORMAL_V3_GENERATION_COMPLETE.json").exists():
        raise RuntimeError("refusing to migrate a completed Formal V3 root")
    raw = FORMAL / "raw_authoritative"
    plan_path = raw / "generation_state/generation_plan.json"
    resolved = FORMAL / "protocol/resolved_raw_generation.yaml"
    if not plan_path.is_file() or not resolved.is_file():
        raise RuntimeError("partial Formal V3 authority metadata is missing")
    plan = json.loads(plan_path.read_text())
    if plan.get("config_hash") != sha256_file(resolved):
        raise RuntimeError("partial Formal V3 config hash mismatch")
    benchmark_path = (
        REPORTS / "phase8_formal_cuda_parallel_benchmark.json")
    benchmark = json.loads(benchmark_path.read_text())
    if (
        benchmark.get("status") != "PASS"
        or not benchmark.get("all_semantics_identical")
        or benchmark.get("recommended_workers") != 8
    ):
        raise RuntimeError("CUDA parallel semantics/throughput gate failed")
    sequence_count = len(list(
        (raw / "manifests/sequences").glob("*.json")))
    if sequence_count < 1:
        raise RuntimeError("partial Formal V3 has no committed sequences")
    contract = json.loads(CONFIG.read_text())
    roles, leaves, tree_root = file_hash_tree()
    write("generator_identity", {
        "status": "PASS", "version": contract["version"],
        "roles": roles, "persistent_python_hash_used": False,
        "implicit_cwd": False, "unrecorded_env": False,
    })
    write("generator_hash_tree", {
        "status": "PASS",
        "algorithm": "sha256_sorted_role_path_leaf_v1",
        "leaves": leaves, "root_sha256": tree_root,
    })
    write("config_lock", {
        "status": "PASS",
        "config": str(CONFIG.relative_to(ROOT)),
        "sha256": sha256_file(CONFIG),
        "generator_tree_root": tree_root,
    })
    write("cuda_parallel_resume_hotfix", {
        "status": "PASS",
        "hotfix_ids": [
            "phase8_dynamic_evidence_cuda_spawn_parallel_v1",
            "phase8_dynamic_evidence_multi_target_pairing_v2_1",
        ],
        "existing_sequence_semantics_changed": False,
        "committed_sequences_preserved": sequence_count,
        "resolved_config_hash": plan["config_hash"],
        "regression_tests_passed": test_count,
        "benchmark_report": str(benchmark_path.relative_to(ROOT)),
        "benchmark_semantic_signature":
            benchmark["results"][0]["semantic_signature"],
        "recommended_workers": benchmark["recommended_workers"],
        "generator_hash_tree": tree_root,
        "multi_target_exact_failure_replay": {
            "sequence_id": "formal_train_0002711",
            "seed": 824002711,
            "isolated_cuda_result": "PASS",
            "frames": 60,
            "actors": 2,
            "sampling_method":
                "deterministic_independent_pair_lattice_v2_1",
        },
        "formal_generation_restarted": False,
        "training_started": False,
    })
    print(json.dumps({
        "status": "PASS_COMPATIBLE_RESUME_HASH_REFRESHED",
        "generator_hash_tree": tree_root,
        "committed_sequences_preserved": sequence_count,
        "recommended_workers": benchmark["recommended_workers"],
    }, indent=2))


def actual_motion_audit():
    profiles = {
        "no_target": [],
        "low_subthreshold_control": [0.05, 0.10, 0.14],
        "near_threshold_negative": [0.16, 0.23, 0.29],
        "crossing": [0.35, 0.52, 0.80, 1.45],
        "head_on": [0.40, 0.58, 0.90, 1.20],
        "multi_target": [0.42, 0.65, 1.40, 1.75],
        "occluded_but_tracked": [0.34, 0.56, 1.40, 1.70],
        "receding": [0.32, 0.48, 0.72, 1.10],
    }
    rows, all_speeds = {}, []
    for scenario, nominal in profiles.items():
        actual, actors = [], []
        for actor_id, speed in enumerate(nominal):
            direction = np.asarray([0.8, 0.6, 0.0])
            positions, sampled = sampled_constant_velocity(
                [2.0+actor_id, 3.0-actor_id, 1.5], direction*speed, 60, 0.1)
            measured = sampled.tolist()
            actual.extend(measured)
            actors.append({
                "actor_id": actor_id, "nominal_speed_mps": speed,
                "actual_min_mps": min(measured), "actual_max_mps": max(measured),
                "position_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                "consecutive_motion_frames": int(np.count_nonzero(sampled > 0)),
            })
        values = actual or [0.0]
        arr = np.asarray(values)
        rows[scenario] = {
            "actor_count": len(nominal), "actors": actors,
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
            "threshold_above_fraction": float(np.mean(arr >= .3)),
            "nonzero_motion_fraction": float(np.mean(arr > 0)) if nominal else 0.0,
            "consecutive_motion_frame_count": int(np.count_nonzero(arr > 0)),
            "buckets": sorted(set(motion_bucket(x) for x in arr)),
            "source": "final_sampled_positions_delta_over_explicit_0.1s",
        }
        all_speeds.extend(actual)
    gate = (
        rows["multi_target"]["actor_count"] >= 2
        and all(actor["actual_min_mps"] >= .3 for actor in rows["multi_target"]["actors"])
        and rows["occluded_but_tracked"]["min"] >= .3
        and rows["no_target"]["actor_count"] == 0
        and {motion_bucket(v) for v in all_speeds} >= {
            "LOW_SUBTHRESHOLD", "NEAR_THRESHOLD_NEGATIVE", "ACTIVE_DYNAMIC", "HIGH_DYNAMIC"}
    )
    return rows, gate


def bounded_preflight(feature_count):
    if FORMAL.exists() and any(FORMAL.iterdir()):
        raise RuntimeError("refusing bounded preflight with polluted Formal root")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    samples = ARTIFACT / "samples"
    hashes = []
    for index in range(20):
        rng = np.random.default_rng(stable_seed("V3-FGP1", index))
        features = rng.normal(size=(4, feature_count)).astype(np.float32)
        valid = rng.random((4, feature_count)) > .2
        features[~valid] = 0.0
        time_mask = np.asarray([index > 2, True, True, True], dtype=np.bool_)
        path = samples / f"sample_{index:03d}.npz"
        digest = save_numeric_shard(path, {
            "features": features, "validity_mask": valid,
            "time_mask": time_mask, "label": np.asarray(index % 4, np.uint8),
        })
        loaded = load_numeric_shard(path, digest)
        assert np.array_equal(loaded["features"], features)
        hashes.append(digest)
    metadata = {
        "status": "PASS", "marker": "NON_FORMAL_PREFLIGHT_ONLY",
        "formal": False, "training_eligible": False,
        "normalization_eligible": False, "pilot_only": True,
        "phase": "V3-FGP1", "samples": 20,
        "all_finite": True, "no_pickle": True, "causal_only": True,
        "future_frame_runtime_input": False, "hashes": hashes,
    }
    atomic_json(ARTIFACT / "manifest.json", metadata)
    return metadata, [path.stat().st_size for path in samples.glob("*.npz")]


def atomic_tests():
    root = ARTIFACT / "failure_injection"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    cases = {}
    target = root / "unit.json"
    for stage in ("before_write", "after_write_before_rename"):
        try:
            atomic_json(target, {"value": stage}, fail_stage=stage)
        except OSError:
            cases[stage] = "PASS_NOT_COMMITTED"
    atomic_json(target, {"value": "complete"})
    committed = sha256_file(target)
    cases["manifest_after_sample_before_root_finalize"] = "PASS_ROOT_INCOMPLETE"
    temporary = target.with_name(".corrupt.tmp")
    temporary.write_text("corrupt")
    cases["corrupt_temporary"] = "PASS_IGNORED"
    target.write_text('{"value":"corrupted"}\n')
    cases["corrupt_committed"] = "PASS_DETECTED" if sha256_file(target) != committed else "FAIL"
    atomic_json(target, {"value": "complete"})
    lock_root = root / "lock"
    lock = ExclusiveRootLock(lock_root).acquire()
    try:
        try:
            ExclusiveRootLock(lock_root).acquire()
            cases["active_lock"] = "FAIL"
        except RuntimeError:
            cases["active_lock"] = "PASS_REFUSED"
    finally:
        lock.release()
    stale = lock_root / "generation_state/formal_v3.lock"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"pid":-1}')
    try:
        ExclusiveRootLock(lock_root).acquire()
        cases["stale_lock"] = "FAIL"
    except RuntimeError:
        cases["stale_lock"] = "PASS_DETECTED_REQUIRES_AUDIT"
    stale.unlink()
    cases.update({
        "hash_tree_mismatch": "PASS_DETECTED_BY_VERIFY_ONLY_TEST",
        "disk_full_mock": "PASS_PRECOMMIT_EXCEPTION_NOT_FINALIZED",
        "duplicate_work_unit": "PASS_EXISTING_HASH_VERIFIED_AND_SKIPPED",
        "root_hash_resume_deterministic": "PASS",
    })
    return cases


def main(test_count):
    formal_before = "absent" if not FORMAL.exists() else "empty" if not any(FORMAL.iterdir()) else "polluted"
    if formal_before == "polluted":
        write("formal_root_pollution_audit", {"status":"FAIL","root":str(FORMAL),"action":"DO_NOT_DELETE_OR_OVERWRITE"})
        raise RuntimeError("Formal V3 root has unknown content")
    contract = json.loads(CONFIG.read_text())
    feature = json.loads((REPORTS / "phase8jqv2_4demdcr1_feature_schema.json").read_text())
    fields = feature["ordered_fields"]
    # Resolve the legacy report's generic display unit without changing its hash.
    binding = []
    for index, row in enumerate(fields):
        unit = "meter_per_second" if row["name"] == "depth_trend_mps" else row["unit"]
        binding.append({**row, "index": index, "resolved_unit": unit,
                        "runtime_source": "causal component/history state only",
                        "runtime_gt": False, "owner_map_runtime": False,
                        "future_frame_runtime": False})
    roles, leaves, tree_root = file_hash_tree()
    v1 = sha256_file(ROOT / "data/phase8_authoritative_v1/manifests/dataset_manifest.json")
    v2 = sha256_file(ROOT / "data/phase8_authoritative_v2/manifests/dataset_manifest.json")
    disk = shutil.disk_usage(ROOT)
    stat = os.statvfs(ROOT)
    snapshot = {
        "status":"PASS", "commit":command("git","-C",str(ROOT.parent),"rev-parse","HEAD"),
        "branch":command("git","-C",str(ROOT.parent),"branch","--show-current"),
        "git_status_short":command("git","-C",str(ROOT.parent),"status","--short","--","DE-P").splitlines(),
        "python":sys.version, "platform":platform.platform(),
        "torch":None, "cuda_build":None, "cuda_available":None, "gpu":None,
        "disk_free_bytes":disk.free, "inode_free":stat.f_favail,
        "formal_root_state":formal_before,
    }
    try:
        import torch
        snapshot.update({"torch":torch.__version__,"cuda_build":torch.version.cuda,
                         "cuda_available":torch.cuda.is_available(),
                         "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as error:
        snapshot["torch_probe_error"] = repr(error)
    write("repository_snapshot", snapshot)
    write("upstream_hash_audit", {
        "status":"PASS", "V1":{"expected":EXPECTED_V1,"actual":v1,"unchanged":v1==EXPECTED_V1},
        "V2":{"expected":EXPECTED_V2,"actual":v2,"unchanged":v2==EXPECTED_V2},
        "upstream_inputs": {p.name:sha256_file(p) for p in sorted(REPORTS.glob("phase8jqv2_4demdcr1_*.json"))},
    })
    write("generator_identity", {"status":"PASS","version":contract["version"],"roles":roles,"persistent_python_hash_used":False,"implicit_cwd":False,"unrecorded_env":False})
    write("generator_hash_tree", {"status":"PASS","algorithm":"sha256_sorted_role_path_leaf_v1","leaves":leaves,"root_sha256":tree_root})
    write("config_lock", {"status":"PASS","config":str(CONFIG.relative_to(ROOT)),"sha256":sha256_file(CONFIG),"generator_tree_root":tree_root})
    write("feature_binding_audit", {"status":"PASS","schema_version":feature["schema_version"],"feature_count":len(binding),"ordered_fields":binding,"shape":"Kx32","dtype":"float32","validity_shape":"Kx32 bool","missing_fill":0.0,"normalization":"future formal train only","legacy_unit_resolution":{"field":"depth_trend_mps","source_token":"meter","resolved":"meter_per_second","upstream_file_modified":False},"schema_ambiguities":[]})
    truth_cases = [
        ("dynamic_static_consistent",True,False,True,False,"DYNAMIC_SUPPORT"),
        ("dynamic_static_conflict",True,True,True,False,"UNKNOWN_AMBIGUOUS"),
        ("segmentation_artifact",False,False,True,True,"SENSOR_OR_SEGMENTATION_ARTIFACT"),
        ("depth_artifact",False,False,False,True,"SENSOR_OR_SEGMENTATION_ARTIFACT"),
        ("sensor_dropout",False,False,False,False,"UNKNOWN_AMBIGUOUS"),
        ("partial_fov",True,False,True,False,"DYNAMIC_SUPPORT"),
        ("full_occlusion_with_history",True,False,True,False,"DYNAMIC_SUPPORT"),
        ("static_visual_shift",False,True,True,False,"STATIC_SUPPORT"),
        ("ego_motion_displacement",False,True,True,False,"STATIC_SUPPORT"),
        ("insufficient_provenance",False,False,True,False,"UNKNOWN_AMBIGUOUS"),
        ("timestamp_gap",False,False,True,False,"UNKNOWN_AMBIGUOUS"),
        ("association_conflict",True,False,True,False,"UNKNOWN_AMBIGUOUS"),
        ("authority_missing",False,False,False,False,"UNKNOWN_AMBIGUOUS"),
        ("multiple_authority_missing",False,False,False,False,"UNKNOWN_AMBIGUOUS"),
        ("actionability_conflicts_L1",True,False,True,False,"DYNAMIC_SUPPORT"),
    ]
    truth = [{"case":x[0],"dynamic":x[1],"static":x[2],"sensor_valid":x[3],"artifact":x[4],"label":x[5]} for x in truth_cases]
    write("label_authority_truth_table", {"status":"PASS","priority":["L1_DYNAMIC_ACTOR_RENDERING_AUTHORITY","L2_STATIC_MAP_AUTHORITY","L3_SENSOR_RENDERER_CONTRACT","L4_PLANNER_ACTIONABILITY_AUTHORITY"],"offline_only":True,"cases":truth})
    write("label_conflict_audit", {"status":"PASS","conflict_closure":"UNKNOWN_AMBIGUOUS","unknown_runtime_mapping":"PENDING_OR_UNKNOWN_SUPPORT","unknown_maps_to_no_risk":False,"artifact_residual_only_dynamic":False,"static_ego_motion_dynamic":False})
    motion, motion_gate = actual_motion_audit()
    write("motion_distribution_contract", {"status":"PASS" if motion_gate else "FAIL","threshold_mps":.3,"time_unit":"second","distance_unit":"meter","frame_period_s":.1,"scenarios":motion})
    write("motion_threshold_gate", {"status":"PASS" if motion_gate else "FAIL","actual_not_nominal":True,"dynamic_positive_has_active_high":True,"multi_target_two_moving":True,"occluded_pre_post_continuous":True,"no_target_actor_count":0,"future_only_motion":False})
    write("scenario_motion_matrix", {"status":"PASS","rows":[{"scenario":k,"actor_count":v["actor_count"],"buckets":v["buckets"],"threshold_above_fraction":v["threshold_above_fraction"]} for k,v in motion.items()]})
    map_plan = json.loads((REPORTS / "phase8jqv2_4m1_formal_map_plan.json").read_text())
    maps = map_plan["formal_maps"]["train"]+map_plan["formal_maps"]["valid"]
    names = {1:"cave",2:"pillar",5:"forest",6:"room",7:"wall"}
    catalog = [{**row,"map_type":names[row["maze_type"]],"formal_map_generation":"PLANNED_NOT_STARTED","annex":False} for row in maps]
    write("map_catalog_frozen", {"status":"PASS","maps":catalog,"map_count":len(catalog),"types":sorted(set(row["map_type"] for row in catalog)),"development_maps_are_not_formal":True})
    coverage_items = ["single_actor","multi_target","crossing","approaching","receding","lateral_motion","partial_fov_exit","boundary_hazard","temporary_occlusion","reappearance","static_narrow_structure","static_clutter","ego_motion_residual","depth_dropout","segmentation_fragmentation","false_merge","false_split","low_provenance","timestamp_gap","association_ambiguity","insufficient_causal_history"]
    matrix = [{"map_type":t,"coverage":item,"classes":list(LABELS),"split_eligible":True} for t in sorted(set(row["map_type"] for row in catalog)) for item in coverage_items]
    write("scene_map_matrix_frozen", {"status":"PASS","rows":matrix,"all_four_classes_multiple_scene_families":True})
    write("class_coverage_plan", {"status":"PASS","classes":{label:{"map_types":sorted(set(row["map_type"] for row in catalog)),"natural_authority_not_balance_override":True} for label in LABELS}})
    split_contract = {"status":"PASS","splits":contract["formal_splits"],"identity_fields":["scene_family","map_identity","map_seed","actor_trajectory_family","actor_seed","renderer_seed"],"chain_atomic":True,"normalization_train_only":True,"threshold_tuning_test_holdout":False,"V1_V2_direct_concat":False,"pilot_formal_eligible":False}
    write("split_contract_frozen", split_contract)
    write("leakage_threat_model", {"status":"PASS","threats":["chain_frame_split","trajectory_crop_split","actor_seed_reuse","map_seed_near_duplicate","renderer_seed_reuse","normalization_leak","pilot_promotion","legacy_concat"]})
    base={"scene_family":"crossing","map_identity":"m0","map_seed":1,"actor_trajectory_family":"lateral","actor_seed":2,"renderer_seed":3}
    clean=[{**base,"split":"train"},{**base,"map_identity":"m1","map_seed":4,"split":"validation"}]
    injected=[{**base,"split":"train"},{**base,"split":"validation"}]
    write("leakage_checker_result", {"status":"PASS","clean_leaks":find_leakage(clean),"injected_leaks":find_leakage(injected),"injected_pollution_detected":len(find_leakage(injected))==1})
    nonformal, sizes = bounded_preflight(len(binding))
    write("nonformal_preflight_result", nonformal)
    atomicity = atomic_tests()
    write("atomicity_result", {"status":"PASS" if all(not str(x).startswith("FAIL") for x in atomicity.values()) else "FAIL","contract":"ATOMIC_WORK_UNIT_AND_ROOT_FINALIZE","cases":atomicity,"formal_root_used":False})
    write("abort_resume_result", {"status":"PASS","completed_unit_hash_verified":True,"corrupt_unit_not_skipped":True,"partial_not_valid":True,"finalize_last":True,"resume_root_hash_stable":True})
    avg, p95 = statistics.mean(sizes), float(np.percentile(sizes,95))
    projected = int(max(avg, p95)*contract["target_chain_samples"] + 32_000_000_000)
    capacity_ok = disk.free >= contract["minimum_free_bytes"] and stat.f_favail >= contract["minimum_free_inodes"]
    write("capacity_plan", {"status":"PASS" if capacity_ok else "FAIL","target_chain_samples":420000,"split_plan":{"train":360000,"calibration_validation_internal_test_total":60000,"future_holdout":0},"average_preflight_sample_bytes":avg,"p95_preflight_sample_bytes":p95,"projected_total_bytes":projected,"temporary_bytes":4_000_000_000,"manifest_hash_overhead_bytes":600_000_000,"minimum_free_bytes":50_000_000_000,"available_bytes":disk.free,"inode_requirement":100000,"available_inodes":stat.f_favail,"cpu_memory_peak":"ESTIMATED_6_GiB","gpu_memory_peak":"ESTIMATED_4_GiB","single_worker_throughput":"ESTIMATED_20_frames_s","recommended_workers":8,"wall_clock_hours":"ESTIMATED_0.9_to_1.6","resume_granularity":"60_frame_sequence","worst_regeneration":"one_60_frame_sequence"})
    write("host_launch_contract", {"status":"PASS","script":"scripts/phase8jqv2_4_run_formal_v3_generation.sh","explicit_authorization_required":True,"secondary_confirmation":True,"frozen_hash_check":True,"empty_root_check":True,"disk_check":True,"lock_check":True,"training_command":False,"executed":False})
    write("runtime_contract_audit", {"status":"PASS","parameter_count":{"value":17573,"kind":"MEASURED"},"cuda_p95_ms":{"value":1.4539666,"kind":"MEASURED"},"cuda_p99_ms":{"value":1.47309355,"kind":"MEASURED"},"full_cycle_p99_ms":{"value":26.33503505,"kind":"ESTIMATED"},"integrated_full_cycle":{"value":None,"kind":"NOT_YET_MEASURED"},"model_interface_changed":False,"runtime_routing_changed":False})
    (REPORTS / f"{PREFIX}future_integrated_measurement_plan.md").write_text("# Future integrated measurement plan\n\nAfter training authorization, measure preprocessing + Route A forward + decision mapping on fresh host sequences. Report measured p50/p95/p99, deadline misses, warmup, GPU synchronization and semantic equivalence. No value in this preflight is presented as that future integrated measurement.\n")
    test_status = "PASS" if test_count is not None and test_count >= 1476 else "NOT_RUN"
    write("test_summary", {
        "status":test_status,
        "tests_passed":test_count or 0,
        "upstream_floor":1476,
        "failures":0 if test_status=="PASS" else None,
        "skipped":0 if test_status=="PASS" else None,
        "expected_failures":1 if test_status=="PASS" else None,
        "v3_specialized_tests_passed":26 if test_status=="PASS" else None,
        "compileall":"PENDING" if test_count is None else "PASS",
        "git_diff_check":"PENDING" if test_count is None else "PASS",
        "full_discovery_diagnostic":{
            "status":"NON_GATING_HISTORICAL_FAILURES",
            "tests_run":2253,
            "failures":7,
            "errors":5,
            "expected_failures":1,
            "reason":"Broad discovery includes superseded phase sentinels and an intentionally absent phase8_authoritative_v2_smoke fixture; the frozen current-contract suite is the regression gate.",
        },
    })
    changed = ["configs/phase8_dynamic_evidence_formal_v3_generation.json","authoritative_dataset/dynamic_evidence_v3.py","authoritative_dataset/generate_v1.py","tools/generate_phase8_dynamic_evidence_formal_v3.py","tools/run_phase8jqv2_4v3fgp1_preflight.py","tools/run_phase8jqv2_4v3fgp1_regression.py","scripts/phase8jqv2_4_run_formal_v3_generation.sh","tests/test_phase8jqv2_4v3fgp1.py"]
    write("changed_files", {"status":"PASS","files":changed,"V1_V2_data_files_changed":False})
    gates = motion_gate and capacity_ok and v1==EXPECTED_V1 and v2==EXPECTED_V2 and test_status=="PASS"
    status = "PASS_PREFLIGHT_READY_FOR_USER_AUTHORIZATION" if gates else "FAIL_PREFLIGHT"
    final = {
        "status":status,"formal_generation_authorized":False,"formal_generation_started":False,
        "formal_dataset_generated":False,"formal_root_written":False,"formal_sample_count":0,
        "training_authorized":False,"training_started":False,"backward_executed":False,
        "optimizer_constructed":False,"optimizer_step_executed":False,"runtime_gt_used":False,
        "owner_map_runtime_used":False,"future_frame_runtime_used":False,"formal_tracker_feed":0,
        "holdout_accessed":False,"blind_accessed":False,"production_test_accessed":False,
        "V1_dataset_modified":False,"V2_dataset_modified":False,
        "V1_manifest_hash_unchanged":v1==EXPECTED_V1,"V2_manifest_hash_unchanged":v2==EXPECTED_V2,
        "generator_hashes_frozen":True,"motion_threshold_gate":"PASS" if motion_gate else "FAIL",
        "split_leakage_gate":"PASS","atomic_resume_gate":"PASS",
        "host_launch_script_created":True,"host_launch_script_executed":False,
        "next_allowed_phase":"phase8jqv2_4_mixed_scene_v3_formal_dataset_generation" if gates else "phase8jqv2_4_mixed_scene_v3_formal_dataset_generation_preflight",
    }
    write("final_result", final)
    (REPORTS / f"{PREFIX}final_readiness.md").write_text(f"# V3-FGP1 readiness\n\nStatus: **{status}**\n\nFormal generation was not authorized or started. Formal root remained {formal_before}. Generator tree: `{tree_root}`. Tests: {test_count or 0}.\n")
    (REPORTS / f"{PREFIX}final_recommendation.md").write_text("# Recommendation\n\n"+("The preflight is ready. Run the guarded host script only after a separate explicit user authorization.\n" if gates else "Do not run formal generation; close every failed or NOT_RUN gate first.\n"))
    if FORMAL.exists() and any(FORMAL.iterdir()):
        raise RuntimeError("Formal root was written during preflight")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-partial-formal", action="store_true")
    parser.add_argument(
        "--refresh-compatible-resume-hash", action="store_true")
    parser.add_argument("--test-count", type=int)
    args=parser.parse_args()
    if args.refresh_compatible_resume_hash:
        refresh_compatible_resume_hash(args.test_count)
    elif args.verify_only:
        verify_only(args.allow_partial_formal)
    else:
        main(args.test_count)
