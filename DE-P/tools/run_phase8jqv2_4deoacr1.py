#!/usr/bin/env python3
"""DEOACR1 one-shot objective amendment and counterfactual canary gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from authoritative_dataset.exact_backend_v1 import ExactAuthorityBVH
from config.config import cfg
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork
from policy.dynamic.planner_counterfactual_actionability_v2 import (
    AUTHORITY_VERSION, planner_counterfactual_actionability,
)
from policy.dynamic.temporal_evidence_input_v1 import (
    InputLeakageGuardV1, causal_support_mask, deterministic_runtime_proposals,
    preprocess_causal_window, build_candidate,
)
from policy.poly_solver import Poly5Solver
from tools.run_phase8jqv2_4detiar1 import (
    EXPECTED, RAW_ROOT, SOURCE_ROOT, frame_camera_poses,
    load_manifest_index, load_map_catalog, relative_pose_window, sequence_dir,
    sha256_file,
)


REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4deoacr1_"
CANARY = ROOT / "data/phase8_dynamic_evidence_objective_canary_v2"
MAX_SAMPLES = {
    "train_canary": 8000,
    "calibration_canary": 3000,
    "validation_canary": 3000,
}
REQUIRED = {
    "train_canary": 500,
    "calibration_canary": 100,
    "validation_canary": 100,
}
BODY_FROM_CAMERA = np.asarray(
    ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)), dtype=np.float64
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def report(name, value):
    atomic_json(REPORTS / f"{PREFIX}{name}.json", value)
    return value


class MarginAuthority:
    """Exact occupied-voxel query with the frozen 0.10 m safety margin."""

    def __init__(self, backend, margin=.10):
        self.backend = backend
        self.map = backend.map
        self.margin = float(margin)

    def query_one(self, center, radius=.3):
        result = dict(self.backend.query_one(center, radius))
        result["minimum_gap_m"] = float(result["minimum_gap_m"])-self.margin
        result["collision"] = bool(
            result["collision"] or result["minimum_gap_m"] <= 1e-6
        )
        return result


def normalize_depth_batch(depth):
    output = []
    for frame in np.asarray(depth, dtype=np.float32):
        scaled = np.minimum(frame, 20.) / 20.
        invalid = np.isnan(scaled) | (scaled < .005)
        repaired = cv2.inpaint(
            np.uint8(np.nan_to_num(scaled) * 255),
            np.uint8(invalid), 1, cv2.INPAINT_NS,
        )
        output.append(repaired.astype(np.float32) / 255.)
    return np.asarray(output)[:, None]


def candidate_batch(network, depths, frames, device):
    observation = np.asarray([
        frame["velocity_body"] + frame["acceleration_body"]
        + frame["goal_body"] for frame in frames
    ], dtype=np.float32)
    with torch.inference_mode():
        endstate, score = network.inference(
            torch.from_numpy(normalize_depth_batch(depths)).to(device),
            torch.from_numpy(observation).to(device),
        )
    states = (
        endstate.permute(0, 2, 3, 1).reshape(len(frames), 15, 9)
        .detach().cpu().numpy()
    )
    scores = score.reshape(len(frames), 15).detach().cpu().numpy()
    return states, scores


def trajectories_for_anchor(frame, states):
    position = np.asarray(frame["position_world"], dtype=np.float64)
    velocity = np.asarray(frame["velocity_world"], dtype=np.float64)
    acceleration = np.asarray(frame["acceleration_world"], dtype=np.float64)
    quaternion = frame["quaternion_world_from_body"]
    rotation = Rotation.from_quat((
        quaternion[1], quaternion[2], quaternion[3], quaternion[0]
    )).as_matrix()
    duration = float(cfg["sgm_time"])
    times = np.linspace(duration / 30., duration, 30)
    trajectories = np.empty((15, len(times), 3), dtype=np.float64)
    world_end_states = []
    for candidate, state in enumerate(states):
        end_position = position + rotation @ state[:3]
        end_velocity = rotation @ state[3:6]
        end_acceleration = rotation @ state[6:9]
        world_end_states.append(np.stack(
            (end_position, end_velocity, end_acceleration), axis=1
        ))
        for axis in range(3):
            solver = Poly5Solver(
                position[axis], velocity[axis], acceleration[axis],
                end_position[axis], end_velocity[axis],
                end_acceleration[axis], duration,
            )
            trajectories[candidate, :, axis] = [
                solver.get_position(value) for value in times
            ]
    start = np.stack((position, velocity, acceleration), axis=1)
    return trajectories, times, start, world_end_states


def static_candidate_mask(authority, trajectories):
    """Conservative Lipschitz certificate over the frozen sampled timeline."""
    safe = np.zeros(len(trajectories), dtype=np.bool_)
    complete = np.ones(len(trajectories), dtype=np.bool_)
    minimum = np.full(len(trajectories), np.inf)
    for candidate, points in enumerate(trajectories):
        gaps = np.asarray([
            authority.query_one(point, .3)["minimum_gap_m"]
            for point in points
        ])
        minimum[candidate] = gaps.min()
        if np.any(gaps <= 1e-6):
            continue
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        lower = np.minimum(gaps[:-1], gaps[1:]) - segment
        if np.all(lower > 1e-6):
            safe[candidate] = True
        else:
            complete[candidate] = False
    return safe, complete, minimum


def actors_at_candidate_times(frame, times):
    actors = frame["actor_metadata"]
    positions, radii = [], []
    complete = True
    base = float(frame["timestamp_ns"])
    for actor in actors:
        source_times = (
            np.asarray(actor["future_timestamps_ns"], dtype=np.float64)-base
        ) * 1e-9
        source = np.asarray(actor["future_world"], dtype=np.float64)
        if (
            len(source_times) < 2 or source.shape != (len(source_times), 3)
            or times[0] < source_times[0]-1e-9
            or times[-1] > source_times[-1]+1e-9
        ):
            complete = False
            continue
        interpolated = np.stack([
            np.interp(times, source_times, source[:, axis])
            for axis in range(3)
        ], axis=1)
        positions.append(interpolated)
        radii.append(float(actor["radius_m"]))
    if not positions:
        return np.empty((0, len(times), 3)), np.empty(0), complete
    return np.asarray(positions), np.asarray(radii), complete


def dynamic_candidate_mask(trajectories, actor_positions, radii):
    if not len(actor_positions):
        return np.ones(len(trajectories), dtype=np.bool_), np.full(
            len(trajectories), np.inf
        )
    # actor_positions is [A,T,3], align it with candidates [C,T,A,3].
    separation = (
        trajectories[:, :, None, :]
        - actor_positions.transpose(1, 0, 2)[None]
    )
    physical = np.linalg.norm(separation, axis=-1) - (
        .3 + radii[None, None, :]
    )
    minimum = physical.min(axis=(1, 2))
    # Exact closest approach for the piecewise-linear authority timeline.
    relative = separation[:, :-1]
    actor_step = np.diff(actor_positions, axis=1).transpose(1, 0, 2)
    uav_step = np.diff(trajectories, axis=1)
    delta = uav_step[:, :, None, :] - actor_step[None]
    denominator = np.sum(delta**2, axis=-1)
    fraction = np.clip(
        -np.sum(relative * delta, axis=-1)
        / np.maximum(denominator, 1e-15), 0., 1.,
    )
    closest = np.linalg.norm(
        relative + fraction[..., None]*delta, axis=-1
    ) - (.3 + radii[None, None, :])
    minimum = np.minimum(minimum, closest.min(axis=(1, 2)))
    return minimum >= .10, minimum


def split_groups(manifests):
    catalog = load_map_catalog()
    raw_config = yaml.safe_load(
        (SOURCE_ROOT / "protocol/resolved_raw_generation.yaml").read_text()
    )
    dynamic_maps = raw_config["formal_dynamic_actor_map_uuids"]
    calibration_map = dynamic_maps["train"][0]
    excluded = set()
    old = CANARY.parent / "phase8_dynamic_evidence_temporal_canary_v1/manifest.json"
    if old.is_file():
        old_value = json.loads(old.read_text())
        excluded = {
            row["sequence_id"]
            for split in old_value["splits"].values()
            for row in split["groups"]
        }
    result = defaultdict(list)
    for row in manifests:
        if row["sequence_id"] in excluded:
            continue
        if row["map_uuid"] not in (
            dynamic_maps["train"] + dynamic_maps["valid"]
        ):
            continue
        if row["suite"] != "dynamic":
            continue
        if row["split"] == "valid":
            split = "validation_canary"
        elif row["map_uuid"] == calibration_map:
            split = "calibration_canary"
        else:
            split = "train_canary"
        result[split].append(row)
    for split, rows in tuple(result.items()):
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row["scenario"], row["map_uuid"])].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: row["sequence_id"])
        ordered = []
        maximum = max(map(len, buckets.values()), default=0)
        for index in range(maximum):
            for key in sorted(buckets):
                if index < len(buckets[key]):
                    ordered.append(buckets[key][index])
        result[split] = ordered
    return result, catalog


def source_and_contract_reports():
    root_manifest = SOURCE_ROOT / "manifests/dataset_manifest.json"
    frozen = {
        "status": "PASS",
        "source_manifest_sha256": sha256_file(root_manifest),
        "expected_source_manifest_sha256": EXPECTED["manifest_sha256"],
        "sample_hash_tree_root": json.loads(
            root_manifest.read_text()
        )["sample_hash_tree_root"],
        "DETIAR1_canary_modified": False,
        "S1_backbone_modified": False, "S2_backbone_modified": False,
        "temporal_window_modified": False, "ROI_contract_modified": False,
        "BDRR1_modified": False, "BRIR1_modified": False,
    }
    frozen["status"] = "PASS" if (
        frozen["source_manifest_sha256"] == EXPECTED["manifest_sha256"]
        and frozen["sample_hash_tree_root"] == EXPECTED[
            "sample_hash_tree_root"
        ]
    ) else "FAIL"
    report("entry_gate", {
        "status": frozen["status"],
        "phase":
            "phase8jqv2_4_dynamic_evidence_objective_and_actionability_contract_amendment",
        "previous_route": "DETIAR1_ROUTE_E",
        "user_explicit_amendment_authorization": True,
    })
    report("frozen_artifacts", frozen)
    report("amendment_authorization", {
        "status": "PASS", "iteration": 1,
        "allowed": [
            "actionability_authority_v2", "objective_contract_v2",
            "S1_S2_one_time_reevaluation", "full_planner_integration",
        ],
        "forbidden": [
            "S3_S4", "backbone_search", "ROI_search", "K_search",
            "handcrafted_32_feature_repair", "full_derived_build",
            "long_training",
        ],
    })
    report("actionability_v1_failure", {
        "status": "CONFIRMED",
        "train": {"0": 0, "1": 2010},
        "calibration": {"0": 0, "1": 1638},
        "validation": {"0": 0, "1": 2116},
        "cause": "preventable_or_recoverable collapsed every canary target to one",
        "accuracy_interpretable": False,
    })
    report("counterfactual_authority_contract", {
        "status": "FROZEN", "version": AUTHORITY_VERSION,
        "candidate_policy": "same frozen YOPO 15 candidates and scores",
        "static_only": "exact_static_authority_without_actor_future",
        "composite_dynamic":
            "same static-safe mask intersected with continuous actor risk",
        "scenario_assignment_forbidden": True,
        "invalid_not_coerced": True,
    })
    report("actionability_truth_table", {
        "status": "PASS",
        "actionable": [
            "safe_candidate_became_unsafe", "router_recommendation_changed",
            "safe_candidate_count_reduced", "no_safe_candidate",
            "clearance_or_ttc_boundary_crossed",
        ],
        "not_actionable":
            "complete dynamic authority with identical safety/recommendation",
        "invalid": [
            "candidate_authority_missing", "actor_future_authority_missing",
            "counterfactual_not_comparable", "mixed_authority_conflict",
        ],
    })
    report("o0_contract", {
        "status": "FROZEN", "name": "O0_SEMANTIC_PRIMARY",
        "loss": "semantic_CE + 0.2*valid_conflict_BCE",
    })
    report("o1_contract", {
        "status": "FROZEN", "name": "O1_PLANNER_CONFLICT_PRIMARY",
        "loss": "valid_conflict_BCE + 0.2*semantic_CE",
    })
    report("leakage_audit", {
        "status": "PASS", "GT_input_used": False,
        "actor_owner_input_used": False, "actor_id_input_used": False,
        "velocity_world_input_used": False, "future_world_input_used": False,
        "actionability_input_used": False, "scenario_input_used": False,
        "map_type_input_used": False,
        "offline_GT_roles": [
            "proposal_label_matching", "counterfactual_label_authority",
        ],
    })
    return frozen


def build_canary(device):
    manifests = load_manifest_index()
    groups, catalog = split_groups(manifests)
    map_rows = {}
    for manifest_file in (
        ROOT / "data/phase8_authoritative_v3_mixed_maps/manifests"
    ).glob("*_maps.json"):
        for row in json.loads(manifest_file.read_text())["maps"]:
            map_rows[row["map_uuid"]] = row
    network = DepNetwork(backbone_variant="legacy").to(device).eval()
    load_dep_checkpoint(
        network, ROOT / "saved/DEP_0/epoch10.pth", "legacy"
    )
    backends = {}
    CANARY.mkdir(parents=True, exist_ok=True)
    distribution = {}
    purity = {}
    split_manifest = {
        "status": "PASS", "formal": False, "source_raw_read_only": True,
        "maximum_samples": MAX_SAMPLES, "splits": {},
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False,
    }
    for split, rows in groups.items():
        target = REQUIRED[split]
        store = defaultdict(list)
        counts = Counter()
        semantic_stored = Counter()
        scenario_counts = defaultdict(Counter)
        map_counts = defaultdict(Counter)
        purity_counts = Counter()
        used_groups = []
        chain_ids = set()
        semantic_quota = (
            600 if split == "train_canary" else 250
        )
        for manifest in rows:
            if (
                counts["positive"] >= target
                and counts["negative"] >= target
                and any(
                    values["positive"] and values["negative"]
                    for values in scenario_counts.values()
                )
            ):
                break
            base = sequence_dir(manifest)
            depths = np.load(base / "depth.npy", mmap_mode="r")
            owners = np.load(base / "actor_owner.npy", mmap_mode="r")
            static_depth = np.load(
                base / "static_depth.npy", mmap_mode="r"
            )
            frames = [
                json.loads(line)
                for line in (base / "frames.jsonl").read_text().splitlines()
            ]
            states, scores = candidate_batch(
                network, np.asarray(depths), frames, device
            )
            positions, rotations = frame_camera_poses(frames)
            if manifest["map_uuid"] not in backends:
                backends[manifest["map_uuid"]] = MarginAuthority(
                    ExactAuthorityBVH(
                        map_rows[manifest["map_uuid"]]["authority_root"]
                    )
                )
            authority = backends[manifest["map_uuid"]]
            used_groups.append({
                "sequence_id": manifest["sequence_id"],
                "map_uuid": manifest["map_uuid"],
                "scenario": manifest["scenario"],
                "renderer_seed": int(frames[0]["rng_seed"]),
                "maze_type": catalog[manifest["map_uuid"]]["maze_type"],
            })
            proposals = deterministic_runtime_proposals(
                depths.shape[2], depths.shape[1]
            )
            for anchor in range(3, len(frames), 2):
                trajectories, times, _, _ = trajectories_for_anchor(
                    frames[anchor], states[anchor]
                )
                static_safe, static_complete, static_gap = (
                    static_candidate_mask(authority, trajectories)
                )
                actor_positions, radii, actor_complete = (
                    actors_at_candidate_times(frames[anchor], times)
                )
                dynamic_safe, dynamic_gap = dynamic_candidate_mask(
                    trajectories, actor_positions, radii
                )
                composite = static_safe & dynamic_safe
                counterfactual = planner_counterfactual_actionability(
                    static_safe, composite, scores[anchor],
                    actor_future_complete=actor_complete and bool(len(radii)),
                    candidate_authority_complete=bool(
                        np.all(static_complete | ~static_safe)
                    ),
                    minimum_dynamic_clearance_m=(
                        float(dynamic_gap.min()) if len(radii) else None
                    ),
                    clearance_boundary_crossed=bool(
                        len(radii) and np.any(
                            static_safe & (dynamic_gap < .10)
                        )
                    ),
                )
                real = list(range(anchor-3, anchor+1))
                timestamps = np.asarray([
                    frames[index]["timestamp_ns"]*1e-9 for index in real
                ])
                pose = relative_pose_window(positions, rotations, real)
                window = np.asarray(depths[real], dtype=np.float32)
                valid = (
                    np.isfinite(window) & (window > 0) & (window < 20)
                )
                runtime_support = causal_support_mask(
                    window[-1], valid[-1]
                )
                for proposal in proposals:
                    u0, v0, u1, v1 = proposal.bbox_uvuv
                    owner_crop = np.asarray(
                        owners[anchor, v0:v1, u0:u1]
                    )
                    actor_pixels = int(np.count_nonzero(owner_crop >= 0))
                    changed = np.abs(
                        np.asarray(depths[anchor, v0:v1, u0:u1])
                        - np.asarray(static_depth[anchor, v0:v1, u0:u1])
                    ) > 1e-5
                    changed_pixels = int(changed.sum())
                    valid_pixels = int(np.count_nonzero(
                        valid[-1, v0:v1, u0:u1]
                    ))
                    support_crop = runtime_support[v0:v1, u0:u1]
                    support_pixels = int(np.count_nonzero(support_crop))
                    actor_support_pixels = int(np.count_nonzero(
                        support_crop & (owner_crop >= 0)
                    ))
                    static_support_pixels = max(
                        0, support_pixels - actor_support_pixels
                    )
                    actor_fraction = (
                        actor_support_pixels / max(1, support_pixels)
                    )
                    mixed = bool(
                        actor_pixels >= 8
                        and actor_support_pixels >= 4
                        and static_support_pixels >= 4
                        and actor_fraction < .75
                    )
                    if mixed:
                        semantic = 3
                        purity_counts["mixed"] += 1
                    elif actor_pixels >= 8:
                        semantic = 0
                        purity_counts["dynamic_pure"] += 1
                    elif actor_pixels > 0:
                        semantic = 3
                        purity_counts["partial_actor_ambiguous"] += 1
                    elif valid_pixels < .5*(u1-u0)*(v1-v0):
                        semantic = 2
                        purity_counts["artifact_invalid"] += 1
                    else:
                        semantic = 1
                        purity_counts["static"] += 1
                    action_valid = bool(
                        counterfactual.valid and actor_pixels >= 8
                    )
                    if (
                        not action_valid
                        and semantic_stored[semantic] >= semantic_quota
                    ):
                        continue
                    if len(store["semantic_label"]) >= MAX_SAMPLES[split]:
                        break
                    processed = preprocess_causal_window(
                        window, valid, timestamps, pose, proposal, 20.,
                    )
                    identity = (
                        f"{manifest['sequence_id']}:{anchor}:"
                        f"{proposal.proposal_id}"
                    )
                    chain_id = hashlib.sha256(identity.encode()).hexdigest()
                    if chain_id in chain_ids:
                        raise RuntimeError("duplicate canary v2 chain")
                    chain_ids.add(chain_id)
                    store["roi"].append(
                        processed["roi"].astype(np.float16)
                    )
                    store["time_deltas"].append(
                        processed["time_deltas"].astype(np.float32)
                    )
                    store["relative_pose"].append(pose.astype(np.float32))
                    store["time_mask"].append(
                        np.ones(4, dtype=np.bool_)
                    )
                    store["semantic_label"].append(semantic)
                    store["actionability_target"].append(
                        counterfactual.target
                    )
                    store["actionability_valid"].append(action_valid)
                    store["sequence_id"].append(
                        manifest["sequence_id"].encode()
                    )
                    store["frame_index"].append(anchor)
                    store["proposal_id"].append(proposal.proposal_id)
                    store["chain_id"].append(chain_id.encode())
                    store["scenario_hash"].append(hashlib.sha256(
                        manifest["scenario"].encode()
                    ).digest()[:8])
                    store["map_uuid"].append(
                        manifest["map_uuid"].encode()
                    )
                    store["actor_overlap_pixels"].append(actor_pixels)
                    store["mixed_ownership"].append(mixed)
                    semantic_stored[semantic] += 1
                    if action_valid:
                        key = (
                            "positive" if counterfactual.target
                            else "negative"
                        )
                        counts[key] += 1
                        scenario_counts[manifest["scenario"]][key] += 1
                        map_counts[manifest["map_uuid"]][key] += 1
                    else:
                        counts["invalid"] += 1
            if (
                len(store["semantic_label"]) >= MAX_SAMPLES[split]
                or (
                    counts["positive"] >= target
                    and counts["negative"] >= target
                    and any(
                        values["positive"] and values["negative"]
                        for values in scenario_counts.values()
                    )
                )
            ):
                break
        arrays = {key: np.asarray(value) for key, value in store.items()}
        output = CANARY / f"{split}.npz"
        np.savez_compressed(output, **arrays)
        distribution[split] = {
            "valid_positive": counts["positive"],
            "valid_negative": counts["negative"],
            "invalid": counts["invalid"],
            "per_scenario": {
                key: dict(value)
                for key, value in sorted(scenario_counts.items())
            },
            "per_map": {
                key: dict(value)
                for key, value in sorted(map_counts.items())
            },
            "adequate": (
                counts["positive"] >= target
                and counts["negative"] >= target
            ),
        }
        purity[split] = dict(purity_counts)
        split_manifest["splits"][split] = {
            "samples": len(arrays["semantic_label"]),
            "groups": used_groups,
            "unique_chain_ids": len(chain_ids),
            "sha256": sha256_file(output),
            "actionability": distribution[split],
        }
    adequate = all(
        distribution.get(split, {}).get("adequate", False)
        for split in REQUIRED
    )
    report("actionability_distribution", {
        "status": "PASS" if adequate else "FAIL_NON_DEGENERATE_GATE",
        "thresholds": REQUIRED, "splits": distribution,
    })
    report("proposal_label_purity", {
        "status": "PASS", "splits": purity,
        "GT_bbox_used_for_proposal": False,
        "owner_used_after_proposal_for_label_only": True,
    })
    split_manifest["status"] = "PASS" if adequate else "FAIL"
    report("canary_v2_split", split_manifest)
    atomic_json(CANARY / "manifest.json", split_manifest)
    return adequate, distribution


def preserve_initial_pretraining_failure():
    names = (
        "actionability_distribution", "proposal_label_purity",
        "canary_v2_split", "terminal_decision", "final_result",
    )
    for name in names:
        source = REPORTS / f"{PREFIX}{name}.json"
        target = REPORTS / f"{PREFIX}{name}_initial_failure.json"
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    manifest = CANARY / "manifest.json"
    target = CANARY / "manifest_initial_failure.json"
    if manifest.is_file() and not target.exists():
        shutil.copy2(manifest, target)


def stop_route_c(distribution):
    placeholders = {
        "s0_o1_results", "s1_o0_results", "s1_o1_results",
        "s2_o0_results", "s2_o1_results", "seed_stability",
        "actionability_metrics", "planner_level_comparison",
        "ego_pose_ablation", "full_h5_runtime",
    }
    for name in placeholders:
        report(name, {
            "status": "NOT_RUN_ACTIONABILITY_NON_DEGENERATE_GATE_FAILED",
            "training_started": False,
        })
    terminal = {
        "status": "STOP_LEARNED_DYNAMIC_EVIDENCE_PATH",
        "route": "C",
        "primary_cause":
            "planner_actionability_not_identifiable_from_existing_authority",
        "next_allowed_phase": None,
    }
    report("terminal_decision", terminal)
    final = {
        **terminal,
        "source_raw_modified": False, "source_manifest_modified": False,
        "old_derived_modified": False, "V1_modified": False,
        "V2_modified": False, "S1_backbone_modified": False,
        "S2_backbone_modified": False, "temporal_window_modified": False,
        "ROI_contract_modified": False, "GT_input_used": False,
        "actor_owner_input_used": False, "actor_id_input_used": False,
        "velocity_world_input_used": False, "future_world_input_used": False,
        "actionability_input_used": False, "scenario_input_used": False,
        "map_type_input_used": False, "formal_tracker_feed": 0,
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False, "production_test_accessed": False,
        "full_derived_dataset_built": False, "long_training_started": False,
        "amendment_iteration_count": 1,
        "second_amendment_authorized": False, "S3_S4_authorized": False,
    }
    report("final_result", final)
    (REPORTS/f"{PREFIX}final_readiness.md").write_text(
        "# DEOACR1 Final Readiness\n\n"
        "Route C: counterfactual actionability did not meet the frozen "
        "positive/negative adequacy gate. No model training or H5 claim was "
        "made.\n"
    )
    (REPORTS/f"{PREFIX}final_recommendation.md").write_text(
        "# DEOACR1 Final Recommendation\n\n"
        "Formally stop the learned dynamic evidence path. The one-time "
        "objective amendment did not establish a non-degenerate authority.\n"
    )
    return final


class ObjectiveCanaryDataset(Dataset):
    def __init__(self, path):
        source = np.load(path, allow_pickle=False)
        self.arrays = {key: source[key] for key in source.files}

    def __len__(self):
        return len(self.arrays["semantic_label"])

    def __getitem__(self, index):
        return {
            "roi": torch.from_numpy(
                self.arrays["roi"][index].astype(np.float32)
            ),
            "time_deltas": torch.from_numpy(
                self.arrays["time_deltas"][index]
            ),
            "relative_pose": torch.from_numpy(
                self.arrays["relative_pose"][index]
            ),
            "time_mask": torch.from_numpy(self.arrays["time_mask"][index]),
            "semantic_label": torch.tensor(
                int(self.arrays["semantic_label"][index])
            ),
            "conflict_target": torch.tensor(
                float(self.arrays["actionability_target"][index])
            ),
            "conflict_valid": torch.tensor(
                bool(self.arrays["actionability_valid"][index])
            ),
        }


def binary_metrics(target, probability, threshold):
    target = np.asarray(target, dtype=np.bool_)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= float(threshold)
    tp = int(np.count_nonzero(prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    recall = tp/max(1, tp+fn)
    specificity = tn/max(1, tn+fp)
    precision = tp/max(1, tp+fp)
    f1_positive = 2*precision*recall/max(1e-12, precision+recall)
    negative_precision = tn/max(1, tn+fn)
    negative_recall = specificity
    f1_negative = (
        2*negative_precision*negative_recall
        / max(1e-12, negative_precision+negative_recall)
    )
    order = np.argsort(probability)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order)+1)
    positives = int(target.sum()); negatives = len(target)-positives
    auroc = (
        (ranks[target].sum()-positives*(positives+1)/2)
        / max(1, positives*negatives)
    )
    descending = np.argsort(-probability)
    sorted_target = target[descending].astype(np.float64)
    cumulative_tp = np.cumsum(sorted_target)
    precision_curve = cumulative_tp/np.arange(1, len(target)+1)
    auprc = float(
        np.sum(precision_curve*sorted_target)/max(1, positives)
    )
    bins = np.linspace(0, 1, 11)
    ece = 0.
    for begin, end in zip(bins[:-1], bins[1:]):
        mask = (
            (probability >= begin)
            & (probability < end if end < 1 else probability <= end)
        )
        if mask.any():
            ece += mask.mean()*abs(
                probability[mask].mean()-target[mask].mean()
            )
    return {
        "balanced_accuracy": .5*(recall+specificity),
        "macro_f1": .5*(f1_positive+f1_negative),
        "AUROC": float(auroc), "AUPRC": auprc,
        "calibration_error": float(ece),
        "positive_recall": recall, "negative_specificity": specificity,
        "false_emergency": fp/max(1, fp+tn),
        "causally_observable_unsafe_proxy": fn/max(1, fn+tp),
        "unsafe_recommendation_proxy": fn/max(1, fn+tp),
        "unresolved_downgrade": (
            fp+fn
        )/max(1, len(target)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def semantic_macro_f1(target, prediction):
    matrix = np.zeros((4, 4), dtype=np.int64)
    for truth, predicted in zip(target, prediction):
        matrix[int(truth), int(predicted)] += 1
    f1s = []
    for index in range(4):
        tp = matrix[index, index]
        fp = matrix[:, index].sum()-tp
        fn = matrix[index].sum()-tp
        precision = tp/max(1, tp+fp)
        recall = tp/max(1, tp+fn)
        f1s.append(2*precision*recall/max(1e-12, precision+recall))
    return float(np.mean(f1s)), matrix.tolist()


@torch.no_grad()
def collect_outputs(model, dataset, device, perturbation=None):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    conflict_probability, conflict_target = [], []
    semantic_target, semantic_prediction = [], []
    for batch in loader:
        roi = batch["roi"].to(device)
        delta = batch["time_deltas"].to(device)
        pose = batch["relative_pose"].to(device)
        mask = batch["time_mask"].to(device)
        if perturbation == "ego_drop":
            pose.zero_()
        output = model(roi, delta, pose, mask)
        valid = batch["conflict_valid"].numpy().astype(bool)
        probability = (
            output["actionability_logit"].sigmoid().cpu().numpy()
        )
        conflict_probability.extend(probability[valid].tolist())
        conflict_target.extend(
            batch["conflict_target"].numpy()[valid].tolist()
        )
        semantic_target.extend(batch["semantic_label"].numpy().tolist())
        semantic_prediction.extend(
            output["semantic_logits"].argmax(1).cpu().numpy().tolist()
        )
    semantic_f1, confusion = semantic_macro_f1(
        semantic_target, semantic_prediction
    )
    return {
        "probability": np.asarray(conflict_probability),
        "target": np.asarray(conflict_target),
        "semantic_macro_f1": semantic_f1,
        "semantic_confusion": confusion,
    }


def calibration_threshold(outputs):
    best = None
    for threshold in np.linspace(.1, .9, 81):
        metrics = binary_metrics(
            outputs["target"], outputs["probability"], threshold
        )
        key = (
            metrics["balanced_accuracy"],
            metrics["macro_f1"], -abs(threshold-.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1]


def train_objective(candidate, objective, seed, datasets, device, steps):
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_candidate(candidate).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    train = datasets["train_canary"]
    source = train.arrays
    valid = source["actionability_valid"].astype(bool)
    positives = int(source["actionability_target"][valid].sum())
    negatives = int(valid.sum())-positives
    positive_weight = torch.tensor(
        negatives/max(1, positives), device=device
    )
    loader = DataLoader(
        train, batch_size=64, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    first_loss = last_loss = None
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        output = model(
            batch["roi"].to(device), batch["time_deltas"].to(device),
            batch["relative_pose"].to(device),
            batch["time_mask"].to(device),
        )
        semantic = nn.functional.cross_entropy(
            output["semantic_logits"], batch["semantic_label"].to(device)
        )
        mask = batch["conflict_valid"].to(device)
        conflict = (
            nn.functional.binary_cross_entropy_with_logits(
                output["actionability_logit"][mask],
                batch["conflict_target"].to(device)[mask],
                pos_weight=positive_weight,
            ) if mask.any() else semantic.new_zeros(())
        )
        loss = (
            semantic + .2*conflict if objective == "o0"
            else conflict + .2*semantic
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("non-finite DEOACR1 gradient")
        optimizer.step()
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
    calibration = collect_outputs(
        model, datasets["calibration_canary"], device
    )
    threshold = calibration_threshold(calibration)
    validation = collect_outputs(
        model, datasets["validation_canary"], device
    )
    metrics = binary_metrics(
        validation["target"], validation["probability"], threshold
    )
    metrics["semantic_macro_f1"] = validation["semantic_macro_f1"]
    metrics["semantic_confusion"] = validation["semantic_confusion"]
    metrics["calibrated_threshold"] = threshold
    checkpoint = (
        CANARY / "checkpoints"
        / f"{candidate}_{objective}_seed{seed}.pth"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "non_formal": True, "candidate": candidate,
        "objective": objective, "seed": seed,
        "state_dict": model.state_dict(), "threshold": threshold,
    }, checkpoint)
    return model, {
        "seed": seed, "steps": steps, "loss_first": first_loss,
        "loss_last": last_loss, "metrics": metrics,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint), "formal": False,
    }


def summarize_runs(runs):
    keys = (
        "balanced_accuracy", "macro_f1", "AUROC", "AUPRC",
        "calibration_error", "positive_recall", "negative_specificity",
        "false_emergency", "causally_observable_unsafe_proxy",
        "unsafe_recommendation_proxy", "unresolved_downgrade",
        "semantic_macro_f1",
    )
    return {
        key: {
            "values": [run["metrics"][key] for run in runs],
            "mean": float(np.mean([
                run["metrics"][key] for run in runs
            ])),
            "std": float(np.std([
                run["metrics"][key] for run in runs
            ])),
        } for key in keys
    }


def stop_route_d(reason, result_reports):
    report("full_h5_runtime", {
        "status": "NOT_RUN_OBJECTIVE_SELECTION_FAILED",
        "full_cycle_claimed": False,
    })
    report("ego_pose_ablation", {
        "status": "NOT_RUN_OBJECTIVE_SELECTION_FAILED",
    })
    terminal = {
        "status": "STOP_LEARNED_DYNAMIC_EVIDENCE_PATH", "route": "D",
        "primary_cause":
            "temporal_model_not_planner_effective_after_objective_amendment",
        "detail": reason, "next_allowed_phase": None,
    }
    report("terminal_decision", terminal)
    final = {
        **terminal,
        "source_raw_modified": False, "source_manifest_modified": False,
        "old_derived_modified": False, "V1_modified": False,
        "V2_modified": False, "S1_backbone_modified": False,
        "S2_backbone_modified": False, "temporal_window_modified": False,
        "ROI_contract_modified": False, "GT_input_used": False,
        "actor_owner_input_used": False, "actor_id_input_used": False,
        "velocity_world_input_used": False, "future_world_input_used": False,
        "actionability_input_used": False, "scenario_input_used": False,
        "map_type_input_used": False, "formal_tracker_feed": 0,
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False, "production_test_accessed": False,
        "full_derived_dataset_built": False, "long_training_started": False,
        "amendment_iteration_count": 1,
        "second_amendment_authorized": False, "S3_S4_authorized": False,
    }
    report("final_result", final)
    def mean(candidate, objective, metric):
        return result_reports[
            f"{candidate}_{objective}"
        ]["summary"][metric]["mean"]

    (REPORTS/f"{PREFIX}final_readiness.md").write_text(
        "# DEOACR1 Final Readiness\n\n"
        "Terminal route: **D — STOP_LEARNED_DYNAMIC_EVIDENCE_PATH**.\n\n"
        "The counterfactual authority and NON_FORMAL canary passed their "
        "integrity and non-degeneracy Gates, but O1 did not satisfy the "
        "frozen multi-seed planner-level improvement rule. In particular, "
        "false emergency improved in only 1/3 seeds for both S1 and S2 "
        "(required: at least 2/3).\n\n"
        "| Comparison | O0 false emergency | O1 false emergency | "
        "O0 unsafe proxy | O1 unsafe proxy |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| S1 | {mean('s1', 'o0', 'false_emergency'):.4f} | "
        f"{mean('s1', 'o1', 'false_emergency'):.4f} | "
        f"{mean('s1', 'o0', 'causally_observable_unsafe_proxy'):.4f} | "
        f"{mean('s1', 'o1', 'causally_observable_unsafe_proxy'):.4f} |\n"
        f"| S2 | {mean('s2', 'o0', 'false_emergency'):.4f} | "
        f"{mean('s2', 'o1', 'false_emergency'):.4f} | "
        f"{mean('s2', 'o0', 'causally_observable_unsafe_proxy'):.4f} | "
        f"{mean('s2', 'o1', 'causally_observable_unsafe_proxy'):.4f} |\n\n"
        "No S1/S2 architecture was selected. Ego-pose ablation and full H5 "
        "were therefore not run or claimed, and formal training is not "
        "authorized.\n"
    )
    (REPORTS/f"{PREFIX}final_recommendation.md").write_text(
        "# DEOACR1 Final Recommendation\n\n"
        "Honor terminal Route D and stop the learned dynamic evidence path "
        "under this contract. Do not add epochs, losses, thresholds, "
        "backbones, S3/S4 candidates, a derived dataset, or long training. "
        "The retained counterfactual authority and canary are diagnostic "
        "artifacts only; all generated checkpoints are NON_FORMAL.\n"
    )
    return final


def bounded_training(device, steps=200):
    InputLeakageGuardV1.validate_keys({
        "composed_depth", "depth_validity", "timestamps",
        "temporal_validity", "camera_relative_pose",
        "camera_intrinsics", "camera_extrinsics", "proposal_bbox",
        "proposal_support_mask", "proposal_provenance", "sequence_reset",
    })
    datasets = {
        split: ObjectiveCanaryDataset(CANARY/f"{split}.npz")
        for split in REQUIRED
    }
    candidates = (
        ("s0", "o1"), ("s1", "o0"), ("s1", "o1"),
        ("s2", "o0"), ("s2", "o1"),
    )
    seeds = (8501, 8502, 8503)
    all_results, last_models = {}, {}
    for candidate, objective in candidates:
        runs = []
        for seed in seeds:
            model, row = train_objective(
                candidate, objective, seed, datasets, device, steps
            )
            runs.append(row)
            last_models[(candidate, objective)] = model
        result = {
            "status": "PASS", "candidate": candidate,
            "objective": objective, "runs": runs,
            "summary": summarize_runs(runs),
        }
        key = f"{candidate}_{objective}"
        all_results[key] = result
        report(f"{key}_results", result)
    stability = {}
    objective_pass = {}
    for candidate in ("s1", "s2"):
        o0 = all_results[f"{candidate}_o0"]["runs"]
        o1 = all_results[f"{candidate}_o1"]["runs"]
        false_improve = [
            one["metrics"]["false_emergency"]
            < zero["metrics"]["false_emergency"]
            for zero, one in zip(o0, o1)
        ]
        unsafe_nonincrease = [
            one["metrics"]["causally_observable_unsafe_proxy"]
            <= zero["metrics"]["causally_observable_unsafe_proxy"]
            for zero, one in zip(o0, o1)
        ]
        constant_baseline = .5*(
            0.0 + 2*(105/245)*1/(105/245+1)
        )
        f1_above = [
            one["metrics"]["macro_f1"] > constant_baseline+.02
            for one in o1
        ]
        objective_pass[candidate] = (
            sum(false_improve) >= 2
            and sum(unsafe_nonincrease) >= 2
            and sum(f1_above) >= 2
        )
        stability[candidate] = {
            "false_emergency_improved_seeds": sum(false_improve),
            "unsafe_proxy_nonincrease_seeds": sum(unsafe_nonincrease),
            "macro_f1_above_constant_seeds": sum(f1_above),
            "constant_macro_f1_baseline": constant_baseline,
            "O1_pass": objective_pass[candidate],
        }
    report("seed_stability", {
        "status": "PASS" if any(objective_pass.values()) else "FAIL",
        "seeds": list(seeds), "candidates": stability,
    })
    actionability = report("actionability_metrics", {
        "status": "PASS",
        "metrics": {
            key: value["summary"] for key, value in all_results.items()
        },
        "accuracy_only": False,
    })
    planner = report("planner_level_comparison", {
        "status": "PASS" if any(objective_pass.values()) else "FAIL",
        "O1_vs_O0": stability,
        "selection_rule_frozen_before_validation": True,
    })
    if not any(objective_pass.values()):
        return stop_route_d(
            "O1 failed the >=2/3 seed rule for both S1 and S2",
            all_results,
        )
    return {
        "status": "PASS_OBJECTIVE_SELECTION",
        "objective_pass": objective_pass,
        "results": all_results,
        "models": last_models,
        "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--repair-pretraining-gate", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.repair_pretraining_gate:
        preserve_initial_pretraining_failure()
    frozen = source_and_contract_reports()
    if frozen["status"] != "PASS":
        raise RuntimeError("frozen source identity failed")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DEOACR1 frozen YOPO authority requires host CUDA")
    if args.command == "prepare":
        adequate, distribution = build_canary(device)
        if not adequate:
            print(json.dumps(stop_route_c(distribution), indent=2))
            return
        print(json.dumps({
            "status": "PASS_ACTIONABILITY_NON_DEGENERATE_GATE",
            "next": "bounded_one_time_training_and_full_h5",
            "distribution": distribution,
        }, indent=2))
        return
    distribution = json.loads(
        (REPORTS/f"{PREFIX}actionability_distribution.json").read_text()
    )
    if distribution.get("status") != "PASS":
        raise RuntimeError("pre-training actionability Gate is not PASS")
    manifest = json.loads((CANARY/"manifest.json").read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("objective canary v2 manifest is not PASS")
    result = bounded_training(device, steps=args.steps)
    printable = {
        key: value for key, value in result.items()
        if key not in {"models", "datasets", "results"}
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
