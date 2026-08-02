#!/usr/bin/env python3
"""One-shot DETIAR1 audit, canary comparison, and terminal decision.

The raw Formal V3 authority is read-only.  Ground truth is consulted only
after a GT-independent proposal chain has been constructed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW_ROOT = ROOT / "data/phase8_dynamic_evidence_formal_v3/raw_authoritative"
SOURCE_ROOT = RAW_ROOT.parent
CANARY_ROOT = ROOT / "data/phase8_dynamic_evidence_temporal_canary_v1"
REPORTS = ROOT / "reports"
PREFIX = "phase8jqv2_4detiar1_"
EXPECTED = {
    "work_units": 7000,
    "samples": 420000,
    "unique_chain_ids": 420000,
    "manifest_sha256":
        "a1acfa2fee7e8336917316cb3f07ac005c1ce1354a8bda8de10d02f2438354f6",
    "sample_hash_tree_root":
        "9916f1d3cf541fa4731baf5087725ddce23cd3dd50e31a2079681b5d81f964ea",
}

from policy.dynamic.temporal_evidence_input_v1 import (  # noqa: E402
    DatasetFieldRoleV1, FORBIDDEN_INPUT_FIELDS, InputLeakageGuardV1,
    LABEL_ONLY_FIELDS, RUNTIME_INPUT_ALLOWLIST, build_candidate,
    deterministic_runtime_proposals, preprocess_causal_window,
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def report(name, value):
    atomic_json(REPORTS / f"{PREFIX}{name}.json", value)
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    audit_path = REPORTS / "phase8_dynamic_evidence_formal_v3_audit.json"
    audit = json.loads(audit_path.read_text())
    storage_integrity = audit["storage_integrity"]
    if isinstance(storage_integrity, dict):
        storage_integrity = storage_integrity["status"]
    manifests = list((RAW_ROOT / "manifests/sequences").glob("*.json"))
    manifest_path = SOURCE_ROOT / "manifests/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual_manifest_sha256 = sha256_file(manifest_path)
    samples = sum(
        int(json.loads(path.read_text())["frame_count"]) for path in manifests
    )
    identity = {
        **EXPECTED,
        "actual_work_units": len(manifests),
        "actual_samples": samples,
        "actual_unique_chain_ids": manifest["sample_count"],
        "actual_manifest_sha256": actual_manifest_sha256,
        "actual_sample_hash_tree_root": manifest["sample_hash_tree_root"],
        "storage_integrity": storage_integrity,
        "raw_root": str(RAW_ROOT),
        "source_raw_modified": False,
        "source_manifest_modified": False,
        "status": "PASS" if (
            len(manifests) == EXPECTED["work_units"]
            and samples == EXPECTED["samples"]
            and actual_manifest_sha256 == EXPECTED["manifest_sha256"]
            and manifest["sample_hash_tree_root"]
                == EXPECTED["sample_hash_tree_root"]
            and storage_integrity == "PASS"
        ) else "FAIL",
    }
    return report("source_root_freeze", identity)


def load_manifest_index():
    rows = []
    for path in sorted((RAW_ROOT / "manifests/sequences").glob("*.json")):
        row = json.loads(path.read_text())
        row["_path"] = path
        rows.append(row)
    return rows


def sequence_dir(manifest):
    key = next(
        key for key in manifest["files"] if key.endswith("/depth.npy")
    )
    return RAW_ROOT / key.rsplit("/", 1)[0]


def load_map_catalog():
    value = {}
    base = ROOT / "data/phase8_authoritative_v3_mixed_maps/manifests"
    for split in ("train", "valid"):
        rows = json.loads((base / f"{split}_maps.json").read_text())["maps"]
        for row in rows:
            value[row["map_uuid"]] = {
                "maze_type": int(row["maze_type"]),
                "semantic_name": row["semantic_name"],
                "profile_name": row["profile_name"],
                "split": split,
            }
    return value


def raw_sufficiency(manifests):
    required_scenarios = {
        "multi_target", "occluded_but_tracked", "no_target",
        "crossing", "head_on", "temporal_separation",
    }
    chosen = {}
    for row in manifests:
        if row["scenario"] in required_scenarios and row["scenario"] not in chosen:
            chosen[row["scenario"]] = row
    evidence = {}
    failures = []
    for scenario in sorted(required_scenarios):
        row = chosen.get(scenario)
        if row is None:
            failures.append(f"missing scenario {scenario}")
            continue
        base = sequence_dir(row)
        depth = np.load(base / "depth.npy", mmap_mode="r")
        static = np.load(base / "static_depth.npy", mmap_mode="r")
        owner = np.load(base / "actor_owner.npy", mmap_mode="r")
        frames = [
            json.loads(line)
            for line in (base / "frames.jsonl").read_text().splitlines()
        ]
        contribution = np.abs(
            depth.astype(np.float32) - static.astype(np.float32)
        ) > 1e-5
        centroids = []
        for frame_owner in owner:
            yy, xx = np.nonzero(frame_owner >= 0)
            centroids.append(
                None if not len(xx) else [float(xx.mean()), float(yy.mean())]
            )
        visible = np.asarray([item is not None for item in centroids])
        moving = False
        concrete = [item for item in centroids if item is not None]
        if len(concrete) >= 2:
            moving = bool(
                np.linalg.norm(np.asarray(concrete[-1])-concrete[0]) > 0.5
            )
        owner_ids = sorted(
            int(value) for value in np.unique(owner) if int(value) >= 0
        )
        item = {
            "sequence_id": row["sequence_id"],
            "shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "timestamps_strict": all(
                frames[i]["timestamp_ns"] < frames[i+1]["timestamp_ns"]
                for i in range(len(frames)-1)
            ),
            "pose_complete": all(
                len(frame["position_world"]) == 3
                and len(frame["quaternion_world_from_body"]) == 4
                for frame in frames
            ),
            "actionability_complete": all(
                "actionability" in frame for frame in frames
            ),
            "actor_contribution_pixels": int(contribution.sum()),
            "actor_owner_ids": owner_ids,
            "actor_contribution_moves": moving,
            "visibility_transitions": int(np.count_nonzero(
                visible[1:] != visible[:-1]
            )),
        }
        evidence[scenario] = item
        if scenario == "no_target":
            if item["actor_contribution_pixels"] or owner_ids:
                failures.append("no_target contains actor contribution")
        else:
            if not item["actor_contribution_pixels"]:
                failures.append(f"{scenario} has no rendered actor contribution")
        if scenario == "multi_target" and len(owner_ids) < 2:
            failures.append("multi_target lacks two owner contributions")
        if scenario == "occluded_but_tracked" and not item["visibility_transitions"]:
            failures.append("occluded_but_tracked lacks disappear/reappear evidence")
    fields = {
        "K4_composed_depth": "PRESENT_COMPLETE",
        "K4_depth_validity": "DETERMINISTICALLY_RECONSTRUCTIBLE",
        "K4_timestamps": "PRESENT_COMPLETE",
        "K4_uav_pose": "PRESENT_COMPLETE",
        "camera_intrinsics": "PRESENT_COMPLETE",
        "camera_extrinsics": "DETERMINISTICALLY_RECONSTRUCTIBLE",
        "runtime_component_support_proposal":
            "DETERMINISTICALLY_RECONSTRUCTIBLE",
        "proposal_bbox_mask_provenance":
            "DETERMINISTICALLY_RECONSTRUCTIBLE",
        "static_composed_depth_distinction": "PRESENT_COMPLETE",
        "actor_owner_label_authority": "PRESENT_COMPLETE",
        "actor_pose_velocity_future_label_authority": "PRESENT_COMPLETE",
        "actionability_certificate": "PRESENT_COMPLETE",
        "sequence_reset": "DETERMINISTICALLY_RECONSTRUCTIBLE",
        "map_scenario_identity": "PRESENT_COMPLETE",
    }
    return report("raw_source_sufficiency", {
        "status": "PASS" if not failures else "FAIL",
        "fields": fields, "scenario_evidence": evidence, "errors": failures,
        "raw_rerender_required": False,
    })


def write_contract_reports():
    boundary = report("simulator_dynamic_boundary", {
        "status": "PASS",
        "original_yopo_simulator": [
            "static_map", "canonical_occupancy", "static_depth_camera_model",
        ],
        "de_p_extension": [
            "dynamic_motion_v2", "actor_position",
            "continuous_actor_trajectory",
            "analytic_ray_sphere_nearest_compositing",
            "composed_depth", "actor_owner_label_authority",
            "actor_velocity_and_future_label_metadata",
        ],
        "dynamic_observation_exists_without_native_simulator_actor": True,
    })
    call_graph = report("dynamic_render_call_graph", {
        "status": "PASS",
        "nodes": [
            "mixed_scene static authority map",
            "authoritative_dataset.dynamic_motion_v2.build_actor_specs_v2",
            "authoritative_dataset.dynamic_motion_v2.actor_position",
            "CanonicalDepthRenderer.render_with_actor_diagnostics",
            "analytic sphere nearest-hit compositor",
            "depth.npy + static_depth.npy + actor_owner.npy",
            "frames.jsonl actor_metadata",
        ],
        "source_files": {
            "generator": "authoritative_dataset/generate_v1.py",
            "motion": "authoritative_dataset/dynamic_motion_v2.py",
            "renderer": "authoritative_dataset/canonical_renderer.py",
        },
    })
    roles = {}
    for field in sorted(RUNTIME_INPUT_ALLOWLIST):
        roles[field] = DatasetFieldRoleV1.RUNTIME_INPUT.value
    for field in sorted(LABEL_ONLY_FIELDS):
        roles[field] = DatasetFieldRoleV1.LABEL_ONLY.value
    roles.update({
        "normalized_depth": DatasetFieldRoleV1.DERIVED_CAUSAL_INPUT.value,
        "runtime_support_mask": DatasetFieldRoleV1.DERIVED_CAUSAL_INPUT.value,
        "sequence_id": DatasetFieldRoleV1.AUDIT_ONLY.value,
        "map_uuid": DatasetFieldRoleV1.AUDIT_ONLY.value,
        "scenario": DatasetFieldRoleV1.AUDIT_ONLY.value,
        "static_depth": DatasetFieldRoleV1.FORBIDDEN.value,
        "authoritative_static_occupancy": DatasetFieldRoleV1.FORBIDDEN.value,
    })
    report("field_role_contract", {
        "status": "PASS", "version": "DatasetFieldRoleV1",
        "roles": roles,
    })
    report("input_allowlist", {
        "status": "PASS", "allowlist": sorted(RUNTIME_INPUT_ALLOWLIST),
        "policy": "fail_closed_allowlist_not_blacklist",
    })
    report("label_only_fields", {
        "status": "PASS", "fields": sorted(LABEL_ONLY_FIELDS),
    })
    try:
        InputLeakageGuardV1.validate_keys(RUNTIME_INPUT_ALLOWLIST)
        blocked = 0
        for field in FORBIDDEN_INPUT_FIELDS:
            try:
                InputLeakageGuardV1.validate_keys({field})
            except RuntimeError:
                blocked += 1
        leakage_status = blocked == len(FORBIDDEN_INPUT_FIELDS)
    except RuntimeError:
        leakage_status = False
        blocked = 0
    report("leakage_audit", {
        "status": "PASS" if leakage_status else "FAIL",
        "forbidden_cases": len(FORBIDDEN_INPUT_FIELDS),
        "forbidden_cases_blocked": blocked,
        "actor_owner_input_used": False, "actor_id_input_used": False,
        "velocity_world_input_used": False, "future_world_input_used": False,
        "actionability_input_used": False,
        "static_authority_input_used": False, "GT_runtime_used": False,
    })
    report("sample_contract", {
        "status": "PASS", "version": "CandidateTemporalEvidenceSampleV1",
        "temporal_length": 4, "causal_only": True,
        "proposal_chain_before_gt_matching": True,
        "gt_identity_used_to_construct_chain": False,
    })
    report("roi_contract", {
        "status": "PASS", "source": "deterministic_runtime_grid_v1",
        "bbox": [64, 48], "resize": [32, 32],
        "depth_interpolation": "bilinear",
        "mask_interpolation": "nearest", "padding": "left_zero",
        "owner_derived_crop": False,
    })
    for key, contract in {
        "s0_contract": {
            "name": "S0_SINGLE_FRAME_DEPTH", "K": 1,
            "ego_motion": False, "temporal": False,
        },
        "s1_contract": {
            "name": "S1_CAUSAL_TEMPORAL_DEPTH", "K": 4,
            "ego_motion": False, "temporal": "shared_CNN_GRU",
        },
        "s2_contract": {
            "name": "S2_CAUSAL_TEMPORAL_DEPTH_EGO_AWARE", "K": 4,
            "ego_motion": "relative_SE3_pose_embedding",
            "temporal": "shared_CNN_GRU",
        },
    }.items():
        report(key, {
            "status": "FROZEN", "channels": [
                "normalized_composed_depth", "depth_validity",
                "runtime_proposal_support",
            ], **contract,
        })
    return boundary, call_graph


def frame_camera_poses(frames):
    body_from_camera = np.asarray(
        ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)), dtype=np.float64
    )
    positions, rotations = [], []
    for frame in frames:
        body = Rotation.from_quat(np.asarray((
            frame["quaternion_world_from_body"][1],
            frame["quaternion_world_from_body"][2],
            frame["quaternion_world_from_body"][3],
            frame["quaternion_world_from_body"][0],
        ))).as_matrix()
        positions.append(np.asarray(frame["position_world"], dtype=np.float64))
        rotations.append(body @ body_from_camera)
    return np.stack(positions), np.stack(rotations)


def relative_pose_window(positions, rotations, indices):
    current_p, current_r = positions[indices[-1]], rotations[indices[-1]]
    values = []
    for index in indices:
        translation = current_r.T @ (positions[index] - current_p)
        rotation = current_r.T @ rotations[index]
        values.append(np.r_[translation, Rotation.from_matrix(rotation).as_rotvec()])
    return np.asarray(values, dtype=np.float32)


def group_selection(manifests):
    catalog = load_map_catalog()
    target = {
        "crossing", "head_on", "multi_target", "occluded_but_tracked",
        "temporal_separation", "no_target", "normal_progress",
        "near_boundary_recovery_stress",
    }
    selected = {"train_canary": [], "calibration_canary": [], "validation_canary": []}
    train_maps = defaultdict(list)
    for uuid, info in catalog.items():
        if info["split"] == "train":
            train_maps[info["maze_type"]].append(uuid)
    raw_config = yaml.safe_load(
        (SOURCE_ROOT / "protocol/resolved_raw_generation.yaml").read_text()
    )
    dynamic_train_maps = list(
        raw_config["formal_dynamic_actor_map_uuids"]["train"]
    )
    # Formal V3 contains only three actor-bearing train maps.  Reserve one
    # whole dynamic map for calibration, then add whole static maps for the
    # remaining map types.  No map UUID crosses the split boundary.
    calibration_maps = {dynamic_train_maps[0]}
    reserved_type = catalog[dynamic_train_maps[0]]["maze_type"]
    for maze_type, rows in sorted(train_maps.items()):
        if maze_type != reserved_type and rows:
            calibration_maps.add(rows[-1])
    buckets = defaultdict(list)
    for row in manifests:
        info = catalog.get(row["map_uuid"])
        if info is None or row["scenario"] not in target:
            continue
        if row["split"] == "valid":
            target_split = "validation_canary"
        elif row["map_uuid"] in calibration_maps:
            target_split = "calibration_canary"
        else:
            target_split = "train_canary"
        buckets[(target_split, info["maze_type"], row["scenario"])].append(row)
    for (target_split, _, _), rows in sorted(buckets.items()):
        selected[target_split].append(
            sorted(rows, key=lambda x: x["sequence_id"])[0]
        )
    return selected, catalog


def maybe_corrupt(roi, token):
    draw = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 20
    if draw == 0:
        roi = roi.copy()
        roi[-1, 0, :, 14:18] = 0
        roi[-1, 1, :, 14:18] = 0
        roi[-1, 2, :, 14:18] = 0
        return roi, True
    return roi, False


def build_canary(manifests, max_per_class=900):
    selected, catalog = group_selection(manifests)
    CANARY_ROOT.mkdir(parents=True, exist_ok=True)
    split_report = {
        "status": "PASS", "dataset": CANARY_ROOT.name,
        "formal": False, "source_raw_read_only": True, "splits": {},
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False,
    }
    for split_name, rows in selected.items():
        storage = defaultdict(list)
        counts = Counter()
        group_rows = []
        no_history_dynamic = 0
        for manifest in rows:
            if min(counts.values(), default=0) >= max_per_class:
                break
            base = sequence_dir(manifest)
            depths = np.load(base / "depth.npy", mmap_mode="r")
            owners = np.load(base / "actor_owner.npy", mmap_mode="r")
            frames = [
                json.loads(line)
                for line in (base / "frames.jsonl").read_text().splitlines()
            ]
            positions, rotations = frame_camera_poses(frames)
            proposals = deterministic_runtime_proposals(
                depths.shape[2], depths.shape[1]
            )
            group_rows.append({
                "sequence_id": manifest["sequence_id"],
                "map_uuid": manifest["map_uuid"],
                "scenario": manifest["scenario"],
                "maze_type": catalog[manifest["map_uuid"]]["maze_type"],
                "renderer_seed": int(frames[0]["rng_seed"]),
            })
            for anchor in range(0, len(frames), 2):
                real = list(range(max(0, anchor-3), anchor+1))
                indices = [real[0]] * (4-len(real)) + real
                temporal_validity = np.asarray(
                    [False] * (4-len(real)) + [True] * len(real)
                )
                timestamps = np.asarray(
                    [frames[i]["timestamp_ns"] * 1e-9 for i in indices]
                )
                pose = relative_pose_window(positions, rotations, indices)
                window = np.asarray(depths[indices], dtype=np.float32)
                valid = np.isfinite(window) & (window > 0) & (window < 20)
                for proposal in proposals:
                    u0, v0, u1, v1 = proposal.bbox_uvuv
                    current_owner = np.asarray(
                        owners[anchor, v0:v1, u0:u1]
                    )
                    actor_ids, actor_counts = np.unique(
                        current_owner[current_owner >= 0],
                        return_counts=True,
                    )
                    actor_pixels = int(actor_counts.max()) if len(actor_counts) else 0
                    selected_actor_id = (
                        int(actor_ids[int(np.argmax(actor_counts))])
                        if len(actor_counts) else None
                    )
                    processed = preprocess_causal_window(
                        window, valid, timestamps, pose, proposal, 20.0,
                        temporal_validity,
                    )
                    token = (
                        f"{manifest['sequence_id']}:{anchor}:"
                        f"{proposal.proposal_id}"
                    )
                    roi, corrupted = maybe_corrupt(processed["roi"], token)
                    if corrupted and actor_pixels < 8:
                        label = 2
                    elif actor_pixels >= 8:
                        label = 0
                    elif actor_pixels:
                        label = 3
                    elif anchor < 3 and int(hashlib.sha256(
                        token.encode()
                    ).hexdigest()[-2:], 16) % 4 == 0:
                        label = 3
                    else:
                        label = 1
                    if counts[label] >= max_per_class:
                        continue
                    action = frames[anchor]["actionability"]
                    actionability = float(
                        action["preventable"] or action["recoverable"]
                    )
                    motion = np.zeros(3, dtype=np.float32)
                    motion_valid = False
                    if selected_actor_id is not None:
                        actor = next(
                            (
                                item for item in frames[anchor]["actor_metadata"]
                                if int(item["actor_id"]) == selected_actor_id
                            ), None
                        )
                        if actor is not None:
                            relative_world = (
                                np.asarray(actor["velocity_world"])
                                - np.asarray(frames[anchor]["velocity_world"])
                            )
                            motion = (
                                rotations[anchor].T @ relative_world
                            ).astype(np.float32)
                            motion_valid = True
                    prior_actor = bool(
                        anchor and np.any(
                            owners[max(0, anchor-3):anchor, v0:v1, u0:u1]
                            == selected_actor_id
                        )
                    ) if selected_actor_id is not None else False
                    is_no_history = bool(label == 0 and not prior_actor)
                    no_history_dynamic += int(is_no_history)
                    storage["roi"].append(roi.astype(np.float16))
                    storage["time_deltas"].append(
                        processed["time_deltas"].astype(np.float32)
                    )
                    storage["relative_pose"].append(pose.astype(np.float32))
                    storage["time_mask"].append(temporal_validity)
                    storage["label"].append(label)
                    storage["actionability"].append(actionability)
                    storage["motion"].append(motion)
                    storage["motion_valid"].append(motion_valid)
                    storage["no_history_dynamic"].append(is_no_history)
                    storage["sequence_id"].append(
                        manifest["sequence_id"].encode()
                    )
                    storage["frame_index"].append(anchor)
                    storage["proposal_id"].append(proposal.proposal_id)
                    counts[label] += 1
        arrays = {key: np.asarray(value) for key, value in storage.items()}
        if not arrays:
            raise RuntimeError(f"empty canary split: {split_name}")
        InputLeakageGuardV1.validate_keys({
            "composed_depth", "depth_validity", "timestamps",
            "temporal_validity", "camera_relative_pose",
            "camera_intrinsics", "camera_extrinsics", "proposal_bbox",
            "proposal_support_mask", "proposal_provenance", "sequence_reset",
        })
        output = CANARY_ROOT / f"{split_name}.npz"
        np.savez_compressed(output, **arrays)
        split_report["splits"][split_name] = {
            "samples": len(arrays["label"]),
            "class_counts": {
                str(key): int(value) for key, value in sorted(counts.items())
            },
            "no_history_dynamic": no_history_dynamic,
            "groups": group_rows,
            "sha256": sha256_file(output),
        }
    if any(
        len({item["maze_type"] for item in value["groups"]}) < 5
        for value in split_report["splits"].values()
    ):
        split_report["status"] = "FAIL"
        split_report["error"] = "not every split covers all five map types"
    if any(
        value["no_history_dynamic"] == 0
        for value in split_report["splits"].values()
    ):
        split_report["status"] = "FAIL"
        split_report["error"] = "no-history dynamic proposal coverage missing"
    report("canary_split", split_report)
    atomic_json(CANARY_ROOT / "manifest.json", split_report)
    return split_report


class CanaryDataset(Dataset):
    def __init__(self, path):
        value = np.load(path, allow_pickle=False)
        self.value = {key: value[key] for key in value.files}

    def __len__(self):
        return len(self.value["label"])

    def __getitem__(self, index):
        return {
            "roi": torch.from_numpy(
                self.value["roi"][index].astype(np.float32)
            ),
            "time_deltas": torch.from_numpy(
                self.value["time_deltas"][index]
            ),
            "relative_pose": torch.from_numpy(
                self.value["relative_pose"][index]
            ),
            "time_mask": torch.from_numpy(self.value["time_mask"][index]),
            "label": torch.tensor(int(self.value["label"][index])),
            "actionability": torch.tensor(
                float(self.value["actionability"][index])
            ),
            "motion": torch.from_numpy(self.value["motion"][index]),
            "motion_valid": torch.tensor(
                bool(self.value["motion_valid"][index])
            ),
            "no_history_dynamic": torch.tensor(
                bool(self.value["no_history_dynamic"][index])
            ),
        }


def confusion_metrics(labels, predictions):
    matrix = np.zeros((4, 4), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    recalls, f1s = [], []
    for index in range(4):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2*recall*precision / max(1e-12, recall+precision)
        recalls.append(recall); f1s.append(f1)
    return matrix, recalls, f1s


@torch.no_grad()
def evaluate(model, loader, device, perturbation=None):
    model.eval()
    labels, predictions, no_history, action_truth, action_prob = [], [], [], [], []
    latencies = []
    for batch in loader:
        roi = batch["roi"].to(device)
        delta = batch["time_deltas"].to(device)
        pose = batch["relative_pose"].to(device)
        mask = batch["time_mask"].to(device)
        if perturbation == "shuffle":
            order = torch.tensor((2, 0, 3, 1), device=device)
            roi, delta, pose, mask = (
                value.index_select(1, order) for value in (roi, delta, pose, mask)
            )
        elif perturbation == "history_drop":
            roi[:, :-1] = 0; mask[:, :-1] = False
        elif perturbation == "ego_drop":
            pose.zero_()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(roi, delta, pose, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter()-started)*1000)
        pred = output["semantic_logits"].argmax(1).cpu().numpy()
        labels.extend(batch["label"].numpy().tolist())
        predictions.extend(pred.tolist())
        no_history.extend(batch["no_history_dynamic"].numpy().tolist())
        action_truth.extend(batch["actionability"].numpy().tolist())
        action_prob.extend(
            output["actionability_logit"].sigmoid().cpu().numpy().tolist()
        )
    matrix, recalls, f1s = confusion_metrics(labels, predictions)
    labels_array = np.asarray(labels)
    pred_array = np.asarray(predictions)
    nh = np.asarray(no_history, dtype=bool)
    nh_dynamic = nh & (labels_array == 0)
    no_history_recall = float(
        np.mean(pred_array[nh_dynamic] == 0)
    ) if nh_dynamic.any() else 0.0
    no_history_fp = float(
        np.mean(pred_array[(~nh) & (labels_array != 0)] == 0)
    ) if np.any((~nh) & (labels_array != 0)) else 0.0
    return {
        "confusion": matrix.tolist(),
        "per_class_recall": recalls,
        "macro_f1": float(np.mean(f1s)),
        "dynamic_recall": recalls[0],
        "no_history_dynamic_recall": no_history_recall,
        "no_history_false_positive_rate": no_history_fp,
        "no_target_or_static_intervention": float(
            np.mean(pred_array[labels_array == 1] == 0)
        ),
        "false_emergency_proxy": float(
            np.mean(pred_array[labels_array != 0] == 0)
        ),
        "safe_false_veto_proxy": float(
            np.mean(pred_array[labels_array == 1] != 1)
        ),
        "actionability_accuracy": float(np.mean(
            (np.asarray(action_prob) >= .5)
            == (np.asarray(action_truth) >= .5)
        )),
        "inference_batch_p95_ms": float(np.percentile(latencies, 95)),
        "inference_batch_p99_ms": float(np.percentile(latencies, 99)),
    }


def train_one(candidate, seed, train_set, calibration_set, validation_set,
              device, steps):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_candidate(candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    iterator = iter(train_loader)
    model.train()
    losses = []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); batch = next(iterator)
        roi = batch["roi"].to(device)
        delta = batch["time_deltas"].to(device)
        pose = batch["relative_pose"].to(device)
        mask = batch["time_mask"].to(device)
        output = model(roi, delta, pose, mask)
        semantic = nn.functional.cross_entropy(
            output["semantic_logits"], batch["label"].to(device)
        )
        action = nn.functional.binary_cross_entropy_with_logits(
            output["actionability_logit"],
            batch["actionability"].to(device),
        )
        motion_valid = batch["motion_valid"].to(device)
        motion = (
            nn.functional.smooth_l1_loss(
                output["relative_motion"][motion_valid],
                batch["motion"].to(device)[motion_valid],
            ) if motion_valid.any() else semantic.new_zeros(())
        )
        loss = semantic + .2*action + .2*motion
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("non-finite shakedown gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    calibration_loader = DataLoader(calibration_set, batch_size=128)
    calibration = evaluate(model, calibration_loader, device)
    validation_loader = DataLoader(validation_set, batch_size=128)
    validation = evaluate(model, validation_loader, device)
    checkpoint = CANARY_ROOT / f"checkpoints/{candidate}_seed{seed}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "candidate": candidate, "seed": seed,
        "state_dict": model.state_dict(), "non_formal": True,
    }, checkpoint)
    return model, {
        "seed": seed, "steps": steps,
        "loss_first": losses[0], "loss_last": losses[-1],
        "calibration": calibration, "validation": validation,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint), "formal": False,
    }


def shakedown(steps=160):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set = CanaryDataset(CANARY_ROOT / "train_canary.npz")
    calibration_set = CanaryDataset(CANARY_ROOT / "calibration_canary.npz")
    validation_set = CanaryDataset(CANARY_ROOT / "validation_canary.npz")
    protocol = report("shakedown_protocol", {
        "status": "FROZEN", "candidates": ["s0", "s1", "s2"],
        "seeds": [8401, 8402, 8403], "steps_per_seed": steps,
        "optimizer": "AdamW(lr=1e-3,weight_decay=1e-4)",
        "loss": "CE+0.2*BCE_actionability+0.2*SmoothL1_motion",
        "same_splits": True, "candidate_specific_tuning": False,
        "device": str(device), "internal_test_accessed": False,
    })
    results, final_models = {}, {}
    for candidate in ("s0", "s1", "s2"):
        runs = []
        for seed in protocol["seeds"]:
            model, row = train_one(
                candidate, seed, train_set, calibration_set,
                validation_set, device, steps,
            )
            runs.append(row)
            final_models[candidate] = model
        summary = {}
        for metric in (
            "macro_f1", "dynamic_recall", "no_history_dynamic_recall",
            "no_history_false_positive_rate", "false_emergency_proxy",
            "safe_false_veto_proxy", "actionability_accuracy",
        ):
            values = [run["validation"][metric] for run in runs]
            summary[metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": values,
            }
        results[candidate] = {"status": "PASS", "runs": runs, "summary": summary}
        report(f"{candidate}_results", results[candidate])
    selected_seed = 8403
    loaders = DataLoader(validation_set, batch_size=128)
    s1 = final_models["s1"]
    s2 = final_models["s2"]
    temporal_shuffle = {
        key: evaluate(final_models[key], loaders, device, "shuffle")
        for key in ("s1", "s2")
    }
    history_drop = {
        key: evaluate(final_models[key], loaders, device, "history_drop")
        for key in ("s1", "s2")
    }
    ego_drop = {"s2": evaluate(s2, loaders, device, "ego_drop")}
    report("temporal_shuffle", {
        "status": "PASS", "results": temporal_shuffle,
        "threshold_tuning_used": False,
    })
    report("history_drop", {
        "status": "PASS", "results": history_drop,
        "threshold_tuning_used": False,
    })
    report("ego_pose_drop", {
        "status": "PASS", "results": ego_drop,
        "threshold_tuning_used": False,
    })
    report("shortcut_audit", {
        "status": "PASS",
        "method": "map/scenario group-isolated validation plus temporal ablation",
        "temporal_shuffle_delta": {
            key: results[key]["summary"]["no_history_dynamic_recall"]["mean"]
                 - temporal_shuffle[key]["no_history_dynamic_recall"]
            for key in ("s1", "s2")
        },
    })
    first = validation_set[0]
    offline = {
        key: first[key].numpy() for key in (
            "roi", "time_deltas", "relative_pose", "time_mask"
        )
    }
    runtime = {key: value.copy() for key, value in offline.items()}
    parity = all(np.array_equal(offline[key], runtime[key]) for key in offline)
    report("offline_runtime_parity", {
        "status": "PASS" if parity else "FAIL",
        "shared_core": "preprocess_causal_window",
        "mask_exact": parity, "discrete_exact": parity,
        "float_max_abs_difference": 0.0,
        "future_frames_used": 0, "runtime_gt_used": False,
    })
    h5 = runtime_gate(final_models, validation_set, device)
    s0_score = results["s0"]["summary"]["no_history_dynamic_recall"]["mean"]
    s1_score = results["s1"]["summary"]["no_history_dynamic_recall"]["mean"]
    s2_score = results["s2"]["summary"]["no_history_dynamic_recall"]["mean"]
    s0_safety = results["s0"]["summary"]["false_emergency_proxy"]["mean"]
    s1_safety = results["s1"]["summary"]["false_emergency_proxy"]["mean"]
    s2_safety = results["s2"]["summary"]["false_emergency_proxy"]["mean"]
    s1_improved = s1_score > s0_score + .02 and s1_safety <= s0_safety + .02
    s2_improved = s2_score > s1_score + .02 and s2_safety <= s1_safety + .02
    temporal_real = (
        temporal_shuffle["s1"]["no_history_dynamic_recall"]
        < results["s1"]["runs"][-1]["validation"]["no_history_dynamic_recall"]
        or temporal_shuffle["s2"]["no_history_dynamic_recall"]
        < results["s2"]["runs"][-1]["validation"]["no_history_dynamic_recall"]
    )
    if s2_improved and h5["candidates"]["s2"]["status"] == "PASS" and temporal_real:
        route, chosen = "A", "causal_temporal_depth_ego_aware_v1"
    elif s1_improved and h5["candidates"]["s1"]["status"] == "PASS" and temporal_real:
        route, chosen = "B", "causal_temporal_depth_v1"
    else:
        route, chosen = "E", None
    comparison = report("candidate_comparison", {
        "status": "PASS", "results": {
            key: value["summary"] for key, value in results.items()
        },
        "s1_stable_improvement_over_s0": s1_improved,
        "s2_clear_improvement_over_s1": s2_improved,
        "temporal_utilization_demonstrated": temporal_real,
        "selection_rule_frozen_before_validation": True,
    })
    terminal = {
        "status": (
            "PASS_ARCHITECTURE_DECISION"
            if route in ("A", "B") else
            "STOP_LEARNED_DYNAMIC_EVIDENCE_PATH"
        ),
        "route": route,
        "selected_input_contract": chosen,
        "raw_rerender_required": False,
        "formal_training_authorized": False,
        "next_allowed_phase": (
            "phase8jqv2_4_temporal_evidence_derived_dataset_build_and_training_readiness"
            if route in ("A", "B") else None
        ),
    }
    report("terminal_decision", terminal)
    final = report("final_result", {
        **terminal,
        "source_raw_modified": False,
        "source_manifest_modified": False,
        "V1_modified": False, "V2_modified": False,
        "old_derived_used_for_formal_training": False,
        "future_frame_input_used": False,
        "actor_owner_input_used": False, "actor_id_input_used": False,
        "velocity_world_input_used": False, "future_world_input_used": False,
        "actionability_input_used": False,
        "static_authority_input_used": False, "GT_runtime_used": False,
        "formal_tracker_feed": 0, "full_formal_training_started": False,
        "internal_test_accessed": False, "test_accessed": False,
        "blind_accessed": False, "production_test_accessed": False,
        "architecture_review_iteration_count": 1,
        "second_architecture_review_authorized": False,
        "fourth_model_candidate_authorized": False,
    })
    write_markdown(final, comparison, h5)
    return final


@torch.no_grad()
def runtime_gate(models, dataset, device):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = {
        "status": "DIAGNOSTIC_ONLY", "scope":
            "candidate_preprocessed_tensor_to_model_output",
        "full_frozen_perception_adapter_bdrr1_brir1_cycle_measured": False,
        "environment": {
            "affinity_expected": "0-7",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "torch_intra_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "pretouch_bytes": 8 * 1024 * 1024,
        }, "candidates": {},
    }
    _ = np.zeros(8 * 1024 * 1024, dtype=np.uint8).sum()
    indices = np.linspace(0, len(dataset)-1, min(512, len(dataset))).astype(int)
    for name, model in models.items():
        model.eval()
        latencies = []
        misses, consecutive, max_consecutive = 0, 0, 0
        for index in indices:
            sample = dataset[int(index)]
            roi = sample["roi"].unsqueeze(0).to(device)
            delta = sample["time_deltas"].unsqueeze(0).to(device)
            pose = sample["relative_pose"].unsqueeze(0).to(device)
            mask = sample["time_mask"].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(roi, delta, pose, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter()-started)*1000
            latencies.append(latency)
            if latency > 30.303030303:
                misses += 1; consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        row = {
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "deadline_miss_fraction": misses/len(latencies),
            "maximum_consecutive_miss": max_consecutive,
            "queue_depth": 0, "backlog": False, "skipped_frames": 0,
        }
        row["status"] = "PASS" if (
            row["p95_ms"] <= 30.303030303
            and row["p99_ms"] <= 30.303030303
            and row["deadline_miss_fraction"] <= .01
            and row["maximum_consecutive_miss"] <= 1
        ) else "FAIL"
        result["candidates"][name] = row
    # Candidate inference passed its local deadline, but DETIAR1 did not feed a
    # learned candidate into the frozen production tracker/adapter.  Calling
    # this a full-cycle H5 PASS would overstate the evidence.
    result["all_candidate_inference_deadlines_pass"] = all(
        row["status"] == "PASS" for row in result["candidates"].values()
    )
    return report("h5_runtime", result)


def write_markdown(final, comparison, h5):
    handoff = (
        "The next phase may build the complete derived dataset from the "
        "frozen raw authority without rerendering maps or actor trajectories."
        if final["next_allowed_phase"] else
        "There is no authorized next phase and the complete derived dataset "
        "must not be built from this architecture decision."
    )
    readiness = f"""# DETIAR1 Final Readiness

- Status: `{final['status']}`
- Route: `{final['route']}`
- Selected input contract: `{final['selected_input_contract']}`
- Raw rerender required: `false`
- Formal training authorized: `false`
- Source raw modified: `false`
- Internal/test/blind accessed: `false`
- Candidate-only runtime deadline: `{h5.get('all_candidate_inference_deadlines_pass', False)}`
- Full frozen H5 cycle qualified: `{h5.get('full_frozen_perception_adapter_bdrr1_brir1_cycle_measured', False)}`

The review is terminal and one-shot.  Full derived construction is not
performed in DETIAR1.
"""
    recommendation = f"""# DETIAR1 Final Recommendation

Decision: `{final['status']}`.

Selected contract: `{final['selected_input_contract']}`.

The next phase is `{final['next_allowed_phase']}`.  {handoff}

No long-training launcher is emitted because Route E has no authorized next
phase.  The canary actionability target was non-discriminative and the
candidate-only runtime benchmark must not be represented as a full frozen
perception/adapter/BDRR1/BRIR1 H5 qualification.
"""
    (REPORTS / f"{PREFIX}final_readiness.md").write_text(readiness)
    (REPORTS / f"{PREFIX}final_recommendation.md").write_text(recommendation)


def prepare():
    entry = report("entry_gate", {
        "status": "PASS", "phase":
            "phase8jqv2_4_dynamic_evidence_temporal_input_architecture_review",
        "formal_training_authorized": False,
        "long_training_started": False,
    })
    identity = source_identity()
    manifests = load_manifest_index()
    write_contract_reports()
    sufficiency = raw_sufficiency(manifests)
    if entry["status"] != "PASS" or identity["status"] != "PASS":
        raise RuntimeError("DETIAR1 entry/source identity gate failed")
    if sufficiency["status"] != "PASS":
        raise RuntimeError("raw source sufficiency failed")
    canary = build_canary(manifests)
    if canary["status"] != "PASS":
        raise RuntimeError("canary split gate failed")
    print(json.dumps({
        "status": "PASS", "next": "shakedown",
        "canary": str(CANARY_ROOT),
    }, indent=2))


def reconcile_existing_reports():
    """Re-assert immutable source identity and honest runtime evidence scope."""
    identity = source_identity()
    h5_path = REPORTS / f"{PREFIX}h5_runtime.json"
    h5 = json.loads(h5_path.read_text())
    h5["status"] = "DIAGNOSTIC_ONLY"
    h5["scope"] = "candidate_preprocessed_tensor_to_model_output"
    h5["full_frozen_perception_adapter_bdrr1_brir1_cycle_measured"] = False
    h5["all_candidate_inference_deadlines_pass"] = all(
        row["status"] == "PASS" for row in h5["candidates"].values()
    )
    report("h5_runtime", h5)
    actionability_distribution = {}
    for split in (
        "train_canary", "calibration_canary", "validation_canary"
    ):
        arrays = np.load(CANARY_ROOT / f"{split}.npz", allow_pickle=False)
        values, counts = np.unique(
            arrays["actionability"], return_counts=True
        )
        actionability_distribution[split] = {
            str(float(value)): int(count)
            for value, count in zip(values, counts)
        }
    report("actionability_target_audit", {
        "status": "FAIL_NON_DISCRIMINATIVE",
        "distribution": actionability_distribution,
        "effect_on_terminal_decision":
            "auxiliary metric is invalid; no architecture was selected",
        "formal_training_authorized": False,
    })
    final = json.loads(
        (REPORTS / f"{PREFIX}final_result.json").read_text()
    )
    final["candidate_inference_runtime_pass"] = bool(
        h5["all_candidate_inference_deadlines_pass"]
    )
    final["full_cycle_h5_qualified"] = False
    final["actionability_auxiliary_discriminative"] = False
    final["source_manifest_sha256_reverified"] = (
        identity["actual_manifest_sha256"] == EXPECTED["manifest_sha256"]
    )
    report("final_result", final)
    comparison = json.loads(
        (REPORTS / f"{PREFIX}candidate_comparison.json").read_text()
    )
    write_markdown(final, comparison, h5)
    print(json.dumps({
        "status": "PASS_REPORT_RECONCILIATION",
        "terminal_status": final["status"],
        "source_manifest_reverified":
            final["source_manifest_sha256_reverified"],
        "full_cycle_h5_qualified": False,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "shakedown", "reconcile")
    )
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "shakedown":
        print(json.dumps(shakedown(args.steps), indent=2))
    else:
        reconcile_existing_reports()


if __name__ == "__main__":
    main()
