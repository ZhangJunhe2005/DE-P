"""Causal temporal depth input contract for the DETIAR1 comparison.

This module is intentionally independent from authoritative labels.  The
preprocessor accepts only runtime-available depth, validity, timestamps, pose,
calibration, and proposal data.  Offline labels are handled by the canary
builder after preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn


CONTRACT_VERSION = "causal_temporal_depth_input_v1"
ROI_SIZE = 32
TEMPORAL_LENGTH = 4


class DatasetFieldRoleV1(str, Enum):
    RUNTIME_INPUT = "RUNTIME_INPUT"
    DERIVED_CAUSAL_INPUT = "DERIVED_CAUSAL_INPUT"
    LABEL_ONLY = "LABEL_ONLY"
    AUDIT_ONLY = "AUDIT_ONLY"
    FORBIDDEN = "FORBIDDEN"


RUNTIME_INPUT_ALLOWLIST = frozenset({
    "composed_depth", "depth_validity", "timestamps",
    "temporal_validity",
    "camera_relative_pose", "camera_intrinsics", "camera_extrinsics",
    "proposal_bbox", "proposal_support_mask", "proposal_provenance",
    "sequence_reset",
})

LABEL_ONLY_FIELDS = frozenset({
    "actor_owner", "actor_id", "position_world", "velocity_world",
    "actual_speed_mps", "future_world", "future_occupancy",
    "semantic_label", "dynamic_actionability", "collision_certificate",
    "true_ttc",
})

FORBIDDEN_INPUT_FIELDS = frozenset({
    *LABEL_ONLY_FIELDS, "authoritative_static_occupancy", "static_depth",
    "offline_visibility", "scenario_label", "map_type",
    "owner_derived_bbox", "owner_derived_association",
})


class InputLeakageGuardV1:
    """Fail closed when a batch contains anything outside the input allowlist."""

    @staticmethod
    def validate_keys(keys: Sequence[str]) -> None:
        keys = frozenset(str(key) for key in keys)
        forbidden = keys & FORBIDDEN_INPUT_FIELDS
        unknown = keys - RUNTIME_INPUT_ALLOWLIST
        if forbidden:
            raise RuntimeError(
                "FAIL_INTEGRITY forbidden model inputs: "
                + ", ".join(sorted(forbidden))
            )
        if unknown:
            raise RuntimeError(
                "FAIL_INTEGRITY non-allowlisted model inputs: "
                + ", ".join(sorted(unknown))
            )

    @classmethod
    def validate_batch(cls, batch: Mapping[str, object]) -> None:
        cls.validate_keys(batch.keys())


@dataclass(frozen=True)
class RuntimeProposalV1:
    """A GT-independent image-space proposal propagated causally by grid ID."""

    proposal_id: int
    bbox_uvuv: tuple[int, int, int, int]
    provenance: str = "deterministic_runtime_grid_v1"


def deterministic_runtime_proposals(
    width: int, height: int, crop_width: int = 64, crop_height: int = 48,
) -> tuple[RuntimeProposalV1, ...]:
    """Return a frozen overlapping grid; it never reads labels or actor state."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    crop_width = min(int(crop_width), width)
    crop_height = min(int(crop_height), height)
    centers_u = np.linspace(crop_width / 2, width - crop_width / 2, 5)
    centers_v = np.linspace(crop_height / 2, height - crop_height / 2, 3)
    proposals = []
    for index, (v, u) in enumerate(
        (pair for v in centers_v for pair in ((v, u) for u in centers_u))
    ):
        u0 = int(round(u - crop_width / 2))
        v0 = int(round(v - crop_height / 2))
        u0 = min(max(u0, 0), width - crop_width)
        v0 = min(max(v0, 0), height - crop_height)
        proposals.append(RuntimeProposalV1(
            index, (u0, v0, u0 + crop_width, v0 + crop_height)
        ))
    return tuple(proposals)


def causal_support_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Current-observation support without static authority or actor ownership."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    if depth.shape != valid.shape:
        raise ValueError("depth and validity shapes differ")
    safe = np.where(valid, depth, 0.0)
    gx = cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3)
    relative = np.hypot(gx, gy) / np.maximum(safe, 0.25)
    return valid & (relative > 0.06)


def _crop_resize(array: np.ndarray, bbox, interpolation) -> np.ndarray:
    u0, v0, u1, v1 = bbox
    crop = np.asarray(array)[v0:v1, u0:u1]
    if crop.size == 0:
        raise ValueError("empty proposal crop")
    return cv2.resize(
        crop, (ROI_SIZE, ROI_SIZE), interpolation=interpolation
    )


def preprocess_causal_window(
    depths: np.ndarray,
    validity: np.ndarray,
    timestamps_s: np.ndarray,
    relative_poses: np.ndarray,
    proposal: RuntimeProposalV1,
    max_depth_m: float,
    temporal_validity: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create Kx3x32x32 tensors from runtime-observable inputs only."""
    InputLeakageGuardV1.validate_keys({
        "composed_depth", "depth_validity", "timestamps",
        "temporal_validity",
        "camera_relative_pose", "camera_intrinsics", "camera_extrinsics",
        "proposal_bbox", "proposal_support_mask", "proposal_provenance",
        "sequence_reset",
    })
    depths = np.asarray(depths, dtype=np.float32)
    validity = np.asarray(validity, dtype=np.bool_)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    relative_poses = np.asarray(relative_poses, dtype=np.float32)
    if depths.ndim != 3 or validity.shape != depths.shape:
        raise ValueError("depth window must be KxHxW with matching validity")
    if len(depths) != TEMPORAL_LENGTH:
        raise ValueError(f"expected K={TEMPORAL_LENGTH}")
    if timestamps_s.shape != (TEMPORAL_LENGTH,):
        raise ValueError("timestamps must have shape K")
    if relative_poses.shape != (TEMPORAL_LENGTH, 6):
        raise ValueError("relative poses must have shape Kx6")
    temporal_validity = (
        np.ones(TEMPORAL_LENGTH, dtype=np.bool_)
        if temporal_validity is None else
        np.asarray(temporal_validity, dtype=np.bool_)
    )
    if temporal_validity.shape != (TEMPORAL_LENGTH,):
        raise ValueError("temporal validity must have shape K")
    channels = np.zeros(
        (TEMPORAL_LENGTH, 3, ROI_SIZE, ROI_SIZE), dtype=np.float32
    )
    for index, (depth, valid) in enumerate(zip(depths, validity)):
        support = causal_support_mask(depth, valid)
        normalized = np.clip(depth / float(max_depth_m), 0.0, 1.0)
        channels[index, 0] = _crop_resize(
            normalized, proposal.bbox_uvuv, cv2.INTER_LINEAR
        )
        channels[index, 1] = _crop_resize(
            valid.astype(np.uint8), proposal.bbox_uvuv, cv2.INTER_NEAREST
        )
        channels[index, 2] = _crop_resize(
            support.astype(np.uint8), proposal.bbox_uvuv, cv2.INTER_NEAREST
        )
        if not temporal_validity[index]:
            channels[index] = 0.0
    deltas = (timestamps_s - timestamps_s[-1]).astype(np.float32)
    return {
        "roi": channels,
        "time_deltas": deltas,
        "relative_pose": relative_poses,
        "time_mask": temporal_validity,
        "proposal_bbox": np.asarray(proposal.bbox_uvuv, dtype=np.int16),
        "proposal_id": np.asarray(proposal.proposal_id, dtype=np.int16),
    }


class SharedSpatialEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(48, output_dim)

    def forward(self, value):
        return self.projection(self.network(value).flatten(1))


class _EvidenceHeads(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.semantic = nn.Linear(hidden_dim, 4)
        self.actionability = nn.Linear(hidden_dim, 1)
        self.relative_motion = nn.Linear(hidden_dim, 3)

    def forward(self, value):
        return {
            "semantic_logits": self.semantic(value),
            "actionability_logit": self.actionability(value).squeeze(-1),
            "relative_motion": self.relative_motion(value),
        }


class S0SingleFrameDepth(nn.Module):
    contract_name = "S0_SINGLE_FRAME_DEPTH"

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.spatial = SharedSpatialEncoder(hidden_dim)
        self.heads = _EvidenceHeads(hidden_dim)

    def forward(self, roi, time_deltas=None, relative_pose=None, time_mask=None):
        return self.heads(self.spatial(roi[:, -1]))


class S1CausalTemporalDepth(nn.Module):
    contract_name = "S1_CAUSAL_TEMPORAL_DEPTH"

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.spatial = SharedSpatialEncoder(hidden_dim)
        self.temporal = nn.GRU(hidden_dim + 1, hidden_dim, batch_first=True)
        self.heads = _EvidenceHeads(hidden_dim)

    def _sequence(self, roi, time_deltas, relative_pose):
        batch, steps = roi.shape[:2]
        spatial = self.spatial(roi.flatten(0, 1)).reshape(batch, steps, -1)
        return torch.cat((spatial, time_deltas.unsqueeze(-1)), dim=-1)

    def forward(self, roi, time_deltas, relative_pose=None, time_mask=None):
        sequence = self._sequence(roi, time_deltas, relative_pose)
        if time_mask is not None:
            sequence = sequence * time_mask.unsqueeze(-1).to(sequence.dtype)
        output, _ = self.temporal(sequence)
        return self.heads(output[:, -1])


class S2CausalTemporalDepthEgoAware(S1CausalTemporalDepth):
    contract_name = "S2_CAUSAL_TEMPORAL_DEPTH_EGO_AWARE"

    def __init__(self, hidden_dim=64, pose_dim=8):
        nn.Module.__init__(self)
        self.spatial = SharedSpatialEncoder(hidden_dim)
        self.pose = nn.Sequential(
            nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, pose_dim)
        )
        self.temporal = nn.GRU(
            hidden_dim + 1 + pose_dim, hidden_dim, batch_first=True
        )
        self.heads = _EvidenceHeads(hidden_dim)

    def _sequence(self, roi, time_deltas, relative_pose):
        batch, steps = roi.shape[:2]
        spatial = self.spatial(roi.flatten(0, 1)).reshape(batch, steps, -1)
        pose = self.pose(relative_pose)
        return torch.cat((spatial, time_deltas.unsqueeze(-1), pose), dim=-1)


def build_candidate(name: str) -> nn.Module:
    candidates = {
        "s0": S0SingleFrameDepth,
        "s1": S1CausalTemporalDepth,
        "s2": S2CausalTemporalDepthEgoAware,
    }
    try:
        return candidates[str(name).lower()]()
    except KeyError as error:
        raise ValueError(f"unknown DETIAR1 candidate: {name}") from error


__all__ = [
    "CONTRACT_VERSION", "DatasetFieldRoleV1", "InputLeakageGuardV1",
    "RuntimeProposalV1", "deterministic_runtime_proposals",
    "preprocess_causal_window", "S0SingleFrameDepth",
    "S1CausalTemporalDepth", "S2CausalTemporalDepthEgoAware",
    "build_candidate",
]
