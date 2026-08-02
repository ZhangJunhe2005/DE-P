"""Versioned runtime-safe dynamic measurement interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DynamicMeasurementV3:
    timestamp: float
    position_camera: np.ndarray
    position_world: np.ndarray
    covariance: np.ndarray
    evidence_source: str
    temporal_support_frames: int
    pixel_support: int
    point_support: int
    stable_overlap_fraction: float
    fov_boundary_fraction: float
    disocclusion_fraction: float
    depth_validity_fraction: float
    predicted_track_id: Optional[int]
    birth_allowed: bool
    reacquisition_only: bool
    validity: bool
    rejection_reason: str = ""
    pixel_indices: Tuple[int, ...] = ()

    def __post_init__(self):
        camera = np.asarray(self.position_camera, dtype=np.float64)
        world = np.asarray(self.position_world, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if camera.shape != (3,) or world.shape != (3,):
            raise ValueError("measurement positions must have shape [3]")
        if covariance.shape != (3, 3):
            raise ValueError("measurement covariance must be [3,3]")
        if not (
            np.isfinite(camera).all() and np.isfinite(world).all()
            and np.isfinite(covariance).all()
        ):
            raise ValueError("measurement geometry must be finite")
        if self.reacquisition_only and self.birth_allowed:
            raise ValueError("reacquisition measurement cannot birth")
        for value in (
            self.stable_overlap_fraction, self.fov_boundary_fraction,
            self.disocclusion_fraction, self.depth_validity_fraction,
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("provenance fractions must be in [0,1]")
        object.__setattr__(self, "position_camera", camera.copy())
        object.__setattr__(self, "position_world", world.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(
            self, "pixel_indices", tuple(int(x) for x in self.pixel_indices)
        )


FORBIDDEN_MEASUREMENT_FIELDS = frozenset({
    "gt_actor_id", "gt_mask", "expected_control_class",
    "authority_correspondence", "future_information", "annex_metadata",
})


__all__ = ["DynamicMeasurementV3", "FORBIDDEN_MEASUREMENT_FIELDS"]
