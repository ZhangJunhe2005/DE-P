"""Strictly separated discovery and existing-track reacquisition paths."""

from __future__ import annotations

import numpy as np

from .causal_depth_track_before_detect_v1 import (
    CausalDepthTrackBeforeDetectV1,
)
from .range_image_foreground_v2_1 import RestrictedTrackReacquisition
from .visibility_aware_causal_residual_v1 import (
    VisibilityAwareCausalResidualV1,
)


class DualPathDynamicPerceptionV1:
    algorithm_name = "dual_path_dynamic_perception_v1"

    def __init__(self, config, parameters):
        self.parameters = dict(parameters)
        self.discovery_residual = VisibilityAwareCausalResidualV1(
            config, self.parameters["residual"]
        )
        self.discovery_tracklet = CausalDepthTrackBeforeDetectV1(
            config, self.parameters["tracklet"]
        )
        self.reacquisition = RestrictedTrackReacquisition(
            **self.parameters.get("reacquisition", {})
        )
        self.last_diagnostics = {}
        self.last_range_seed = self.last_free_seed = self.last_component_mask = None

    def reset(self):
        self.discovery_residual.reset()
        self.discovery_tracklet.reset()
        self.last_diagnostics = {}

    def extract(self, frame, free_space_seed=None):
        residual, _ = self.discovery_residual.extract(frame, free_space_seed)
        tracklet, _ = self.discovery_tracklet.extract(frame, free_space_seed)
        merged = []
        for observation in (*residual, *tracklet):
            if any(
                np.linalg.norm(
                    observation.centroid_world-other.centroid_world
                ) <= float(self.parameters.get(
                    "duplicate_distance_m", 0.45
                ))
                for other in merged
            ):
                continue
            merged.append(observation)
        labels = np.full(len(frame.points_camera), -1, dtype=np.int64)
        lookup = np.full(frame.depth_m.shape, -1, dtype=np.int64)
        lookup[
            frame.pixels_uv[:, 1], frame.pixels_uv[:, 0]
        ] = np.arange(len(frame.pixels_uv))
        component_mask = np.zeros(frame.depth_m.shape, dtype=bool)
        for index, observation in enumerate(merged):
            flat = np.asarray(observation.pixel_indices, dtype=np.int64)
            v, u = np.divmod(flat, frame.camera_model.width)
            point_indices = lookup[v, u]
            labels[point_indices[point_indices >= 0]] = index
            component_mask[v, u] = True
        self.last_component_mask = component_mask
        self.last_range_seed = self.discovery_residual.last_range_seed
        self.last_free_seed = np.zeros_like(component_mask)
        self.last_diagnostics = {
            "mode": self.algorithm_name,
            "history_size": max(
                self.discovery_residual.history_size,
                self.discovery_tracklet.history_size,
            ),
            "history_capacity": 4,
            "future_frames_used": 0,
            "runtime_gt_used": False,
            "range_seed_count": int(
                0 if self.last_range_seed is None
                else self.last_range_seed.sum()
            ),
            "component_count": len(merged),
            "component_pixel_count": int(component_mask.sum()),
            "weak_support_pixels": int(component_mask.sum()),
            "discovery_residual_count": len(residual),
            "discovery_tracklet_count": len(tracklet),
            "duplicates_suppressed":
                len(residual)+len(tracklet)-len(merged),
            "reacquisition_birth_allowed": False,
        }
        return tuple(merged), labels

    def evaluate_reacquisition(
        self, residual_mask, predicted_roi, existing_track_state,
    ):
        alive = bool(
            existing_track_state
            and existing_track_state.get("confirmed")
            and existing_track_state.get("alive")
        )
        value = self.reacquisition.evaluate(
            residual_mask, predicted_roi, alive
        )
        return {
            **value,
            "predicted_track_id": (
                existing_track_state.get("track_id") if alive else None
            ),
            "reacquisition_only": True,
            "birth_allowed": False,
            "association_bypassed": False,
            "prediction_used_as_measurement": False,
        }

    @property
    def history_size(self):
        return max(
            self.discovery_residual.history_size,
            self.discovery_tracklet.history_size,
        )


__all__ = ["DualPathDynamicPerceptionV1"]
