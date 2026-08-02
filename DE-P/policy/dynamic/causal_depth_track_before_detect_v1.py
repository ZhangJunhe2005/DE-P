"""Bounded depth-only causal track-before-detect candidate."""

from __future__ import annotations

from .visibility_aware_causal_residual_v1 import (
    VisibilityAwareCausalResidualV1,
)


class CausalDepthTrackBeforeDetectV1(VisibilityAwareCausalResidualV1):
    algorithm_name = "causal_depth_track_before_detect_v1"

    def __init__(self, config, parameters):
        merged = {
            "history_frames": 4,
            "minimum_temporal_support_frames": 3,
            "maximum_tracklets": 32,
            "minimum_world_speed_mps": 0.15,
            "maximum_world_speed_mps": 3.0,
            "minimum_direction_consistency": 0.70,
            **dict(parameters),
        }
        super().__init__(config, merged)

    def _accept(self, row):
        state = row["tracklet"]
        speed = float((state.velocity @ state.velocity) ** 0.5)
        return bool(
            state.support >= int(
                self.parameters["minimum_temporal_support_frames"]
            )
            and row["fov_fraction"]
                <= float(self.parameters.get("maximum_fov_fraction", 0.5))
            and row["invalid_fraction"]
                <= float(self.parameters.get("maximum_invalid_fraction", 0.5))
            and float(self.parameters["minimum_world_speed_mps"])
                <= speed <= float(self.parameters["maximum_world_speed_mps"])
            and state.direction_consistency
                >= float(self.parameters["minimum_direction_consistency"])
        )


__all__ = ["CausalDepthTrackBeforeDetectV1"]
