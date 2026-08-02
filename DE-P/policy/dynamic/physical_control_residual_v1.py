"""Frozen TF1 candidate-C foreground followed by runtime provenance filtering."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
from scipy.ndimage import binary_dilation

from .range_image_foreground_v2_1 import CandidateRangeImageForeground
from .types import DepthFrame, DynamicPerceptionConfig
from .visibility_aware_causal_residual_v1 import (
    VisibilityAwareCausalResidualV1,
)
from .visibility_provenance_v1 import compute_visibility_provenance


class PhysicalControlResidualV1(VisibilityAwareCausalResidualV1):
    algorithm_name = "physical_control_residual_v1"

    def __init__(self, config: DynamicPerceptionConfig, parameters):
        merged = {
            "history_frames": config.range_history_frames,
            "minimum_temporal_support_frames": 2,
            "maximum_tracklets": 32,
            "minimum_world_speed_mps": 0.15,
            "maximum_world_speed_mps": 3.0,
            "minimum_direction_consistency": 0.7,
            **dict(parameters),
        }
        super().__init__(config, merged)
        self._tf1 = CandidateRangeImageForeground(
            config, "candidate_c",
            {"minimum_channel_support": 1, "require_closer_or_free": True},
        )

    def reset(self):
        super().reset()
        if hasattr(self, "_tf1"):
            self._tf1.reset()

    def extract(self, frame: DepthFrame, free_space_seed=None):
        started = time.perf_counter()
        observations, _ = self._tf1.extract(frame, free_space_seed)
        if not self._history:
            rows = []
        else:
            previous = self._history[-1]
            provenance = compute_visibility_provenance(
                previous.depth_m, frame.depth_m,
                previous.camera_pose_world, frame.camera_pose_world,
                (
                    frame.camera_model.fx, frame.camera_model.fy,
                    frame.camera_model.cx, frame.camera_model.cy,
                ),
                frame.camera_model.max_depth,
                float(self.parameters.get(
                    "depth_consistency_tolerance_m", 0.2
                )),
                frame.camera_model.min_depth,
            )
            rows = []
            boundary_hazard = binary_dilation(
                provenance.newly_visible_candidate
                | provenance.newly_invalid_candidate
                | provenance.previous_depth_invalid
                | provenance.current_depth_invalid
                | provenance.max_depth_transition,
                iterations=int(self.parameters[
                    "boundary_hazard_dilation_pixels"
                ]),
            )
            for observation in observations:
                flat = np.asarray(observation.pixel_indices, dtype=np.int64)
                v, u = np.divmod(flat, frame.camera_model.width)
                if not len(flat):
                    continue
                rows.append({
                    "observation": observation,
                    "pixels_vu": np.stack((v, u), axis=1),
                    "geometric_fraction": float(np.mean(
                        provenance.geometric_overlap[v, u]
                    )),
                    "fov_fraction": float(np.mean(
                        provenance.newly_visible_candidate[v, u]
                    )),
                    "disocclusion_fraction": float(np.mean(
                        provenance.disocclusion_candidate[v, u]
                    )),
                    "invalid_fraction": float(np.mean(
                        provenance.previous_depth_invalid[v, u]
                        | provenance.current_depth_invalid[v, u]
                        | provenance.max_depth_transition[v, u]
                    )),
                    "closer_fraction": float(np.mean(
                        provenance.occlusion_candidate[v, u]
                    )),
                    "pixel_count": len(flat),
                    "boundary_hazard_fraction": float(np.mean(
                        boundary_hazard[v, u]
                    )),
                })
            self._update_tracklets(rows, frame.timestamp)
        accepted = [row for row in rows if self._accept(row)]
        result = tuple(row["observation"] for row in accepted)
        labels = np.full(len(frame.points_camera), -1, dtype=np.int64)
        lookup = np.full(frame.depth_m.shape, -1, dtype=np.int64)
        lookup[
            frame.pixels_uv[:, 1], frame.pixels_uv[:, 0]
        ] = np.arange(len(frame.pixels_uv))
        component_mask = np.zeros(frame.depth_m.shape, dtype=bool)
        for index, row in enumerate(accepted):
            pixels = row["pixels_vu"]
            point_indices = lookup[pixels[:, 0], pixels[:, 1]]
            labels[point_indices[point_indices >= 0]] = index
            component_mask[pixels[:, 0], pixels[:, 1]] = True
        self._history.append(frame)
        self._frame_index += 1
        self.last_range_seed = (
            None if self._tf1.last_range_seed is None
            else self._tf1.last_range_seed.copy()
        )
        self.last_free_seed = (
            None if self._tf1.last_free_seed is None
            else self._tf1.last_free_seed.copy()
        )
        self.last_component_mask = component_mask
        self.last_diagnostics = {
            "mode": self.algorithm_name,
            "history_size": len(self._history),
            "history_capacity": self.history_frames,
            "future_frames_used": 0,
            "runtime_gt_used": False,
            "base_tf1_candidate": "candidate_c_grid_0",
            "range_seed_count": int(
                0 if self.last_range_seed is None
                else self.last_range_seed.sum()
            ),
            "component_count": len(result),
            "component_pixel_count": int(component_mask.sum()),
            "candidate_component_count": len(rows),
            "tracklet_count": len(self._tracklets),
            "weak_support_pixels": int(
                self._tf1.last_diagnostics.get("weak_support_pixels", 0)
            ),
            "components": [{
                "pixel_count": len(row["pixels_vu"]),
                "stable_overlap_fraction": row["geometric_fraction"],
                "fov_boundary_fraction": row["fov_fraction"],
                "disocclusion_fraction": row["disocclusion_fraction"],
                "depth_validity_fraction": row["invalid_fraction"],
                "closer_fraction": row["closer_fraction"],
                "temporal_support_frames": row["tracklet"].support,
                "world_speed_mps":
                    float(np.linalg.norm(row["tracklet"].velocity)),
                "direction_consistency":
                    row["tracklet"].direction_consistency,
                "boundary_hazard_fraction":
                    row["boundary_hazard_fraction"],
                "accepted": self._accept(row),
            } for row in rows],
            "foreground_ms": (time.perf_counter()-started)*1000,
        }
        return result, labels

    def _accept(self, row):
        state = row["tracklet"]
        support = state.support
        speed = float(np.linalg.norm(state.velocity))
        if row["fov_fraction"] > float(
            self.parameters.get("maximum_fov_fraction", 0.5)
        ):
            return False
        if row["invalid_fraction"] > float(
            self.parameters.get("maximum_invalid_fraction", 0.5)
        ):
            return False
        residual_birth = bool(
            support >= 2
            and row["pixel_count"] >= int(self.parameters[
                "minimum_component_pixels_for_residual_birth"
            ])
            and row["closer_fraction"] >= float(
                self.parameters["minimum_closer_fraction_for_birth"]
            )
            and speed <= 2.5
        )
        motion_birth = bool(
            support >= int(self.parameters["minimum_motion_support_frames"])
            and row["pixel_count"] >= int(
                self.parameters["minimum_component_pixels_for_motion_birth"]
            )
            and float(self.parameters["minimum_world_speed_mps"])
                <= speed <= float(self.parameters["maximum_world_speed_mps"])
            and state.direction_consistency >= float(
                self.parameters["minimum_direction_consistency"]
            )
            and row["boundary_hazard_fraction"] <= float(
                self.parameters[
                    "maximum_boundary_hazard_fraction_for_motion_birth"
                ]
            )
        )
        return residual_birth or motion_birth


__all__ = ["PhysicalControlResidualV1"]
