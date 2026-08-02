"""Visibility-aware causal residual components with bounded temporal support."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import label as connected_components

from .image_foreground_components import ImageForegroundComponent
from .range_image_foreground import _observation, depth_edge_magnitude
from .types import DepthFrame, DynamicPerceptionConfig
from .visibility_provenance_v1 import compute_visibility_provenance


@dataclass
class _Tracklet:
    centroid: np.ndarray
    velocity: np.ndarray
    support: int
    last_frame: int
    direction_consistency: float


class VisibilityAwareCausalResidualV1:
    algorithm_name = "visibility_aware_causal_residual_v1"

    def __init__(self, config: DynamicPerceptionConfig, parameters):
        self.config = config
        self.parameters = dict(parameters)
        self.history_frames = int(self.parameters.get("history_frames", 4))
        if not 3 <= self.history_frames <= 6:
            raise ValueError("history_frames must be in [3,6]")
        self._history = deque(maxlen=self.history_frames)
        self._tracklets = []
        self._frame_index = 0
        self.last_diagnostics = {}
        self.last_range_seed = None
        self.last_free_seed = None
        self.last_component_mask = None

    def reset(self):
        self._history.clear()
        self._tracklets.clear()
        self._frame_index = 0
        self.last_diagnostics = {}
        self.last_range_seed = self.last_free_seed = self.last_component_mask = None

    def _components(self, frame, provenance):
        threshold = (
            float(self.parameters.get("residual_abs_m", 0.08))
            + float(self.parameters.get("residual_relative", 0.02))
            * frame.depth_m
        )
        error = provenance.depth_consistency_error
        residual = (
            provenance.geometric_overlap
            & (np.abs(error) > threshold)
            & ~provenance.max_depth_transition
        )
        # Retain interiors rather than blanket-cropping image boundaries.
        edge = depth_edge_magnitude(frame.depth_m, frame.valid_mask)
        interior = residual & (edge <= max(
            self.config.range_edge_guard_threshold, 0.35
        ))
        seeds = residual & (
            interior
            | provenance.occlusion_candidate
            | provenance.disocclusion_candidate
        )
        labels, count = connected_components(
            seeds, structure=np.ones((3, 3), dtype=np.uint8)
        )
        rows = []
        minimum = int(self.parameters.get("minimum_component_pixels", 8))
        maximum = int(self.parameters.get("maximum_component_pixels", 4096))
        for component_id in range(1, count+1):
            pixels = np.argwhere(labels == component_id).astype(np.int32)
            if not minimum <= len(pixels) <= maximum:
                continue
            v, u = pixels.T
            depth = frame.depth_m[v, u]
            if (
                float(np.ptp(depth))
                > float(self.parameters.get(
                    "maximum_component_depth_span_m", 1.2
                ))
            ):
                continue
            geometric = float(np.mean(
                provenance.geometric_overlap[v, u]
            ))
            fov = float(np.mean(
                provenance.newly_visible_candidate[v, u]
            ))
            disocclusion = float(np.mean(
                provenance.disocclusion_candidate[v, u]
            ))
            invalid = float(np.mean(
                provenance.previous_depth_invalid[v, u]
                | provenance.current_depth_invalid[v, u]
                | provenance.max_depth_transition[v, u]
            ))
            closer = float(np.mean(
                provenance.occlusion_candidate[v, u]
            ))
            component = ImageForegroundComponent(
                pixels_vu=pixels,
                seed_count=len(pixels),
                free_space_seed_count=0,
                range_seed_count=int(
                    provenance.occlusion_candidate[v, u].sum()
                ),
                mean_history_support=float(len(self._history)),
                mean_residual=float(np.mean(np.abs(error[v, u]))),
                static_consistency=float(np.mean(
                    provenance.stable_overlap[v, u]
                )),
                confidence=float(np.clip(
                    .4*geometric+.3*max(closer, disocclusion)
                    +.3*min(1.0, len(pixels)/32), 0, 1,
                )),
            )
            observation = _observation(
                component, frame, len(rows), self.config
            )
            rows.append({
                "component": component,
                "observation": observation,
                "geometric_fraction": geometric,
                "fov_fraction": fov,
                "disocclusion_fraction": disocclusion,
                "invalid_fraction": invalid,
                "closer_fraction": closer,
            })
        return rows, residual

    def _update_tracklets(self, rows, timestamp):
        used = set()
        dt = (
            timestamp-self._history[-1].timestamp
            if self._history else 0.1
        )
        for row in rows:
            centroid = row["observation"].centroid_world
            choices = [
                (float(np.linalg.norm(centroid-state.centroid)), index)
                for index, state in enumerate(self._tracklets)
                if index not in used
                and self._frame_index-state.last_frame <= 1
            ]
            choices = [
                value for value in choices
                if value[0] <= float(
                    self.parameters.get("association_distance_m", 0.65)
                )
            ]
            if choices:
                _, index = min(choices)
                state = self._tracklets[index]
                velocity = (centroid-state.centroid)/max(dt, 1e-6)
                old_norm = np.linalg.norm(state.velocity)
                new_norm = np.linalg.norm(velocity)
                cosine = (
                    float(np.dot(state.velocity, velocity)
                          /(old_norm*new_norm))
                    if old_norm > 1e-6 and new_norm > 1e-6 else 1.0
                )
                state.direction_consistency = min(
                    state.direction_consistency, max(-1.0, cosine)
                )
                state.velocity = velocity
                state.centroid = centroid.copy()
                state.support += 1
                state.last_frame = self._frame_index
                used.add(index)
            else:
                state = _Tracklet(
                    centroid.copy(), np.zeros(3), 1,
                    self._frame_index, 1.0,
                )
                self._tracklets.append(state)
                index = len(self._tracklets)-1
                used.add(index)
            row["tracklet"] = state
        self._tracklets = sorted(
            (
                state for state in self._tracklets
                if self._frame_index-state.last_frame <= self.history_frames
            ),
            key=lambda state: (-state.support, state.last_frame),
        )[:int(self.parameters.get("maximum_tracklets", 32))]

    def _accept(self, row):
        state = row["tracklet"]
        support = int(self.parameters.get(
            "minimum_temporal_support_frames", 2
        ))
        if state.support < support:
            return False
        if row["fov_fraction"] > float(
            self.parameters.get("maximum_fov_fraction", 0.5)
        ):
            return False
        if row["invalid_fraction"] > float(
            self.parameters.get("maximum_invalid_fraction", 0.5)
        ):
            return False
        speed = float(np.linalg.norm(state.velocity))
        if row["closer_fraction"] >= float(
            self.parameters.get("minimum_closer_fraction_for_birth", 0.05)
        ):
            return True
        return bool(
            row["disocclusion_fraction"] > 0
            and speed >= float(
                self.parameters.get("minimum_world_speed_mps", 0.15)
            )
            and speed <= float(
                self.parameters.get("maximum_world_speed_mps", 3.0)
            )
            and state.direction_consistency >= float(
                self.parameters.get("minimum_direction_consistency", 0.7)
            )
        )

    def extract(self, frame: DepthFrame, free_space_seed=None):
        if not isinstance(frame, DepthFrame):
            raise TypeError("visibility-aware residual requires DepthFrame")
        started = time.perf_counter()
        if not self._history:
            rows, residual = [], np.zeros_like(frame.valid_mask)
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
            rows, residual = self._components(frame, provenance)
            self._update_tracklets(rows, frame.timestamp)
        accepted = [row for row in rows if self._accept(row)]
        observations = tuple(row["observation"] for row in accepted)
        labels = np.full(len(frame.points_camera), -1, dtype=np.int64)
        lookup = np.full(frame.depth_m.shape, -1, dtype=np.int64)
        lookup[
            frame.pixels_uv[:, 1], frame.pixels_uv[:, 0]
        ] = np.arange(len(frame.pixels_uv))
        component_mask = np.zeros(frame.depth_m.shape, dtype=bool)
        for index, row in enumerate(accepted):
            pixels = row["component"].pixels_vu
            point_indices = lookup[pixels[:, 0], pixels[:, 1]]
            labels[point_indices[point_indices >= 0]] = index
            component_mask[pixels[:, 0], pixels[:, 1]] = True
        self._history.append(frame)
        self._frame_index += 1
        self.last_range_seed = residual.copy()
        self.last_free_seed = np.zeros_like(residual)
        self.last_component_mask = component_mask
        self.last_diagnostics = {
            "mode": self.algorithm_name,
            "history_size": len(self._history),
            "history_capacity": self.history_frames,
            "future_frames_used": 0,
            "runtime_gt_used": False,
            "range_seed_count": int(residual.sum()),
            "component_count": len(accepted),
            "component_pixel_count": int(component_mask.sum()),
            "candidate_component_count": len(rows),
            "tracklet_count": len(self._tracklets),
            "weak_support_pixels": int(residual.sum()),
            "components": [{
                "pixel_count": len(row["component"].pixels_vu),
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
                "accepted": self._accept(row),
            } for row in rows],
            "foreground_ms": (time.perf_counter()-started)*1000,
        }
        return observations, labels

    @property
    def history_size(self):
        return len(self._history)


__all__ = ["VisibilityAwareCausalResidualV1"]
