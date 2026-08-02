"""Deterministic causal range-image foreground and observation extraction."""

from __future__ import annotations

from collections import deque
import time

import numpy as np

from .image_foreground_components import grow_seeded_components
from .pointcloud import camera_points_to_world
from .types import ClusterObservation, DepthFrame, DynamicPerceptionConfig


def reproject_history_depth(history: DepthFrame, current: DepthFrame):
    """Vectorized historical depth reprojection with a nearest-pixel z-buffer."""
    v, u = np.nonzero(history.valid_mask)
    z_history = history.depth_m[v, u]
    camera = history.camera_model
    points_history = np.stack((
        (u.astype(np.float32) - camera.cx) * z_history / camera.fx,
        (v.astype(np.float32) - camera.cy) * z_history / camera.fy,
        z_history,
    ), axis=1)
    world = camera_points_to_world(points_history, history.camera_pose_world)
    pose = current.camera_pose_world
    camera_points = (pose.rotation_world_from_camera.T
                     @ (world - pose.position_world).T).T
    z = camera_points[:, 2]
    camera = current.camera_model
    valid = np.isfinite(camera_points).all(axis=1) & (z >= camera.min_depth) & (z < camera.max_depth)
    u = np.rint(camera.fx * camera_points[:, 0] / np.maximum(z, 1e-9) + camera.cx).astype(np.int64)
    v = np.rint(camera.fy * camera_points[:, 1] / np.maximum(z, 1e-9) + camera.cy).astype(np.int64)
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    output = np.full(camera.width * camera.height, np.inf, dtype=np.float32)
    np.minimum.at(output, v[valid] * camera.width + u[valid], z[valid].astype(np.float32))
    return output.reshape(camera.height, camera.width)


def depth_edge_magnitude(depth, valid):
    edge = np.zeros_like(depth, dtype=np.float32)
    for axis in (0, 1):
        delta = np.abs(np.diff(depth, axis=axis))
        pair_valid = (valid[1:, :] & valid[:-1, :] if axis == 0
                      else valid[:, 1:] & valid[:, :-1])
        delta = np.where(pair_valid, delta, 0.0)
        if axis == 0:
            edge[1:, :] = np.maximum(edge[1:, :], delta)
            edge[:-1, :] = np.maximum(edge[:-1, :], delta)
        else:
            edge[:, 1:] = np.maximum(edge[:, 1:], delta)
            edge[:, :-1] = np.maximum(edge[:, :-1], delta)
    return edge


def _observation(component, frame, temporary_id, config):
    pixels = component.pixels_vu
    z = frame.depth_m[pixels[:, 0], pixels[:, 1]]
    u = pixels[:, 1].astype(np.float32); v = pixels[:, 0].astype(np.float32)
    camera = frame.camera_model
    points_camera = np.stack(((u - camera.cx) * z / camera.fx,
                              (v - camera.cy) * z / camera.fy, z), axis=1)
    points_world = camera_points_to_world(points_camera, frame.camera_pose_world)
    median = np.median(points_world, axis=0)
    distance = np.linalg.norm(points_world - median, axis=1)
    retain = max(1, int(round(len(points_world) * (1 - config.centroid_trim_fraction))))
    indices = np.argsort(distance, kind="stable")[:retain]
    retained_world, retained_camera = points_world[indices], points_camera[indices]
    covariance = (np.cov(retained_world, rowvar=False, ddof=1)
                  if retain > 1 else np.zeros((3, 3)))
    bbox = np.stack((points_world.min(axis=0), points_world.max(axis=0)))
    extent = bbox[1] - bbox[0]
    measurement_covariance = covariance / retain + np.diag(np.square(
        config.measurement_noise + config.centroid_extent_covariance_scale * extent
    ))
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    largest = max(float(eigenvalues[-1]), 1e-12)
    ratios = np.asarray(((eigenvalues[-1] - eigenvalues[-2]) / largest,
                         (eigenvalues[-2] - eigenvalues[0]) / largest))
    voxel_keys = np.unique(np.floor(
        points_world / config.background_voxel_size
    ).astype(np.int64), axis=0)
    if len(voxel_keys) > 256:
        voxel_keys = voxel_keys[np.linspace(0, len(voxel_keys) - 1, 256, dtype=int)]
    flat_pixels = np.sort(
        pixels[:, 0].astype(np.int64) * frame.camera_model.width
        + pixels[:, 1].astype(np.int64)
    )
    pixel_bbox = (
        int(pixels[:, 1].min()), int(pixels[:, 0].min()),
        int(pixels[:, 1].max()), int(pixels[:, 0].max()),
    )
    return ClusterObservation(
        temporary_cluster_id=int(temporary_id),
        centroid_world=retained_world.mean(axis=0),
        centroid_camera=retained_camera.mean(axis=0),
        point_count=len(points_world), bounding_box_world=bbox,
        position_covariance=measurement_covariance, timestamp=frame.timestamp,
        extent=extent, covariance_eigenvalues=eigenvalues, shape_ratios=ratios,
        voxel_signature=tuple(tuple(int(x) for x in row) for row in voxel_keys),
        foreground_support=component.confidence,
        pixel_indices=tuple(int(x) for x in flat_pixels),
        pixel_bbox=pixel_bbox,
        seed_pixel_count=int(component.seed_count),
        range_seed_count=int(component.range_seed_count),
        free_space_seed_count=int(component.free_space_seed_count),
        component_pixel_count=len(pixels),
        history_support=float(np.clip(
            component.mean_history_support / max(config.range_history_frames, 1),
            0.0, 1.0,
        )),
        direct_image_evidence=True,
        camera_inside_fraction=1.0,
        depth_consistency=float(np.clip(1.0 - component.static_consistency, 0.0, 1.0)),
    )


class CausalRangeImageForeground:
    algorithm_name = "causal_ego_compensated_range_image_hybrid_v1"

    def __init__(self, config: DynamicPerceptionConfig):
        config.validate(); self.config = config
        self._history = deque(maxlen=config.range_history_frames)
        self._last_timestamp = None
        self.last_diagnostics = {}
        self.last_range_seed = None
        self.last_free_seed = None
        self.last_component_mask = None

    def reset(self):
        self._history.clear(); self._last_timestamp = None; self.last_diagnostics = {}
        self.last_range_seed = self.last_free_seed = self.last_component_mask = None

    def extract(self, frame: DepthFrame, free_space_seed=None):
        if not isinstance(frame, DepthFrame):
            raise TypeError("range-image foreground requires DepthFrame")
        started = time.perf_counter()
        reset_due_gap = False
        if self._last_timestamp is not None:
            gap = frame.timestamp - self._last_timestamp
            if gap <= 0:
                raise ValueError("range history timestamps must be strictly increasing")
            if gap > self.config.range_reset_gap:
                self.reset(); reset_due_gap = True
        while self._history and frame.timestamp - self._history[0].timestamp > (
                self.config.range_history_max_age):
            self._history.popleft()
        predictions = [reproject_history_depth(old, frame) for old in self._history]
        shape = frame.depth_m.shape
        support = np.zeros(shape, dtype=np.int16)
        closer_support = np.zeros(shape, dtype=np.int16)
        static_support = np.zeros(shape, dtype=np.int16)
        max_residual = np.zeros(shape, dtype=np.float32)
        if predictions:
            predicted = np.stack(predictions)
            valid = np.isfinite(predicted) & frame.valid_mask[None]
            residuals = predicted - frame.depth_m[None]
            threshold = (self.config.range_abs_residual_threshold
                         + self.config.range_rel_residual_threshold * frame.depth_m)
            support = valid.sum(axis=0).astype(np.int16)
            closer_support = (valid & (residuals > threshold[None])).sum(axis=0).astype(np.int16)
            static_support = (valid & (np.abs(residuals) <=
                              self.config.range_static_consistency_threshold)).sum(axis=0).astype(np.int16)
            max_residual = np.max(np.where(valid, residuals, 0.0), axis=0).astype(np.float32)
        edge = depth_edge_magnitude(frame.depth_m, frame.valid_mask)
        static_explained = static_support >= self.config.range_min_history_support
        range_seed = ((closer_support >= self.config.range_min_history_support)
                      & ~static_explained
                      & (edge <= self.config.range_edge_guard_threshold)
                      & frame.valid_mask)
        free_seed = (np.zeros(shape, dtype=bool) if free_space_seed is None
                     else np.asarray(free_space_seed, dtype=bool))
        if free_seed.shape != shape:
            raise ValueError("free_space_seed must match raw depth shape")
        seeds = (range_seed | free_seed) & ~static_explained
        components = grow_seeded_components(
            frame, seeds, range_seed, free_seed, static_explained,
            support, max_residual, self.config,
        )
        observations = tuple(_observation(component, frame, index, self.config)
                             for index, component in enumerate(components))
        labels = np.full(len(frame.points_camera), -1, dtype=np.int64)
        point_lookup = np.full(shape, -1, dtype=np.int64)
        point_lookup[frame.pixels_uv[:, 1], frame.pixels_uv[:, 0]] = np.arange(
            len(frame.pixels_uv), dtype=np.int64
        )
        for index, component in enumerate(components):
            point_indices = point_lookup[component.pixels_vu[:, 0], component.pixels_vu[:, 1]]
            labels[point_indices[point_indices >= 0]] = index
        component_mask = np.zeros(shape, dtype=bool)
        for component in components:
            component_mask[component.pixels_vu[:, 0], component.pixels_vu[:, 1]] = True
        self.last_range_seed = range_seed.copy()
        self.last_free_seed = free_seed.copy()
        self.last_component_mask = component_mask
        self._history.append(frame); self._last_timestamp = frame.timestamp
        self.last_diagnostics = {
            "mode": "range_image_hybrid", "history_size": len(self._history),
            "history_capacity": self.config.range_history_frames,
            "future_frames_used": 0, "reset_due_gap": reset_due_gap,
            "range_seed_count": int(range_seed.sum()),
            "free_space_seed_count": int(free_seed.sum()),
            "static_explained_count": int(static_explained.sum()),
            "component_count": len(components),
            "component_pixel_count": int(sum(len(x.pixels_vu) for x in components)),
            "component_evidence": [dict(
                range_seed_count=x.range_seed_count,
                free_space_seed_count=x.free_space_seed_count,
                seed_count=x.seed_count, seed_fraction=x.seed_count / len(x.pixels_vu),
                history_support=x.mean_history_support, mean_residual=x.mean_residual,
                static_consistency=x.static_consistency, confidence=x.confidence,
                pixel_count=len(x.pixels_vu),
            ) for x in components],
            "foreground_ms": (time.perf_counter() - started) * 1000.0,
        }
        return observations, labels

    @property
    def history_size(self):
        return len(self._history)
