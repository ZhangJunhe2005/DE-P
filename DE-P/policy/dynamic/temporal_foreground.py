"""Causal bounded world-frame temporal voxel foreground extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .types import DynamicPerceptionConfig


@dataclass
class _VoxelState:
    centroid: np.ndarray
    first_frame: int
    last_frame: int
    persistence: int
    stable_background: bool = False
    last_speed: float = 0.0


@dataclass
class _FreeVoxelState:
    last_frame: int
    persistence: int


class TemporalVoxelForeground:
    """Separate stable background using only history and the current frame.

    The first frames are a deliberate classification warm-up and emit no
    foreground.  A voxel becomes background only after repeated, low-motion
    support.  A non-background voxel must persist before it is emitted.
    """

    algorithm_name = "causal_world_temporal_voxel_v1"

    def __init__(self, config: DynamicPerceptionConfig):
        config.validate()
        self.config = config
        self._voxels = {}
        self._free_voxels = {}
        self._frame_index = -1
        self._last_timestamp = None
        self.last_diagnostics = {}

    def reset(self):
        self._voxels.clear()
        self._free_voxels.clear()
        self._frame_index = -1
        self._last_timestamp = None
        self.last_diagnostics = {}

    def _keys(self, points):
        return np.floor(points / self.config.background_voxel_size).astype(np.int64)

    def previously_free_mask(self, points_world, timestamp=None):
        """Read-only causal free-space evidence before the current-frame update."""
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_world must be finite [N,3]")
        output = np.zeros(len(points), dtype=bool)
        required_free_persistence = max(
            self.config.foreground_min_persistence,
            self.config.background_min_persistence,
        )
        stable_keys = [key for key, state in self._free_voxels.items()
                       if state.persistence >= required_free_persistence]
        if (stable_keys and len(points)
                and self._frame_index + 1 >= self.config.background_min_persistence):
            centroids = (np.asarray(stable_keys, dtype=np.float64) + 0.5) * (
                self.config.background_voxel_size
            )
            distance, _ = cKDTree(centroids).query(points, k=1, workers=1)
            output = distance <= self.config.background_match_distance
            stable_occupied = [state.centroid for state in self._voxels.values()
                               if state.stable_background]
            if stable_occupied:
                occupied_distance, _ = cKDTree(np.asarray(stable_occupied)).query(
                    points, k=1, workers=1
                )
                output &= occupied_distance > self.config.background_match_distance
            keys = self._keys(points)
            for index, key_array in enumerate(keys):
                if not output[index]:
                    continue
                state = self._voxels.get(tuple(int(value) for value in key_array))
                if state is None:
                    continue
                if state.stable_background:
                    output[index] = False
                    continue
                consecutive = state.last_frame == self._frame_index
                if consecutive and state.persistence + 1 >= self.config.background_min_persistence:
                    if timestamp is None or self._last_timestamp is None:
                        output[index] = False
                    else:
                        dt = max(float(timestamp) - self._last_timestamp, 1e-6)
                        speed = np.linalg.norm(points[index] - state.centroid) / dt
                        if speed <= self.config.background_update_speed_limit:
                            output[index] = False
        return output

    def extract(self, points_world, timestamp, camera_origin_world=None):
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_world must be finite [N,3]")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        camera_origin = None
        if camera_origin_world is not None:
            camera_origin = np.asarray(camera_origin_world, dtype=np.float64)
            if camera_origin.shape != (3,) or not np.isfinite(camera_origin).all():
                raise ValueError("camera_origin_world must be finite shape [3]")
        reset_due_gap = False
        if self._last_timestamp is not None:
            gap = timestamp - self._last_timestamp
            if gap <= 0:
                raise ValueError("temporal foreground timestamps must be strictly increasing")
            if gap > self.config.foreground_reset_gap:
                self.reset()
                reset_due_gap = True
        self._frame_index += 1
        frame_index = self._frame_index
        if not len(points):
            self._expire(frame_index)
            self._last_timestamp = timestamp
            self.last_diagnostics = {
                "input_points": 0, "foreground_points": 0,
                "stable_background_voxels": sum(
                    state.stable_background for state in self._voxels.values()
                ),
                "voxel_count": len(self._voxels), "reset_due_gap": reset_due_gap,
                "free_voxel_count": len(self._free_voxels),
                "warmup": frame_index < self.config.background_min_persistence,
            }
            return np.zeros((0,), dtype=bool)

        keys = self._keys(points)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        # O(N) grouped centroid accumulation.  The former boolean scan per
        # unique voxel was O(N * V) and dominated the 160x90 sensor period.
        centroids = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(centroids, inverse, points)
        counts = np.bincount(inverse, minlength=len(unique_keys))
        centroids /= counts[:, None]
        stable_states = [state for state in self._voxels.values()
                         if state.stable_background]
        explained = np.zeros(len(unique_keys), dtype=bool)
        if stable_states:
            stable_centroids = np.stack([state.centroid for state in stable_states])
            distance, _ = cKDTree(stable_centroids).query(centroids, k=1, workers=1)
            explained = distance <= self.config.background_match_distance

        previously_free = np.zeros(len(unique_keys), dtype=bool)
        stable_free_keys = [key for key, state in self._free_voxels.items()
                            if state.persistence >= self.config.foreground_min_persistence]
        if camera_origin is not None and stable_free_keys:
            free_centroids = (
                np.asarray(stable_free_keys, dtype=np.float64) + 0.5
            ) * self.config.background_voxel_size
            distance, _ = cKDTree(free_centroids).query(centroids, k=1, workers=1)
            previously_free = distance <= self.config.background_match_distance

        foreground_unique = np.zeros(len(unique_keys), dtype=bool)
        promoted = 0
        for index, (key_array, centroid) in enumerate(zip(unique_keys, centroids)):
            key = tuple(int(value) for value in key_array)
            previous = self._voxels.get(key)
            if previous is None:
                state = _VoxelState(centroid.copy(), frame_index, frame_index, 1)
            else:
                consecutive = previous.last_frame == frame_index - 1
                persistence = previous.persistence + 1 if consecutive else 1
                frame_dt = max(timestamp - self._last_timestamp, 1e-6)
                speed = float(np.linalg.norm(centroid - previous.centroid) / frame_dt)
                state = _VoxelState(
                    centroid=0.5 * previous.centroid + 0.5 * centroid,
                    first_frame=previous.first_frame if consecutive else frame_index,
                    last_frame=frame_index,
                    persistence=persistence,
                    stable_background=previous.stable_background,
                    last_speed=speed,
                )
            if (not state.stable_background
                    and state.persistence >= self.config.background_min_persistence
                    and state.last_speed <= self.config.background_update_speed_limit):
                state.stable_background = True
                promoted += 1
            self._voxels[key] = state
            foreground_unique[index] = bool(
                not explained[index]
                and not state.stable_background
                and (
                    previously_free[index]
                    if camera_origin is not None
                    else state.persistence >= self.config.foreground_min_persistence
                )
                and frame_index >= self.config.background_min_persistence
            )

        if camera_origin is not None:
            self._update_free_space(points, camera_origin, frame_index)
        self._expire(frame_index)
        self._bound_capacity()
        foreground_mask = foreground_unique[inverse]
        seed_count = int(foreground_mask.sum())
        self._last_timestamp = timestamp
        self.last_diagnostics = {
            "input_points": len(points),
            "foreground_points": int(foreground_mask.sum()),
            "foreground_seed_points": seed_count,
            "foreground_fraction": float(foreground_mask.mean()),
            "current_unique_voxels": len(unique_keys),
            "stable_explained_voxels": int(explained.sum()),
            "promoted_background_voxels": promoted,
            "stable_background_voxels": sum(
                state.stable_background for state in self._voxels.values()
            ),
            "voxel_count": len(self._voxels),
            "stable_free_voxels": sum(
                state.persistence >= self.config.foreground_min_persistence
                for state in self._free_voxels.values()
            ),
            "free_voxel_count": len(self._free_voxels),
            "previously_free_current_voxels": int(previously_free.sum()),
            "capacity": self.config.background_max_voxels,
            "reset_due_gap": reset_due_gap,
            "warmup": frame_index < self.config.background_min_persistence,
            "future_frames_used": 0,
        }
        return foreground_mask

    def _expire(self, frame_index):
        expired = [key for key, state in self._voxels.items()
                   if frame_index - state.last_frame > self.config.background_max_age]
        for key in expired:
            del self._voxels[key]
        expired_free = [key for key, state in self._free_voxels.items()
                        if frame_index - state.last_frame > self.config.background_max_age]
        for key in expired_free:
            del self._free_voxels[key]

    def _update_free_space(self, endpoints_world, camera_origin, frame_index):
        # Depth points arrive in deterministic image order.  A 2x endpoint
        # subsample retains ray coverage while bounding per-frame CPU work.
        endpoints_world = endpoints_world[::2]
        vectors = endpoints_world - camera_origin[None, :]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 2.0 * self.config.background_voxel_size
        if not valid.any():
            return
        vectors, lengths = vectors[valid], lengths[valid]
        directions = vectors / lengths[:, None]
        step = 2.0 * self.config.background_voxel_size
        sampled_keys = []
        for distance in np.arange(step, float(lengths.max()) - step, step):
            ray_mask = lengths > distance + step
            if ray_mask.any():
                samples = camera_origin + directions[ray_mask] * distance
                sampled_keys.append(self._keys(samples))
        if sampled_keys:
            free_keys = np.unique(np.concatenate(sampled_keys, axis=0), axis=0)
            for key_array in free_keys:
                key = tuple(int(value) for value in key_array)
                previous = self._free_voxels.get(key)
                consecutive = previous is not None and previous.last_frame == frame_index - 1
                self._free_voxels[key] = _FreeVoxelState(
                    last_frame=frame_index,
                    persistence=(previous.persistence + 1 if consecutive else 1),
                )
        # A measured endpoint is occupied now and therefore cannot remain free.
        for key_array in np.unique(self._keys(endpoints_world), axis=0):
            self._free_voxels.pop(tuple(int(value) for value in key_array), None)

    def _bound_capacity(self):
        excess = len(self._voxels) - self.config.background_max_voxels
        if excess > 0:
            ordered = sorted(self._voxels, key=lambda key: (
                self._voxels[key].last_frame,
                self._voxels[key].stable_background,
                key,
            ))
            for key in ordered[:excess]:
                del self._voxels[key]
        total_excess = (
            len(self._voxels) + len(self._free_voxels)
            - self.config.background_max_voxels
        )
        if total_excess > 0:
            ordered_free = sorted(
                self._free_voxels,
                key=lambda key: (self._free_voxels[key].last_frame, key),
            )
            for key in ordered_free[:total_excess]:
                del self._free_voxels[key]

    @property
    def voxel_count(self):
        return len(self._voxels) + len(self._free_voxels)
