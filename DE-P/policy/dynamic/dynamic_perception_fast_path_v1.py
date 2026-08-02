"""Semantics-preserving DIRO1 fast path beside the frozen reference path."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from .attention import build_dynamic_attention_with_projection
from .dynamic_frame_artifacts_v1 import DynamicFrameArtifactBuilderV1
from .physical_control_residual_v1 import PhysicalControlResidualV1
from .temporal_foreground import _FreeVoxelState, _VoxelState
from .track_manager import TrackManager
from .types import (
    CameraModel, DynamicPerceptionConfig, DynamicPerceptionResult, Pose,
)


FAST_PATH_VERSION = "dynamic_perception_fast_path_v1"
_KEY_DTYPE = np.dtype([
    ("x", "<i8"), ("y", "<i8"), ("z", "<i8"),
])


def _structured(keys):
    return np.ascontiguousarray(
        keys, dtype=np.int64
    ).view(_KEY_DTYPE).reshape(-1)


def _keys_from_structured(values):
    return np.ascontiguousarray(values).view(np.int64).reshape(-1, 3)


def _lookup(sorted_keys, query):
    if not len(sorted_keys) or not len(query):
        return (
            np.zeros(len(query), dtype=np.int64),
            np.zeros(len(query), dtype=bool),
        )
    source = _structured(sorted_keys)
    target = _structured(query)
    positions = np.searchsorted(source, target)
    valid = positions < len(source)
    matched = np.zeros(len(query), dtype=bool)
    matched[valid] = source[positions[valid]] == target[valid]
    return positions, matched


@dataclass
class _OccupiedTable:
    keys: np.ndarray
    centroids: np.ndarray
    first: np.ndarray
    last: np.ndarray
    persistence: np.ndarray
    stable: np.ndarray
    speed: np.ndarray

    @classmethod
    def empty(cls):
        return cls(
            np.empty((0, 3), np.int64), np.empty((0, 3), np.float64),
            np.empty(0, np.int64), np.empty(0, np.int64),
            np.empty(0, np.int64), np.empty(0, bool),
            np.empty(0, np.float64),
        )

    def select(self, keep):
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name)[keep])


@dataclass
class _FreeTable:
    keys: np.ndarray
    last: np.ndarray
    persistence: np.ndarray

    @classmethod
    def empty(cls):
        return cls(
            np.empty((0, 3), np.int64),
            np.empty(0, np.int64), np.empty(0, np.int64),
        )

    def select(self, keep):
        self.keys = self.keys[keep]
        self.last = self.last[keep]
        self.persistence = self.persistence[keep]


class FastTemporalVoxelForegroundV1:
    """Vectorized storage for the frozen TemporalVoxelForeground state machine."""

    algorithm_name = "causal_world_temporal_voxel_fast_v1"

    def __init__(self, config: DynamicPerceptionConfig):
        config.validate()
        self.config = config
        self._occupied = _OccupiedTable.empty()
        self._free = _FreeTable.empty()
        self._frame_index = -1
        self._last_timestamp = None
        self.last_diagnostics = {}

    def reset(self):
        self._occupied = _OccupiedTable.empty()
        self._free = _FreeTable.empty()
        self._frame_index = -1
        self._last_timestamp = None
        self.last_diagnostics = {}

    def _keys(self, points):
        return np.floor(
            points/self.config.background_voxel_size
        ).astype(np.int64)

    def previously_free_mask(self, points_world, timestamp=None):
        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("points_world must be finite [N,3]")
        output = np.zeros(len(points), dtype=bool)
        required = max(
            self.config.foreground_min_persistence,
            self.config.background_min_persistence,
        )
        stable_free = self._free.keys[self._free.persistence >= required]
        if (
            len(stable_free) and len(points)
            and self._frame_index+1 >= self.config.background_min_persistence
        ):
            centroids = (
                stable_free.astype(np.float64)+.5
            )*self.config.background_voxel_size
            distance, _ = cKDTree(centroids).query(
                points, k=1, workers=1
            )
            output = distance <= self.config.background_match_distance
            stable_occupied = self._occupied.centroids[
                self._occupied.stable
            ]
            if len(stable_occupied):
                distance, _ = cKDTree(stable_occupied).query(
                    points, k=1, workers=1
                )
                output &= (
                    distance > self.config.background_match_distance
                )
            query_indices = np.flatnonzero(output)
            if len(query_indices):
                positions, matched = _lookup(
                    self._occupied.keys,
                    self._keys(points[query_indices]),
                )
                indices = query_indices[matched]
                states = positions[matched]
                output[indices[self._occupied.stable[states]]] = False
                candidate = (
                    ~self._occupied.stable[states]
                    & (self._occupied.last[states] == self._frame_index)
                    & (
                        self._occupied.persistence[states]+1
                        >= self.config.background_min_persistence
                    )
                )
                if candidate.any():
                    candidate_indices = indices[candidate]
                    candidate_states = states[candidate]
                    if timestamp is None or self._last_timestamp is None:
                        output[candidate_indices] = False
                    else:
                        dt = max(
                            float(timestamp)-self._last_timestamp, 1e-6
                        )
                        speed = np.linalg.norm(
                            points[candidate_indices]
                            - self._occupied.centroids[candidate_states],
                            axis=1,
                        )/dt
                        output[candidate_indices[
                            speed <= self.config.background_update_speed_limit
                        ]] = False
        return output

    def _update_occupied(self, unique_keys, centroids, timestamp, frame):
        old = self._occupied
        union = np.union1d(
            _structured(old.keys), _structured(unique_keys)
        )
        union_keys = _keys_from_structured(union)
        old_positions = np.searchsorted(union, _structured(old.keys))
        new_positions = np.searchsorted(union, _structured(unique_keys))
        table = _OccupiedTable(
            union_keys,
            np.zeros((len(union), 3), np.float64),
            np.full(len(union), frame, np.int64),
            np.full(len(union), -10**9, np.int64),
            np.zeros(len(union), np.int64),
            np.zeros(len(union), bool),
            np.zeros(len(union), np.float64),
        )
        if len(old.keys):
            for name in (
                "centroids", "first", "last", "persistence",
                "stable", "speed",
            ):
                getattr(table, name)[old_positions] = getattr(old, name)
        previous_exists = table.last[new_positions] > -10**8
        consecutive = (
            previous_exists
            & (table.last[new_positions] == frame-1)
        )
        previous_centroids = table.centroids[new_positions].copy()
        frame_dt = max(
            timestamp-self._last_timestamp, 1e-6
        ) if self._last_timestamp is not None else 1e-6
        speed = np.zeros(len(unique_keys), np.float64)
        speed[previous_exists] = np.linalg.norm(
            centroids[previous_exists]-previous_centroids[previous_exists],
            axis=1,
        )/frame_dt
        updated_centroids = centroids.copy()
        updated_centroids[previous_exists] = (
            .5*previous_centroids[previous_exists]
            +.5*centroids[previous_exists]
        )
        table.centroids[new_positions] = updated_centroids
        table.first[new_positions] = np.where(
            consecutive, table.first[new_positions], frame
        )
        table.last[new_positions] = frame
        table.persistence[new_positions] = np.where(
            consecutive, table.persistence[new_positions]+1, 1
        )
        table.speed[new_positions] = speed
        promote = (
            ~table.stable[new_positions]
            & (
                table.persistence[new_positions]
                >= self.config.background_min_persistence
            )
            & (
                table.speed[new_positions]
                <= self.config.background_update_speed_limit
            )
        )
        table.stable[new_positions[promote]] = True
        self._occupied = table
        return new_positions, int(promote.sum())

    def _update_free_space(self, endpoints, origin, frame):
        endpoints = endpoints[::2]
        vectors = endpoints-origin[None]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 2.*self.config.background_voxel_size
        if valid.any():
            vectors, lengths = vectors[valid], lengths[valid]
            directions = vectors/lengths[:, None]
            step = 2.*self.config.background_voxel_size
            sampled = []
            for distance in np.arange(
                step, float(lengths.max())-step, step
            ):
                ray_mask = lengths > distance+step
                if ray_mask.any():
                    sampled.append(self._keys(
                        origin+directions[ray_mask]*distance
                    ))
            free_keys = (
                np.unique(np.concatenate(sampled), axis=0)
                if sampled else np.empty((0, 3), np.int64)
            )
        else:
            free_keys = np.empty((0, 3), np.int64)
        old = self._free
        union = np.union1d(
            _structured(old.keys), _structured(free_keys)
        )
        keys = _keys_from_structured(union)
        old_positions = np.searchsorted(union, _structured(old.keys))
        new_positions = np.searchsorted(union, _structured(free_keys))
        last = np.full(len(union), -10**9, np.int64)
        persistence = np.zeros(len(union), np.int64)
        if len(old.keys):
            last[old_positions] = old.last
            persistence[old_positions] = old.persistence
        consecutive = last[new_positions] == frame-1
        persistence[new_positions] = np.where(
            consecutive, persistence[new_positions]+1, 1
        )
        last[new_positions] = frame
        self._free = _FreeTable(keys, last, persistence)
        occupied_endpoint_keys = np.unique(
            self._keys(endpoints), axis=0
        )
        positions, matched = _lookup(
            self._free.keys, occupied_endpoint_keys
        )
        if matched.any():
            keep = np.ones(len(self._free.keys), bool)
            keep[positions[matched]] = False
            self._free.select(keep)

    def _expire_and_bound(self, frame):
        age = self.config.background_max_age
        self._occupied.select(frame-self._occupied.last <= age)
        self._free.select(frame-self._free.last <= age)
        maximum = self.config.background_max_voxels
        excess = len(self._occupied.keys)-maximum
        if excess > 0:
            order = np.lexsort((
                self._occupied.keys[:, 2],
                self._occupied.keys[:, 1],
                self._occupied.keys[:, 0],
                self._occupied.stable,
                self._occupied.last,
            ))
            keep = np.ones(len(order), bool)
            keep[order[:excess]] = False
            self._occupied.select(keep)
        total_excess = (
            len(self._occupied.keys)+len(self._free.keys)-maximum
        )
        if total_excess > 0:
            order = np.lexsort((
                self._free.keys[:, 2], self._free.keys[:, 1],
                self._free.keys[:, 0], self._free.last,
            ))
            keep = np.ones(len(order), bool)
            keep[order[:total_excess]] = False
            self._free.select(keep)

    def extract(self, points_world, timestamp, camera_origin_world=None):
        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("points_world must be finite [N,3]")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        origin = (
            None if camera_origin_world is None else
            np.asarray(camera_origin_world, dtype=np.float64)
        )
        if origin is not None and (
            origin.shape != (3,) or not np.isfinite(origin).all()
        ):
            raise ValueError("camera_origin_world must be finite shape [3]")
        reset_due_gap = False
        if self._last_timestamp is not None:
            gap = timestamp-self._last_timestamp
            if gap <= 0:
                raise ValueError("timestamps must increase strictly")
            if gap > self.config.foreground_reset_gap:
                self.reset()
                reset_due_gap = True
        self._frame_index += 1
        frame = self._frame_index
        if not len(points):
            self._expire_and_bound(frame)
            self._last_timestamp = timestamp
            self.last_diagnostics = {
                "input_points": 0, "foreground_points": 0,
                "stable_background_voxels": int(
                    self._occupied.stable.sum()
                ),
                "voxel_count": len(self._occupied.keys),
                "reset_due_gap": reset_due_gap,
                "free_voxel_count": len(self._free.keys),
                "warmup":
                    frame < self.config.background_min_persistence,
            }
            return np.zeros(0, bool)
        keys = self._keys(points)
        unique_keys, inverse = np.unique(
            keys, axis=0, return_inverse=True
        )
        centroids = np.zeros((len(unique_keys), 3), np.float64)
        np.add.at(centroids, inverse, points)
        counts = np.bincount(inverse, minlength=len(unique_keys))
        centroids /= counts[:, None]
        stable_centroids = self._occupied.centroids[
            self._occupied.stable
        ]
        explained = np.zeros(len(unique_keys), bool)
        if len(stable_centroids):
            distance, _ = cKDTree(stable_centroids).query(
                centroids, k=1, workers=1
            )
            explained = (
                distance <= self.config.background_match_distance
            )
        stable_free = self._free.keys[
            self._free.persistence
            >= self.config.foreground_min_persistence
        ]
        previously_free = np.zeros(len(unique_keys), bool)
        if origin is not None and len(stable_free):
            free_centroids = (
                stable_free.astype(np.float64)+.5
            )*self.config.background_voxel_size
            distance, _ = cKDTree(free_centroids).query(
                centroids, k=1, workers=1
            )
            previously_free = (
                distance <= self.config.background_match_distance
            )
        positions, promoted = self._update_occupied(
            unique_keys, centroids, timestamp, frame
        )
        state = self._occupied
        foreground_unique = (
            ~explained & ~state.stable[positions]
            & (
                previously_free if origin is not None else
                state.persistence[positions]
                >= self.config.foreground_min_persistence
            )
            & (frame >= self.config.background_min_persistence)
        )
        if origin is not None:
            self._update_free_space(points, origin, frame)
        self._expire_and_bound(frame)
        foreground = foreground_unique[inverse]
        self._last_timestamp = timestamp
        self.last_diagnostics = {
            "input_points": len(points),
            "foreground_points": int(foreground.sum()),
            "foreground_seed_points": int(foreground.sum()),
            "foreground_fraction": float(foreground.mean()),
            "current_unique_voxels": len(unique_keys),
            "stable_explained_voxels": int(explained.sum()),
            "promoted_background_voxels": promoted,
            "stable_background_voxels":
                int(self._occupied.stable.sum()),
            "voxel_count": len(self._occupied.keys),
            "stable_free_voxels": int((
                self._free.persistence
                >= self.config.foreground_min_persistence
            ).sum()),
            "free_voxel_count": len(self._free.keys),
            "previously_free_current_voxels":
                int(previously_free.sum()),
            "capacity": self.config.background_max_voxels,
            "reset_due_gap": reset_due_gap,
            "warmup":
                frame < self.config.background_min_persistence,
            "future_frames_used": 0,
        }
        return foreground

    @property
    def voxel_count(self):
        return len(self._occupied.keys)+len(self._free.keys)


class CachedTemporalVoxelForegroundV1:
    """Frozen dict state machine with O(changed) expiry and cached counters."""

    algorithm_name = "causal_world_temporal_voxel_cached_v1"

    def __init__(self, config):
        config.validate()
        self.config = config
        self.reset()

    def reset(self):
        self._voxels = {}
        self._free_voxels = {}
        self._frame_index = -1
        self._last_timestamp = None
        self._occupied_by_frame = {}
        self._free_by_frame = {}
        self._stable_background_count = 0
        self._foreground_stable_free = set()
        self._query_stable_free = set()
        self._stable_centroids_cache = np.empty((0, 3), np.float64)
        self._foreground_free_centroids_cache = np.empty(
            (0, 3), np.float64
        )
        self._query_free_centroids_cache = np.empty((0, 3), np.float64)
        self.last_diagnostics = {}

    def _keys(self, points):
        return np.floor(
            points/self.config.background_voxel_size
        ).astype(np.int64)

    @staticmethod
    def _free_key(row):
        return np.ascontiguousarray(row, dtype=np.int64).tobytes()

    @staticmethod
    def _free_keys(rows):
        """Return canonical 24-byte keys with conversion performed in C."""
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if not len(rows):
            return []
        # void preserves trailing zero bytes; string dtype would truncate
        # them and could collapse distinct signed int64 triples.
        return rows.view("V24").reshape(-1).tolist()

    @staticmethod
    def _free_key_tuple(key):
        return tuple(int(value) for value in np.frombuffer(
            key, dtype=np.int64, count=3
        ))

    @staticmethod
    def _free_key_array(keys):
        if not keys:
            return np.empty((0, 3), np.int64)
        return np.frombuffer(
            b"".join(keys), dtype=np.int64
        ).reshape(-1, 3)

    def _refresh_free_membership(self, key, persistence):
        if persistence is None:
            self._foreground_stable_free.discard(key)
            self._query_stable_free.discard(key)
            return
        if persistence >= self.config.foreground_min_persistence:
            self._foreground_stable_free.add(key)
        else:
            self._foreground_stable_free.discard(key)
        required = max(
            self.config.foreground_min_persistence,
            self.config.background_min_persistence,
        )
        if persistence >= required:
            self._query_stable_free.add(key)
        else:
            self._query_stable_free.discard(key)

    def _refresh_state_caches(self):
        stable = [
            state.centroid for state in self._voxels.values()
            if state.stable_background
        ]
        self._stable_centroids_cache = (
            np.stack(stable) if stable
            else np.empty((0, 3), np.float64)
        )
        self._foreground_free_centroids_cache = (
            self._free_key_array(self._foreground_stable_free)
            .astype(np.float64)+.5
        )*self.config.background_voxel_size
        self._query_free_centroids_cache = (
            self._free_key_array(self._query_stable_free)
            .astype(np.float64)+.5
        )*self.config.background_voxel_size

    def previously_free_mask(self, points_world, timestamp=None):
        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("points_world must be finite [N,3]")
        output = np.zeros(len(points), dtype=bool)
        if (
            self._query_stable_free and len(points)
            and self._frame_index+1 >= self.config.background_min_persistence
        ):
            distance, _ = cKDTree(
                self._query_free_centroids_cache
            ).query(
                points, k=1, workers=1
            )
            output = distance <= self.config.background_match_distance
            if len(self._stable_centroids_cache):
                distance, _ = cKDTree(
                    self._stable_centroids_cache
                ).query(
                    points, k=1, workers=1
                )
                output &= (
                    distance > self.config.background_match_distance
                )
            keys = self._keys(points)
            for index in np.flatnonzero(output):
                state = self._voxels.get(tuple(
                    int(value) for value in keys[index]
                ))
                if state is None:
                    continue
                if state.stable_background:
                    output[index] = False
                    continue
                consecutive = state.last_frame == self._frame_index
                if (
                    consecutive
                    and state.persistence+1
                    >= self.config.background_min_persistence
                ):
                    if timestamp is None or self._last_timestamp is None:
                        output[index] = False
                    else:
                        dt = max(
                            float(timestamp)-self._last_timestamp, 1e-6
                        )
                        speed = np.linalg.norm(
                            points[index]-state.centroid
                        )/dt
                        if (
                            speed
                            <= self.config.background_update_speed_limit
                        ):
                            output[index] = False
        return output

    def _expire(self, frame):
        target = frame-self.config.background_max_age-1
        for key in self._occupied_by_frame.pop(target, ()):
            state = self._voxels.get(key)
            if state is not None and state.last_frame == target:
                if state.stable_background:
                    self._stable_background_count -= 1
                del self._voxels[key]
        for key in self._free_by_frame.pop(target, ()):
            packed = self._free_voxels.get(key)
            if packed is not None and packed >> 3 == target:
                del self._free_voxels[key]
                self._refresh_free_membership(key, None)

    def _update_free_space(self, endpoints_world, origin, frame):
        endpoints_world = endpoints_world[::2]
        vectors = endpoints_world-origin[None]
        lengths = np.linalg.norm(vectors, axis=1)
        valid = lengths > 2.*self.config.background_voxel_size
        touched = []
        if valid.any():
            vectors, lengths = vectors[valid], lengths[valid]
            directions = vectors/lengths[:, None]
            step = 2.*self.config.background_voxel_size
            sampled = []
            for distance in np.arange(
                step, float(lengths.max())-step, step
            ):
                ray_mask = lengths > distance+step
                if ray_mask.any():
                    sampled.append(self._keys(
                        origin+directions[ray_mask]*distance
                    ))
            if sampled:
                free_keys = np.unique(
                    np.concatenate(sampled), axis=0
                )
                keys = self._free_keys(free_keys)
                previous = np.fromiter(
                    (
                        self._free_voxels.get(key, -1)
                        for key in keys
                    ),
                    dtype=np.int64, count=len(keys),
                )
                previous_frame = np.right_shift(previous, 3)
                previous_persistence = np.where(
                    previous >= 0, np.bitwise_and(previous, 7), 0
                )
                consecutive = (
                    (previous >= 0) & (previous_frame == frame-1)
                )
                persistence = np.where(
                    consecutive,
                    np.minimum(previous_persistence+1, 7),
                    1,
                )
                packed = (frame << 3) | persistence
                self._free_voxels.update(zip(
                    keys, (int(value) for value in packed)
                ))
                foreground_threshold = (
                    self.config.foreground_min_persistence
                )
                query_threshold = max(
                    foreground_threshold,
                    self.config.background_min_persistence,
                )
                for membership, threshold in (
                    (self._foreground_stable_free,
                     foreground_threshold),
                    (self._query_stable_free, query_threshold),
                ):
                    changed = (
                        (previous_persistence >= threshold)
                        != (persistence >= threshold)
                    )
                    promoted = [
                        keys[index] for index in np.flatnonzero(
                            changed & (persistence >= threshold)
                        )
                    ]
                    demoted = [
                        keys[index] for index in np.flatnonzero(
                            changed & (persistence < threshold)
                        )
                    ]
                    membership.update(promoted)
                    membership.difference_update(demoted)
                touched.extend(keys)
        for key in self._free_keys(np.unique(
            self._keys(endpoints_world), axis=0
        )):
            if self._free_voxels.pop(key, None) is not None:
                self._refresh_free_membership(key, None)
        self._free_by_frame[frame] = tuple(touched)

    def _bound_capacity(self):
        excess = (
            len(self._voxels)-self.config.background_max_voxels
        )
        if excess > 0:
            ordered = sorted(self._voxels, key=lambda key: (
                self._voxels[key].last_frame,
                self._voxels[key].stable_background, key,
            ))
            for key in ordered[:excess]:
                if self._voxels[key].stable_background:
                    self._stable_background_count -= 1
                del self._voxels[key]
        total_excess = (
            len(self._voxels)+len(self._free_voxels)
            -self.config.background_max_voxels
        )
        if total_excess > 0:
            ordered = sorted(
                self._free_voxels,
                key=lambda key: (
                    self._free_voxels[key] >> 3,
                    self._free_key_tuple(key),
                ),
            )
            for key in ordered[:total_excess]:
                del self._free_voxels[key]
                self._refresh_free_membership(key, None)

    def extract(self, points_world, timestamp, camera_origin_world=None):
        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("points_world must be finite [N,3]")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        origin = (
            None if camera_origin_world is None else
            np.asarray(camera_origin_world, dtype=np.float64)
        )
        if origin is not None and (
            origin.shape != (3,) or not np.isfinite(origin).all()
        ):
            raise ValueError("camera_origin_world must be finite shape [3]")
        reset_due_gap = False
        if self._last_timestamp is not None:
            gap = timestamp-self._last_timestamp
            if gap <= 0:
                raise ValueError("timestamps must increase strictly")
            if gap > self.config.foreground_reset_gap:
                self.reset()
                reset_due_gap = True
        self._frame_index += 1
        frame = self._frame_index
        if not len(points):
            self._expire(frame)
            self._refresh_state_caches()
            self._last_timestamp = timestamp
            self.last_diagnostics = {
                "input_points": 0, "foreground_points": 0,
                "stable_background_voxels":
                    self._stable_background_count,
                "voxel_count": len(self._voxels),
                "reset_due_gap": reset_due_gap,
                "free_voxel_count": len(self._free_voxels),
                "warmup":
                    frame < self.config.background_min_persistence,
            }
            return np.zeros(0, bool)
        keys = self._keys(points)
        unique_keys, inverse = np.unique(
            keys, axis=0, return_inverse=True
        )
        centroids = np.zeros((len(unique_keys), 3), np.float64)
        np.add.at(centroids, inverse, points)
        counts = np.bincount(inverse, minlength=len(unique_keys))
        centroids /= counts[:, None]
        explained = np.zeros(len(unique_keys), bool)
        if len(self._stable_centroids_cache):
            distance, _ = cKDTree(
                self._stable_centroids_cache
            ).query(
                centroids, k=1, workers=1
            )
            explained = (
                distance <= self.config.background_match_distance
            )
        previously_free = np.zeros(len(unique_keys), bool)
        if origin is not None and self._foreground_stable_free:
            distance, _ = cKDTree(
                self._foreground_free_centroids_cache
            ).query(
                centroids, k=1, workers=1
            )
            previously_free = (
                distance <= self.config.background_match_distance
            )
        foreground_unique = np.zeros(len(unique_keys), bool)
        promoted = 0
        touched = []
        for index, (row, centroid) in enumerate(
            zip(unique_keys, centroids)
        ):
            key = tuple(int(value) for value in row)
            previous = self._voxels.get(key)
            if previous is None:
                state = _VoxelState(
                    centroid.copy(), frame, frame, 1
                )
            else:
                consecutive = previous.last_frame == frame-1
                persistence = (
                    previous.persistence+1 if consecutive else 1
                )
                dt = max(timestamp-self._last_timestamp, 1e-6)
                speed = float(np.linalg.norm(
                    centroid-previous.centroid
                )/dt)
                state = _VoxelState(
                    .5*previous.centroid+.5*centroid,
                    previous.first_frame if consecutive else frame,
                    frame, persistence, previous.stable_background,
                    speed,
                )
            if (
                not state.stable_background
                and state.persistence
                >= self.config.background_min_persistence
                and state.last_speed
                <= self.config.background_update_speed_limit
            ):
                state.stable_background = True
                promoted += 1
                self._stable_background_count += 1
            self._voxels[key] = state
            touched.append(key)
            foreground_unique[index] = bool(
                not explained[index] and not state.stable_background
                and (
                    previously_free[index] if origin is not None else
                    state.persistence
                    >= self.config.foreground_min_persistence
                )
                and frame >= self.config.background_min_persistence
            )
        self._occupied_by_frame[frame] = tuple(touched)
        if origin is not None:
            self._update_free_space(points, origin, frame)
        self._expire(frame)
        self._bound_capacity()
        self._refresh_state_caches()
        foreground = foreground_unique[inverse]
        self._last_timestamp = timestamp
        self.last_diagnostics = {
            "input_points": len(points),
            "foreground_points": int(foreground.sum()),
            "foreground_seed_points": int(foreground.sum()),
            "foreground_fraction": float(foreground.mean()),
            "current_unique_voxels": len(unique_keys),
            "stable_explained_voxels": int(explained.sum()),
            "promoted_background_voxels": promoted,
            "stable_background_voxels":
                self._stable_background_count,
            "voxel_count": len(self._voxels),
            "stable_free_voxels":
                len(self._foreground_stable_free),
            "free_voxel_count": len(self._free_voxels),
            "previously_free_current_voxels":
                int(previously_free.sum()),
            "capacity": self.config.background_max_voxels,
            "reset_due_gap": reset_due_gap,
            "warmup":
                frame < self.config.background_min_persistence,
            "future_frames_used": 0,
        }
        return foreground

    @property
    def voxel_count(self):
        return len(self._voxels)+len(self._free_voxels)


class DynamicPerceptionFastPathV1:
    """Independent fast implementation with frozen detector/tracker semantics."""

    def __init__(
        self, config: DynamicPerceptionConfig, feature_shape=(3, 5),
        attention_device="cpu", foreground_parameters=None,
    ):
        config.validate()
        self.config = config
        self.feature_shape = tuple(int(value) for value in feature_shape)
        self.attention_device = attention_device
        self.track_manager = TrackManager(config)
        self.foreground = CachedTemporalVoxelForegroundV1(config)
        self.range_foreground = PhysicalControlResidualV1(
            config, foreground_parameters or {}
        )
        self._artifact_builder = None
        self._last_timestamp = None
        self._frame_index = 0
        self._next_observation_id = 0
        self.last_artifacts = None

    def reset(self):
        self.track_manager.reset()
        self.foreground.reset()
        self.range_foreground.reset()
        self._artifact_builder = None
        self._last_timestamp = None
        self._frame_index = 0
        self._next_observation_id = 0
        self.last_artifacts = None

    def update_depth(
        self, depth, camera_pose_world: Pose, timestamp,
        camera_model: CameraModel,
    ):
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must increase strictly")
        if self._artifact_builder is None:
            self._artifact_builder = DynamicFrameArtifactBuilderV1(
                camera_model, self.config.depth_stride
            )
        artifacts = self._artifact_builder.build(
            depth, camera_pose_world, timestamp, self._frame_index
        )
        self.last_artifacts = artifacts
        frame = artifacts.depth_frame
        points_world = artifacts.points_world
        free_mask = self.foreground.previously_free_mask(
            points_world, timestamp
        )
        free_seed = np.zeros(frame.depth_m.shape, bool)
        free_pixels = frame.pixels_uv[free_mask]
        free_seed[free_pixels[:, 1], free_pixels[:, 0]] = True
        observations, labels = self.range_foreground.extract(
            frame, free_seed
        )
        foreground_mask = labels >= 0
        background_points = points_world[~foreground_mask][::8]
        self.foreground.extract(
            background_points, timestamp,
            camera_pose_world.position_world,
        )
        identified = []
        for row in observations:
            identified.append(replace(
                row, observation_id=self._next_observation_id,
                frame_index=self._frame_index,
            ))
            self._next_observation_id += 1
        observations = tuple(identified)
        all_tracks = self.track_manager.update(
            observations, timestamp, depth_frame=frame
        )
        confirmed = tuple(
            track for track in all_tracks if track.is_confirmed
        )
        dynamic = tuple(
            track for track in confirmed
            if track.is_dynamic and track.attention_authorized
        )
        attention, projected = build_dynamic_attention_with_projection(
            dynamic, camera_pose_world, camera_model,
            self.feature_shape, self.config, self.attention_device,
        )
        foreground_diagnostics = dict(
            self.range_foreground.last_diagnostics
        )
        foreground_diagnostics[
            "free_space_source"
        ] = self.foreground.last_diagnostics
        diagnostics = {
            "coordinate_convention":
                "camera optical +X right, +Y down, +Z forward; tracking in world",
            "clustering_algorithm":
                "standard_dbscan_sklearn_eps_min_samples",
            "input_point_count": len(frame.points_camera),
            "valid_point_count": len(frame.points_camera),
            "foreground_point_count": int(foreground_mask.sum()),
            "foreground": foreground_diagnostics,
            "cluster_count": len(observations),
            "noise_point_count": int(np.sum(labels == -1)),
            "track_manager": self.track_manager.last_diagnostics,
            "batch_semantics":
                "one DynamicPerception instance per temporal stream",
            "input_kind": "depth", "raw_depth_shape": list(frame.depth_m.shape),
            "raw_depth_dtype": str(frame.depth_m.dtype),
            "sampled_pixel_count": len(frame.pixels_uv),
            "pixel_point_identity_preserved": True,
            "fast_path_version": FAST_PATH_VERSION,
            "runtime_gt_used": False,
        }
        self._last_timestamp = timestamp
        self._frame_index += 1
        return DynamicPerceptionResult(
            all_tracks=all_tracks, confirmed_tracks=confirmed,
            dynamic_tracks=dynamic, projected_dynamic_tracks=projected,
            attention_map=attention, observations=observations,
            diagnostics=diagnostics,
        )


__all__ = [
    "FAST_PATH_VERSION", "FastTemporalVoxelForegroundV1",
    "CachedTemporalVoxelForegroundV1",
    "DynamicPerceptionFastPathV1",
]
