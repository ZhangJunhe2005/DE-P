"""Exact canonical-occupancy sight-line geometry for natural occlusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


GEOMETRY_VERSION = "authority_sightline_occlusion_geometry_v1"


@dataclass(frozen=True)
class ExposedFace:
    voxel_index: tuple[int, int, int]
    axis: int
    sign: int
    center_world: tuple[float, float, float]
    normal_world: tuple[float, float, float]


def ray_aabb_interval(origin, direction, minimum, maximum, max_distance=np.inf):
    """Return the exact forward ray interval through an axis-aligned box."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    parallel = np.abs(direction) < 1e-15
    if np.any(parallel & ((origin < minimum) | (origin > maximum))):
        return None
    safe = np.where(parallel, 1.0, direction)
    first = (minimum-origin)/safe
    second = (maximum-origin)/safe
    first[parallel], second[parallel] = -np.inf, np.inf
    entry = max(0.0, float(np.maximum.reduce(np.minimum(first, second))))
    exit_ = min(float(max_distance), float(
        np.minimum.reduce(np.maximum(first, second))))
    return (entry, exit_) if exit_ >= entry else None


def segment_aabb_interval(start, end, minimum, maximum):
    direction = np.asarray(end, dtype=np.float64)-np.asarray(
        start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length == 0:
        return None
    value = ray_aabb_interval(
        start, direction/length, minimum, maximum, length)
    return value


def exposed_faces(backend):
    occupied = np.asarray(backend.occupied, dtype=np.int64)
    occupied_set = {tuple(row) for row in occupied.tolist()}
    resolution = float(backend.map.resolution)
    origin = np.asarray(backend.map.origin, dtype=np.float64)
    result = []
    for voxel in occupied:
        for axis in range(3):
            for sign in (-1, 1):
                neighbour = voxel.copy()
                neighbour[axis] += sign
                if tuple(neighbour) in occupied_set:
                    continue
                center = origin + (voxel.astype(np.float64)+0.5)*resolution
                center[axis] += sign*0.5*resolution
                normal = np.zeros(3, dtype=np.float64)
                normal[axis] = sign
                result.append(ExposedFace(
                    tuple(int(value) for value in voxel),
                    axis, sign, tuple(center), tuple(normal)))
    return result


def exact_first_hit(backend, camera, point, exclude_endpoint=False):
    """Return the nearest occupied voxel AABB hit on camera→point."""
    camera = np.asarray(camera, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    direction = point-camera
    length = float(np.linalg.norm(direction))
    if length == 0:
        return None
    unit = direction/length
    selected = None
    upper = length-1e-10 if exclude_endpoint else length
    for row, minimum, maximum, linear in zip(
        backend.occupied, backend.minimum, backend.maximum, backend.linear
    ):
        interval = ray_aabb_interval(
            camera, unit, minimum, maximum, upper)
        if interval is None:
            continue
        candidate = (interval[0], int(linear), row, interval)
        if selected is None or candidate[:2] < selected[:2]:
            selected = candidate
    if selected is None:
        return None
    return {
        "distance_m": float(selected[0]),
        "voxel_index": selected[2].astype(int).tolist(),
        "interval_m": [float(value) for value in selected[3]],
        "authority_hash": backend.map.metadata["artifact_manifest_hash"],
        "geometry_version": GEOMETRY_VERSION,
    }


def actor_behind_occluder(backend, camera, actor_center, actor_radius):
    hit = exact_first_hit(backend, camera, actor_center)
    if hit is None:
        return {"behind": False, "first_hit": None}
    center_distance = float(np.linalg.norm(
        np.asarray(actor_center)-np.asarray(camera)))
    return {
        "behind": hit["distance_m"] < center_distance-float(actor_radius),
        "first_hit": hit,
    }


def actor_static_collision(backend, center, radius, margin=0.0):
    physical = backend.query_one(center, float(radius))
    clearance = backend.query_one(center, float(radius)+float(margin))
    return {
        "physical_static_collision": bool(physical["collision"]),
        "static_clearance_margin_satisfied": not bool(clearance["collision"]),
        "physical_query": physical,
        "margin_query": clearance,
    }


def occluder_evidence_hash(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()
