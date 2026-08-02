"""Differentiable local surrogate backed by exact authority correspondence.

Nearest-voxel correspondence is selected outside autograd from the canonical
occupancy BVH.  Distance to that fixed voxel AABB is differentiable.  This is
only a training surrogate; ``SafetyEvaluatorV2_1`` remains the final verifier.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch

from geometry_authority.static_v1 import DEFAULT_UAV_RADIUS_M


SURROGATE_VERSION = "authoritative_local_aabb_surrogate_v2_1"


def nearest_voxel_bounds(backend, positions):
    """Return deterministic nearest occupied AABB bounds for each xyz point."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("positions must be finite [N,3]")
    if backend.root < 0:
        raise ValueError("local AABB surrogate requires non-empty occupancy")
    minimum_bounds = np.empty_like(points)
    maximum_bounds = np.empty_like(points)
    indices = np.empty((len(points), 3), dtype=np.int64)
    for row, point in enumerate(points):
        best = math.inf
        selected = -1
        queue = [(
            backend._distance(
                point,
                backend.nodes[backend.root].minimum,
                backend.nodes[backend.root].maximum,
            ),
            backend.root,
        )]
        while queue:
            lower, node_index = heapq.heappop(queue)
            if lower > best + 1e-12:
                break
            node = backend.nodes[node_index]
            if node.indices is not None:
                delta = np.maximum(np.maximum(
                    backend.minimum[node.indices] - point,
                    point - backend.maximum[node.indices],
                ), 0.0)
                distances = np.linalg.norm(delta, axis=1)
                local = float(distances.min())
                candidates = node.indices[distances <= local + 1e-12]
                candidate = int(candidates[
                    np.argmin(backend.linear[candidates])
                ])
                if (
                    local < best - 1e-12
                    or (
                        abs(local-best) <= 1e-12
                        and (
                            selected < 0
                            or backend.linear[candidate] < backend.linear[selected]
                        )
                    )
                ):
                    selected = candidate
                best = min(best, local)
            else:
                for child in (node.left, node.right):
                    child_node = backend.nodes[child]
                    lower = backend._distance(
                        point, child_node.minimum, child_node.maximum
                    )
                    if lower <= best + 1e-12:
                        heapq.heappush(queue, (lower, child))
        if selected < 0:
            raise RuntimeError("nearest occupied voxel was not found")
        minimum_bounds[row] = backend.minimum[selected]
        maximum_bounds[row] = backend.maximum[selected]
        indices[row] = backend.occupied[selected]
    return minimum_bounds, maximum_bounds, indices


def local_aabb_signed_gap(
    positions,
    voxel_minimum,
    voxel_maximum,
    *,
    uav_radius_m=DEFAULT_UAV_RADIUS_M,
    stable_epsilon=1e-12,
):
    """Differentiable point-to-selected-AABB sphere gap."""
    if positions.shape != voxel_minimum.shape or positions.shape != voxel_maximum.shape:
        raise ValueError("positions and voxel bounds must have identical shapes")
    if positions.shape[-1] != 3:
        raise ValueError("surrogate geometry must use xyz")
    radius = float(uav_radius_m)
    if abs(radius-DEFAULT_UAV_RADIUS_M) > 1e-12:
        raise ValueError("authoritative UAV radius is frozen at 0.3 m")
    epsilon = positions.new_tensor(float(stable_epsilon))
    delta = torch.maximum(
        torch.maximum(voxel_minimum-positions, positions-voxel_maximum),
        torch.zeros_like(positions),
    )
    distance = (delta.square().sum(-1) + epsilon.square()).sqrt() - epsilon
    return distance-radius

