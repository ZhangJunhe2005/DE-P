"""Exact BVH-accelerated sphere-to-occupied-voxel backend."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from geometry_authority.static_v1 import (
    BACKEND_VERSION, CONTACT_TOLERANCE_M, EMPTY_GAP_M,
    StaticAuthorityMap,
)


@dataclass
class _Node:
    minimum: np.ndarray
    maximum: np.ndarray
    left: int = -1
    right: int = -1
    indices: np.ndarray | None = None


class ExactAuthorityBVH:
    version = "static_authority_exact_bvh_v1"

    def __init__(self, root, leaf_size=16):
        self.map = StaticAuthorityMap(root)
        self.occupied = self.map.occupied_indices
        self.linear = (
            self.occupied[:, 0]
            * self.map.dimensions[1] * self.map.dimensions[2]
            + self.occupied[:, 1] * self.map.dimensions[2]
            + self.occupied[:, 2]
        ) if len(self.occupied) else np.empty(0, dtype=np.int64)
        self.minimum = (
            self.map.origin + self.occupied * self.map.resolution
        )
        self.maximum = self.minimum + self.map.resolution
        self.nodes = []
        self.root = self._build(np.arange(len(self.occupied)), leaf_size)

    def _build(self, indices, leaf_size):
        if not len(indices):
            return -1
        minimum = self.minimum[indices].min(axis=0)
        maximum = self.maximum[indices].max(axis=0)
        node_index = len(self.nodes)
        self.nodes.append(_Node(minimum, maximum))
        if len(indices) <= leaf_size:
            self.nodes[node_index].indices = indices
            return node_index
        centers = (self.minimum[indices] + self.maximum[indices]) * .5
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        order = indices[np.argsort(
            centers[:, axis], kind="mergesort"
        )]
        middle = len(order) // 2
        self.nodes[node_index].left = self._build(
            order[:middle], leaf_size
        )
        self.nodes[node_index].right = self._build(
            order[middle:], leaf_size
        )
        return node_index

    @staticmethod
    def _distance(point, minimum, maximum):
        delta = np.maximum(np.maximum(minimum-point, point-maximum), 0)
        return float(np.linalg.norm(delta))

    def query_one(self, center, radius=.3):
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
        oob = bool(np.any(
            center-radius < self.map.bounds_min-CONTACT_TOLERANCE_M
        ) or np.any(
            center+radius > self.map.bounds_max+CONTACT_TOLERANCE_M
        ))
        if self.root < 0:
            return {
                "collision": oob, "minimum_gap_m": EMPTY_GAP_M,
                "contacted_voxel_index": [-1, -1, -1],
                "out_of_bounds": oob, "queried_voxel_count": 0,
                "empty_space": True, "backend_version": self.version,
            }
        best_distance = EMPTY_GAP_M
        selected = -1
        queried = 0
        queue = [(self._distance(
            center, self.nodes[self.root].minimum,
            self.nodes[self.root].maximum), self.root)]
        while queue:
            lower, node_index = heapq.heappop(queue)
            if lower > best_distance + CONTACT_TOLERANCE_M:
                break
            node = self.nodes[node_index]
            if node.indices is not None:
                queried += len(node.indices)
                delta = np.maximum(np.maximum(
                    self.minimum[node.indices] - center,
                    center - self.maximum[node.indices]), 0)
                distances = np.linalg.norm(delta, axis=1)
                minimum = float(distances.min())
                candidates = node.indices[
                    distances <= minimum + CONTACT_TOLERANCE_M
                ]
                candidate = int(candidates[
                    np.argmin(self.linear[candidates])
                ])
                if (
                    minimum < best_distance-CONTACT_TOLERANCE_M
                    or (
                        abs(minimum-best_distance)
                        <= CONTACT_TOLERANCE_M
                        and (
                            selected < 0
                            or self.linear[candidate]
                            < self.linear[selected]
                        )
                    )
                ):
                    selected = candidate
                best_distance = min(best_distance, minimum)
            else:
                for child in (node.left, node.right):
                    value = self.nodes[child]
                    lower = self._distance(
                        center, value.minimum, value.maximum
                    )
                    if lower <= best_distance + CONTACT_TOLERANCE_M:
                        heapq.heappush(queue, (lower, child))
        gap = best_distance-radius
        collision = oob or gap <= CONTACT_TOLERANCE_M
        voxel = (
            self.occupied[selected].tolist()
            if collision and selected >= 0 else [-1, -1, -1]
        )
        return {
            "collision": bool(collision), "minimum_gap_m": gap,
            "contacted_voxel_index": voxel,
            "out_of_bounds": oob, "queried_voxel_count": queried,
            "empty_space": False, "backend_version": self.version,
            "map_authority_hash":
                self.map.metadata["artifact_manifest_hash"],
        }

