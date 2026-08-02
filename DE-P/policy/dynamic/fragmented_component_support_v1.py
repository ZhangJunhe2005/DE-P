"""Bounded, causal grouping metadata for safety-only component fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


CONTRACT_VERSION = "fragmented_component_support_v1"


@dataclass(frozen=True)
class FragmentedSafetySupportV1:
    frame_index: int
    source_component_ids: Tuple[int, ...]
    centroid_world: np.ndarray
    pixel_bbox: Tuple[int, int, int, int]
    depth_interval_m: Tuple[float, float]
    bounded_separation_m: float
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if not 1 <= len(self.source_component_ids) <= 4:
            raise ValueError("fragment group must be bounded")
        if len(set(self.source_component_ids)) != len(
            self.source_component_ids
        ):
            raise ValueError("duplicate component IDs")
        center = np.asarray(self.centroid_world, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("centroid must be finite")
        center = center.copy()
        center.setflags(write=False)
        object.__setattr__(self, "centroid_world", center)


def bbox_gap(left, right):
    lu0, lv0, lu1, lv1 = left
    ru0, rv0, ru1, rv1 = right
    du = max(0, max(lu0, ru0)-min(lu1, ru1)-1)
    dv = max(0, max(lv0, rv0)-min(lv1, rv1)-1)
    return max(du, dv)


def compatible_fragments(left, right, parameters):
    """No semantic/GT identity: geometry, image, and depth only."""
    distance = float(np.linalg.norm(
        np.asarray(left["centroid_world"])
        - np.asarray(right["centroid_world"])
    ))
    left_depth = left.get("depth_interval_m")
    right_depth = right.get("depth_interval_m")
    if left_depth is None or right_depth is None:
        depth_compatible = False
    else:
        depth_gap = max(
            0., max(left_depth[0], right_depth[0])
            - min(left_depth[1], right_depth[1])
        )
        depth_compatible = (
            depth_gap <= parameters["maximum_depth_gap_m"]
        )
    return bool(
        distance <= parameters["maximum_centroid_distance_m"]
        and bbox_gap(left["pixel_bbox"], right["pixel_bbox"])
        <= parameters["maximum_bbox_gap_pixels"]
        and depth_compatible
    )


__all__ = [
    "CONTRACT_VERSION", "FragmentedSafetySupportV1",
    "bbox_gap", "compatible_fragments",
]
