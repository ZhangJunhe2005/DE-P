"""Canonical label-based component aggregation without full-image rescans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AGGREGATION_VERSION = "component_aggregation_fast_v1"


@dataclass(frozen=True)
class ComponentAggregateV1:
    label: int
    flat_indices: np.ndarray
    pixel_count: int
    u_min: int
    v_min: int
    u_max: int
    v_max: int
    depth_min: float
    depth_max: float
    depth_sum: float


def canonical_component_aggregates(labels, depth):
    labels = np.asarray(labels)
    depth = np.asarray(depth)
    if labels.shape != depth.shape or labels.ndim != 2:
        raise ValueError("labels/depth must share a 2-D shape")
    flat_labels = labels.reshape(-1)
    # Runtime point labels use -1 for background and zero-based accepted
    # component IDs.  Preserve component zero; scipy image labels are an
    # internal implementation detail and are never passed to this API.
    selected = np.flatnonzero(flat_labels >= 0)
    if not len(selected):
        return ()
    selected_labels = flat_labels[selected].astype(np.int64, copy=False)
    order = np.argsort(selected_labels, kind="stable")
    selected = selected[order]
    selected_labels = selected_labels[order]
    boundaries = np.r_[0, np.flatnonzero(
        selected_labels[1:] != selected_labels[:-1]
    )+1, len(selected)]
    width = depth.shape[1]
    result = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = selected[start:stop]
        values = depth.reshape(-1)[indices]
        v, u = np.divmod(indices, width)
        result.append(ComponentAggregateV1(
            label=int(selected_labels[start]),
            flat_indices=indices.copy(), pixel_count=len(indices),
            u_min=int(u.min()), v_min=int(v.min()),
            u_max=int(u.max()), v_max=int(v.max()),
            depth_min=float(values.min()), depth_max=float(values.max()),
            depth_sum=float(values.sum(dtype=np.float64)),
        ))
    # scipy.ndimage.label numbers by the first row-major pixel.  This is also
    # the frozen component/observation ordering.
    return tuple(sorted(result, key=lambda row: int(row.flat_indices[0])))


__all__ = [
    "AGGREGATION_VERSION", "ComponentAggregateV1",
    "canonical_component_aggregates",
]
