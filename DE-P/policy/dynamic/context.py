"""Serializable boundary between temporal perception and the DEP network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch


ATTENTION_BACKBONE_OUTPUT = "backbone_output"


def _plain(value):
    """Convert diagnostics to JSON-compatible values without retaining ROS objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class DynamicTrackSummary:
    track_id: int
    position_world: Tuple[float, float, float]
    velocity_world: Tuple[float, float, float]
    is_dynamic: bool
    timestamp: float

    @classmethod
    def from_track(cls, track):
        return cls(
            track_id=int(track.track_id),
            position_world=tuple(float(x) for x in track.position_world),
            velocity_world=tuple(float(x) for x in track.velocity_world),
            is_dynamic=bool(track.is_dynamic),
            timestamp=float(track.timestamp),
        )


@dataclass(frozen=True)
class DynamicContext:
    """Per-sample dynamic input; tensors are the only fields read by the network.

    Every attention tensor is ``[B,1,H,W]``. A context represents exactly that
    batch, while each temporal stream must still own a separate tracker upstream.
    """

    attention_maps_by_level: Mapping[str, torch.Tensor] = field(default_factory=dict)
    dynamic_tracks: Tuple[DynamicTrackSummary, ...] = ()
    timestamp: Optional[float] = None
    source: str = "none"
    valid: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in {"none", "depth", "pointcloud", "test"}:
            raise ValueError(f"invalid DynamicContext source: {self.source!r}")
        if self.timestamp is not None and not np.isfinite(self.timestamp):
            raise ValueError("DynamicContext timestamp must be finite or None")
        maps = dict(self.attention_maps_by_level)
        batch_size = None
        for level, attention in maps.items():
            if not isinstance(level, str) or not level:
                raise ValueError("attention level names must be non-empty strings")
            if not torch.is_tensor(attention):
                raise TypeError("DynamicContext attention maps must be Torch tensors")
            if attention.ndim != 4 or attention.shape[1] != 1:
                raise ValueError(
                    f"attention map {level!r} must have shape [B,1,H,W], got {tuple(attention.shape)}"
                )
            if batch_size is None:
                batch_size = attention.shape[0]
            elif attention.shape[0] != batch_size:
                raise ValueError("all DynamicContext attention maps must use the same batch size")
            if not bool(torch.isfinite(attention).all()):
                raise FloatingPointError("DynamicContext attention contains NaN/Inf")
            if bool((attention < 0).any()) or bool((attention > 1).any()):
                raise ValueError("DynamicContext attention must be bounded in [0,1]")
        object.__setattr__(self, "attention_maps_by_level", maps)
        object.__setattr__(self, "dynamic_tracks", tuple(self.dynamic_tracks))
        object.__setattr__(self, "diagnostics", _plain(self.diagnostics))

    @property
    def batch_size(self):
        if not self.attention_maps_by_level:
            return None
        return next(iter(self.attention_maps_by_level.values())).shape[0]

    def attention(self, level=ATTENTION_BACKBONE_OUTPUT):
        return self.attention_maps_by_level.get(level)

    def to(self, device):
        """Move tensor inputs without changing per-sample context semantics."""
        return DynamicContext(
            attention_maps_by_level={
                level: value.to(device)
                for level, value in self.attention_maps_by_level.items()
            },
            dynamic_tracks=self.dynamic_tracks,
            timestamp=self.timestamp,
            source=self.source,
            valid=self.valid,
            diagnostics=self.diagnostics,
        )

    def to_serializable(self):
        return {
            "attention_shapes_by_level": {
                key: list(value.shape) for key, value in self.attention_maps_by_level.items()
            },
            "dynamic_tracks": [_plain(track.__dict__) for track in self.dynamic_tracks],
            "timestamp": self.timestamp,
            "source": self.source,
            "valid": self.valid,
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_perception_result(cls, result, timestamp, source):
        return cls(
            attention_maps_by_level={ATTENTION_BACKBONE_OUTPUT: result.attention_map},
            dynamic_tracks=tuple(DynamicTrackSummary.from_track(track) for track in result.dynamic_tracks),
            timestamp=float(timestamp),
            source=source,
            valid=True,
            diagnostics=result.diagnostics,
        )

    @classmethod
    def invalid(cls, source, timestamp=None, reason="unavailable"):
        return cls(
            timestamp=timestamp,
            source=source,
            valid=False,
            diagnostics={"fallback_reason": reason},
        )
