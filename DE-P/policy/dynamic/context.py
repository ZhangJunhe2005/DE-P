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
    last_direct_observation_timestamp: float | None = None
    prediction_only_age: int = 0
    confidence: float = 1.0
    state_covariance: Tuple[Tuple[float, ...], ...] = ()
    observed_extent: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_track(cls, track):
        # ``DynamicTrackSummary`` is the collision-shield view, not the
        # attention-authorisation view.  A confirmed moving object remains a
        # dynamic safety risk during a bounded prediction-only occlusion even
        # when confidence decay temporarily clears ``track.is_dynamic``.
        safety_dynamic = bool(
            track.is_dynamic
            or getattr(track, "ever_confirmed_dynamic", False)
        )
        return cls(
            track_id=int(track.track_id),
            position_world=tuple(float(x) for x in track.position_world),
            velocity_world=tuple(float(x) for x in track.velocity_world),
            is_dynamic=safety_dynamic,
            timestamp=float(track.timestamp),
            last_direct_observation_timestamp=(
                None if getattr(
                    track, "last_direct_observation_timestamp", None
                ) is None else float(track.last_direct_observation_timestamp)
            ),
            prediction_only_age=int(
                getattr(track, "prediction_only_age", 0)
            ),
            confidence=float(getattr(track, "confidence", 1.0)),
            state_covariance=tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(
                    getattr(track, "state_covariance", np.empty((0, 0))),
                    dtype=np.float64,
                )
            ),
            observed_extent=tuple(float(value) for value in getattr(
                track, "last_observed_extent", (0.0, 0.0, 0.0)
            )),
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
        # Attention authorization deliberately remains stricter than the
        # deterministic collision shield.  The attention map in ``result`` is
        # still built only from attention-authorized tracks, while the safety
        # consumer receives every confirmed causal dynamic track.  Reusing the
        # attention gate here previously made the hard filter silently blind
        # whenever an otherwise confirmed track narrowly missed the neural
        # confidence/visibility policy.
        safety_tracks = tuple(
            track for track in result.confirmed_tracks
            if bool(
                track.is_dynamic
                or getattr(track, "ever_confirmed_dynamic", False)
            )
            and bool(getattr(track, "ever_directly_observed", True))
            and int(getattr(track, "prediction_only_age", 0)) <= 3
            and getattr(track, "visibility_state", "unknown")
                != "clear_missing"
        )
        return cls(
            attention_maps_by_level={ATTENTION_BACKBONE_OUTPUT: result.attention_map},
            dynamic_tracks=tuple(
                DynamicTrackSummary.from_track(track)
                for track in safety_tracks
            ),
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
