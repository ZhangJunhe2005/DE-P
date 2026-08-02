"""Bounded runtime telemetry; formatting/serialization stays off-path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


TELEMETRY_VERSION = "runtime_telemetry_fast_v1"


@dataclass(frozen=True)
class RuntimeStageSampleV1:
    frame_id: int
    stage: str
    wall_ms: float
    input_points: int = 0
    component_count: int = 0
    accepted_component_count: int = 0


class BoundedRuntimeTelemetryV1:
    def __init__(self, capacity=4096, enabled=True):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.enabled = bool(enabled)
        self._samples = deque(maxlen=self.capacity)
        self.overflow_count = 0

    @property
    def samples(self):
        return tuple(self._samples)

    def measure(self, frame_id, stage, operation, **counts):
        if not self.enabled:
            return operation()
        started = time.perf_counter()
        value = operation()
        sample = RuntimeStageSampleV1(
            int(frame_id), str(stage),
            (time.perf_counter()-started)*1000., **counts,
        )
        if len(self._samples) == self.capacity:
            self.overflow_count += 1
        self._samples.append(sample)
        return value


__all__ = [
    "TELEMETRY_VERSION", "RuntimeStageSampleV1",
    "BoundedRuntimeTelemetryV1",
]
