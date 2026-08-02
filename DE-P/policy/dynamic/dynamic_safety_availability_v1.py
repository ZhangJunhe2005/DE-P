"""Read-only development adapter for causal dynamic-safety availability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time


CONTRACT_VERSION = "dynamic_safety_availability_v1"


class ObservabilityClassV1(str, Enum):
    CAUSALLY_OBSERVABLE_DYNAMIC_RISK = "CAUSALLY_OBSERVABLE_DYNAMIC_RISK"
    PREVIOUSLY_OBSERVED_COASTABLE_RISK = "PREVIOUSLY_OBSERVED_COASTABLE_RISK"
    PROVISIONAL_OBSERVABLE_RISK = "PROVISIONAL_OBSERVABLE_RISK"
    CAUSALLY_UNOBSERVABLE_FUTURE_RISK = "CAUSALLY_UNOBSERVABLE_FUTURE_RISK"
    ODD_VIOLATION = "ODD_VIOLATION"


@dataclass(frozen=True)
class DynamicSafetyAvailabilityRecordV1:
    identity: str
    risk_source: str
    provenance: tuple[int, ...]
    state_age_s: float
    observability_class: ObservabilityClassV1
    payload: object
    runtime_gt_used: bool = False

    def __post_init__(self):
        if self.runtime_gt_used:
            raise ValueError("runtime GT is forbidden")
        if self.state_age_s < 0:
            raise ValueError("state age must be non-negative")


class DynamicSafetyAvailabilityV1:
    """Priority merge; it does not modify tracks, reachability, or routing."""

    priority = {
        "ACTIVE_FORMAL_REACHABILITY": 0,
        "FORMAL_TRACK_SAFETY_AVAILABILITY": 1,
        "PROVISIONAL_DIRECT": 2,
        "PROVISIONAL_SHORT_MEMORY": 3,
    }

    def merge(self, *groups):
        selected = {}
        suppressed = []
        for group in groups:
            for row in group:
                if not isinstance(row, DynamicSafetyAvailabilityRecordV1):
                    raise TypeError("availability records have wrong type")
                provenance_key = frozenset(row.provenance)
                key = provenance_key or frozenset((row.identity,))
                previous = selected.get(key)
                if (
                    previous is None
                    or self.priority[row.risk_source]
                    < self.priority[previous.risk_source]
                ):
                    if previous is not None:
                        suppressed.append(previous.identity)
                    selected[key] = row
                else:
                    suppressed.append(row.identity)
        rows = tuple(sorted(
            selected.values(),
            key=lambda row: (self.priority[row.risk_source], row.identity),
        ))
        return rows, tuple(sorted(suppressed))

    @staticmethod
    def no_active_semantics():
        return (
            "no currently consumable causal dynamic risk state; "
            "it does not assert absence of world dynamic hazards"
        )


class RuntimePlannerMeasurementV1:
    """Runtime-only profiler with explicit CUDA synchronization boundaries."""

    runtime_modules = (
        "depth_preprocessing", "dynamic_perception", "measurement",
        "formal_tracker", "provisional_availability",
        "geometry_reachability", "yopo_inference", "snapshot",
        "risk", "router",
    )
    forbidden_modules = (
        "gt_actor_read", "owner_map_read", "exact_gt_shape_risk",
        "collision_proxy", "report_serialization", "heavy_diagnostics",
    )

    def __init__(self, cuda_synchronize=None):
        self._sync = cuda_synchronize
        self._active = None
        self.samples = []

    def start(self, metadata=None):
        if self._active is not None:
            raise RuntimeError("runtime timer already active")
        if self._sync is not None:
            self._sync()
        self._active = {
            "started": time.perf_counter(),
            "metadata": dict(metadata or {}),
            "modules": {},
        }

    def measure(self, module, operation):
        if module not in self.runtime_modules:
            raise ValueError("offline/unknown module excluded from runtime timer")
        if self._active is None:
            raise RuntimeError("runtime timer is not active")
        if self._sync is not None and module == "yopo_inference":
            self._sync()
        started = time.perf_counter()
        result = operation()
        if self._sync is not None and module == "yopo_inference":
            self._sync()
        self._active["modules"][module] = (
            self._active["modules"].get(module, 0.)
            +(time.perf_counter()-started)*1000.
        )
        return result

    def stop(self):
        if self._active is None:
            raise RuntimeError("runtime timer is not active")
        if self._sync is not None:
            self._sync()
        sample = {
            **self._active["metadata"],
            "cpu_wall_ms": (
                time.perf_counter()-self._active["started"]
            )*1000.,
            "per_module_ms": dict(self._active["modules"]),
            "runtime_gt_used": False,
            "offline_evaluation_included": False,
        }
        self.samples.append(sample)
        self._active = None
        return sample


__all__ = [
    "CONTRACT_VERSION", "ObservabilityClassV1",
    "DynamicSafetyAvailabilityRecordV1", "DynamicSafetyAvailabilityV1",
    "RuntimePlannerMeasurementV1",
]
