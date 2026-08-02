"""Development-only, reversible host-runtime qualification contract."""

from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "runtime_environment_contract_v1"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentContractV1:
    candidate_id: str
    process_affinity: tuple[int, ...]
    numerical_library_threads: int | None
    torch_intraop_threads: int | None
    torch_interop_threads: int | None
    gc_policy: str
    memory_pre_touch_bytes: int
    warmup_frames_per_episode: int
    logging_mode: str
    qualification_mode: bool
    governor_requirement: str
    restore_procedure: str

    def __post_init__(self):
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.process_affinity:
            raise ValueError("process affinity cannot be empty")
        if any(int(value) < 0 for value in self.process_affinity):
            raise ValueError("CPU IDs must be non-negative")
        if len(set(self.process_affinity)) != len(self.process_affinity):
            raise ValueError("CPU affinity cannot contain duplicates")
        for value in (
            self.numerical_library_threads,
            self.torch_intraop_threads,
            self.torch_interop_threads,
        ):
            if value is not None and int(value) < 1:
                raise ValueError("thread limits must be positive")
        if self.gc_policy not in {
            "DEFAULT", "EPISODE_BOUNDARY", "PROFILE_ONLY",
        }:
            raise ValueError("unsupported GC policy")
        if self.memory_pre_touch_bytes < 0:
            raise ValueError("memory pre-touch cannot be negative")
        if self.warmup_frames_per_episode != 10:
            raise ValueError("frozen warm-up is 10 frames")
        if self.logging_mode not in {
            "LOW_OVERHEAD_QUALIFICATION", "PROFILE",
        }:
            raise ValueError("unsupported logging mode")
        if self.governor_requirement not in {
            "READ_ONLY_CURRENT", "PERFORMANCE_IF_REVERSIBLE",
        }:
            raise ValueError("unsafe governor requirement")


__all__ = ["CONTRACT_VERSION", "RuntimeEnvironmentContractV1"]
