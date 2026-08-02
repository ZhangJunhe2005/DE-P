"""Low-overhead process telemetry for independent qualification runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
import resource


TELEMETRY_VERSION = "runtime_qualification_telemetry_v1"


@dataclass(frozen=True, slots=True)
class ProcessTelemetryV1:
    minor_faults: int
    major_faults: int
    voluntary_context_switches: int
    involuntary_context_switches: int
    peak_rss_bytes: int
    process_thread_count: int
    affinity: tuple[int, ...]

    @classmethod
    def capture(cls):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        status = {}
        try:
            with open("/proc/self/status") as stream:
                for line in stream:
                    key, _, value = line.partition(":")
                    status[key] = value.strip()
        except OSError:
            pass
        return cls(
            int(usage.ru_minflt), int(usage.ru_majflt),
            int(usage.ru_nvcsw), int(usage.ru_nivcsw),
            int(usage.ru_maxrss)*1024,
            int(status.get("Threads", "0")),
            tuple(sorted(os.sched_getaffinity(0))),
        )

    def delta(self, before):
        return {
            "minor_faults": self.minor_faults-before.minor_faults,
            "major_faults": self.major_faults-before.major_faults,
            "voluntary_context_switches":
                self.voluntary_context_switches
                - before.voluntary_context_switches,
            "involuntary_context_switches":
                self.involuntary_context_switches
                - before.involuntary_context_switches,
            "peak_rss_growth_bytes":
                max(0, self.peak_rss_bytes-before.peak_rss_bytes),
            "process_thread_count_end": self.process_thread_count,
            "affinity_end": list(self.affinity),
        }


__all__ = ["TELEMETRY_VERSION", "ProcessTelemetryV1"]
