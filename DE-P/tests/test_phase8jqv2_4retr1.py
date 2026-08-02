"""RETR1 standard-library unittest gate (84 contract cases)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest

from policy.dynamic.runtime_environment_contract_v1 import (
    RuntimeEnvironmentContractV1,
)
from policy.dynamic.runtime_qualification_telemetry_v1 import (
    ProcessTelemetryV1,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PREFIX = "phase8jqv2_4retr1_"
REQUIRED = (
    "entry_gate", "frozen_artifacts", "semantic_freeze",
    "host_environment", "cpu_topology", "cpu_frequency_thermal",
    "background_load", "thread_pool", "memory_page_fault",
    "h0_multirun_baseline", "deadline_manifest",
    "outlier_reproducibility", "dynamic_perception_tail",
    "h1_clean_host", "h2_cpu_affinity", "h3_thread_pool",
    "h4_memory_gc", "h5_combined", "candidate_comparison",
    "profile_vs_qualification", "multirun_qualification",
    "deadline_and_backlog", "frequency_sensitivity",
    "runtime_environment_contract", "semantic_equivalence",
    "frame_output_equivalence", "metric_equivalence",
    "implementation_contract", "determinism", "regression",
    "compatibility_matrix", "policy_convergence_handoff",
    "terminal_runtime_decision", "final_result",
)


def report(name):
    return json.loads((REPORTS/f"{PREFIX}{name}.json").read_text())


class RETR1Gate(unittest.TestCase):
    def _case(self, index):
        entry = report("entry_gate")
        freeze = report("semantic_freeze")
        final = report("final_result")
        h0 = report("h0_multirun_baseline")
        h5 = report("h5_combined")
        qualification = report("multirun_qualification")
        deadline = report("deadline_and_backlog")
        regression = report("regression")
        equivalence = report("semantic_equivalence")
        route_a = final["route"] == "A"
        terminal_fail = final["route"] in {"B", "C", "D"}
        checks = {
            1: entry["checks"]["perto1_route_b"],
            2: entry["checks"]["semantic_equivalence"],
            3: entry["checks"]["authorizer_not_bottleneck"],
            4: entry["checks"]["dynamic_perception_primary"],
            5: len(json.loads((
                REPORTS/"phase8jqv2_4perto1_deadline_miss_manifest.json"
            ).read_text())["frames"]) == 5,
            6: any(
                row["authorizer_call_count"] == 0
                for row in json.loads((
                    REPORTS/"phase8jqv2_4perto1_deadline_miss_manifest.json"
                ).read_text())["frames"]
                if row["total_frame_time_ms"] >= 50.
            ),
            7: freeze["no_target_provisional_birth"] == 7,
            8: freeze["static_false_dynamic_birth"] == 7,
            9: freeze["true_support_recall"] == .5692307692307692,
            10: freeze["history_support_recall"] == 1.,
            11: freeze["no_history_support_recall"] == .7307692307692307,
            12: not report("frozen_artifacts")[
                "pdscr1_artifacts_modified"],
            13: not report("frozen_artifacts")[
                "dmcr1_artifacts_modified"],
            14: not report("frozen_artifacts")[
                "mar1_artifacts_modified"],
            15: not report("frozen_artifacts")[
                "diro1_artifacts_modified"],
            16: h0["run_count"] == 5,
            17: h0["steady_frame_count"] >= 1500,
            18: report("host_environment")["status"] == "PASS",
            19: report("cpu_topology")["status"] == "PASS",
            20: not report("cpu_topology")["illegal_core_id_present"],
            21: set(report("h2_cpu_affinity")["affinity_effective"]).issubset(
                report("cpu_topology")["allowed_cpu_set"]
            ),
            22: report("thread_pool")["status"] == "PASS_AUDITED",
            23: all(
                "voluntary_context_switches" in row
                for row in report("memory_page_fault")["h5"]
            ),
            24: all(
                "major_faults" in row
                for row in report("memory_page_fault")["h5"]
            ),
            25: report("cpu_frequency_thermal")[
                "status"] == "PASS_READ_ONLY",
            26: report("background_load")[
                "status"] == "PASS_CLEAN_CONTROL_AVAILABLE",
            27: report("profile_vs_qualification")[
                "profile_frames_used_for_gate"] == 0,
            28: not report("profile_vs_qualification")[
                "tracemalloc_in_qualification"],
            29: h0["no_outlier_deleted"] and h5["no_outlier_deleted"],
            30: report("outlier_reproducibility")[
                "status"] == "PASS_ANALYZED",
            31: set(report("dynamic_perception_tail")[
                "qualification_dynamic_perception_ms"]) == {
                    "ordinary", "near_deadline", "extreme",
                },
            32: not report("frozen_artifacts")[
                "evidence_authorizer_semantics_modified"],
            33: equivalence["all_candidates_equal"],
            34: h0["all_no_skipped_frame"] and h5["all_no_skipped_frame"],
            35: all(
                row["summary"]["steady_frame_count"] == 300
                for row in self._runs()
            ),
            36: not report("frozen_artifacts")[
                "foreground_semantics_modified"],
            37: equivalence["foreground_exact"],
            38: equivalence["component_exact"],
            39: equivalence["measurement_exact"],
            40: equivalence["provisional_exact"],
            41: equivalence["planner_semantic_exact"],
            42: report("h1_clean_host")["status"] == "PASS_PROFILED",
            43: report("h2_cpu_affinity")["status"] == "PASS_PROFILED",
            44: report("h3_thread_pool")["status"] == "PASS_PROFILED",
            45: report("h4_memory_gc")["status"] == "PASS_PROFILED",
            46: report("h5_combined")["status"] == "PASS_MEASURED",
            47: report("candidate_comparison")["selected"]
                == "H5_MINIMAL_ENVIRONMENT_CONTRACT",
            48: qualification["run_count"] == 5,
            49: qualification["steady_frame_count"] >= 1500,
            50: (
                qualification["cycle_ms"]["p95"]
                <= deadline["gate_ms"]
            ) if route_a else terminal_fail,
            51: (
                qualification["cycle_ms"]["p99"]
                <= deadline["gate_ms"]
            ) if route_a else terminal_fail,
            52: (
                qualification["deadline_miss_rate"] <= .01
            ) if route_a else terminal_fail,
            53: (
                qualification["consecutive_deadline_miss_max"] <= 1
            ) if route_a else terminal_fail,
            54: deadline["queue_depth"] == 0,
            55: not deadline["unbounded_backlog"],
            56: report("memory_page_fault")["memory_bounded"],
            57: deadline["same_frame_only"],
            58: deadline["atomic_join_before_snapshot"],
            59: report("frequency_sensitivity")[
                "same_runtime_samples"],
            60: final["route"] in {"A", "B", "C", "D", "E"},
            61: final["next_allowed_phase"]
                != "phase8jqv2_4_runtime_environment_and_tail_review",
            62: not final["second_runtime_tail_review_authorized"],
            63: not freeze["runtime_gt_used"],
            64: freeze["formal_tracker_feed"] == 0,
            65: not report("runtime_environment_contract")[
                "production_default_changed"],
            66: not final["production_activation_authorized"],
            67: not final["training_authorized"],
            68: regression["PERTO1"] == "FROZEN",
            69: regression["PECR1"] == "FROZEN",
            70: regression["PDSCR1_DMCR1"] == "FROZEN",
            71: regression["MAR1_DIRO1"] == "FROZEN",
            72: regression["CLDSR1_BRIR1"] == "FROZEN",
            73: regression["BDRR1_SAMSR1"] == "FROZEN",
            74: regression["DOGMR1_PTAR1"] == "FROZEN",
            75: regression["KUCR1_OCSR1"] == "FROZEN",
            76: regression["TCCR1_SOCR1_EOSR1"] == "FROZEN",
            77: report("implementation_contract")["status"] == "PASS",
            78: self._implementation_hashes_current(),
            79: self._reports_complete(),
            80: final["runtime_tail_review_iteration_count"] == 1,
            81: report("candidate_comparison")[
                "unplanned_experiment_count"] == 0,
            82: self._contract_valid(),
            83: isinstance(ProcessTelemetryV1.capture(), ProcessTelemetryV1),
            84: report("policy_convergence_handoff")[
                "threshold_repair_forbidden"],
        }
        self.assertTrue(checks[index])

    @staticmethod
    def _runs():
        return [
            json.loads(path.read_text())
            for path in sorted((
                ROOT/"diagnostics/phase8jqv2_4retr1/runs"
            ).glob("h[05]_[0-9][0-9].json"))
        ]

    @staticmethod
    def _implementation_hashes_current():
        value = report("implementation_contract")
        return value["files"] == {
            path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in value["files"]
        }

    @staticmethod
    def _reports_complete():
        return (
            all((REPORTS/f"{PREFIX}{name}.json").exists() for name in REQUIRED)
            and all((REPORTS/f"{PREFIX}{name}.md").exists() for name in (
                "final_recommendation", "final_readiness",
            ))
        )

    @staticmethod
    def _contract_valid():
        allowed = tuple(sorted(os.sched_getaffinity(0)))
        value = RuntimeEnvironmentContractV1(
            "TEST", allowed, 1, 1, 1, "DEFAULT", 0, 10,
            "LOW_OVERHEAD_QUALIFICATION", True, "READ_ONLY_CURRENT",
            "process exit",
        )
        return value.qualification_mode and value.process_affinity == allowed


def _install():
    for index in range(1, 85):
        def test(self, value=index):
            self._case(value)
        test.__name__ = f"test_{index:02d}"
        setattr(RETR1Gate, test.__name__, test)


_install()
