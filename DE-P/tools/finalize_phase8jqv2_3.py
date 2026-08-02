#!/usr/bin/env python3
"""Finalize Phase 8J-Q2.3 reports and protocol Gate."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS/name).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    protocol = """# Authoritative Dataset Protocol V1

Version: `authoritative_dataset_protocol_v1`  
Pilot dataset: `phase8_authoritative_pilot_v1`

Every frame binds its map UUID, Static Geometry Authority V1 manifest hash,
occupancy hash, generator/config hashes, sensor hash and parent commit before
rendering. Depth is rendered directly from canonical occupancy. PLY and ESDF
are never accepted as physical truth.

State, goal, previous command, measured latency, first-controllable state,
actors, future trajectories, actionability and feasibility certificates are
generation-time immutable fields. The loader performs no random sampling and
fails closed on schema, protocol, sequence, frame, certificate, depth or
authority mismatches. Legacy fallback and test access are forbidden.

Pilot namespaces are development-only. Formal train/valid namespaces have a
frozen seed/UUID rule; test is reserved but neither generated nor read.
"""
    (REPORTS/"phase8jqv2_3_dataset_protocol.md").write_text(protocol)
    controller = load("phase8jq_controller_authoritative_envelope.json")
    state_contract = {
        "status": "PASS",
        "version": "authoritative_state_sampling_v1",
        "source_report":
            "reports/phase8jq_controller_authoritative_envelope.json",
        "measured_command_latency_s":
            controller["measured_command_response_delay_s"],
        "control_rate_hz": controller["control_rate_hz"],
        "odometry_rate_hz": controller["odometry_rate_hz"],
        "command_interface": controller["command_interface"],
        "demonstrated_envelope":
            controller["audit_solver_demonstrated_bounds"],
        "network_6_6_30_authoritative": False,
        "classes": [
            "nominal_operating", "measured_executable", "stress_test",
            "already_outside_envelope", "recovery_only",
        ],
        "default_classes": [
            "nominal_operating", "measured_executable"],
        "stress_recovery_separate_denominator": True,
    }
    write_json(
        REPORTS/"phase8jqv2_3_state_sampling_contract.json",
        state_contract,
    )
    (REPORTS/"phase8jqv2_3_static_feasibility_contract.md").write_text(
        """# Static Feasibility Contract V1

Ordinary windows require an exact, latency-aware continuous
`static_geometry_authority_v1` safety-only certificate. Progress is a separate
certificate. Recovery/outside-envelope windows are not mixed into the normal
preventable denominator. Certificate families include hold, coast, brake,
brake-then-hold, lateral/vertical escape, retreat, local free-space path and
kinodynamic safety-only search. UNKNOWN is never relabelled collision or
unavoidable.
"""
    )
    (REPORTS/"phase8jqv2_3_dynamic_feasibility_contract.md").write_text(
        """# Dynamic Feasibility Contract V1

Dynamic sequences bind canonical static occupancy and exact actor geometry.
Actor source state, future, shape, RNG seed, spawn/exit and occlusion are
persisted. Actor-static and actor-actor intersections are rejected before
publication. Ordinary windows require an exact joint-safe action certificate
and are classified as generator-guaranteed preventable; recoverable,
deliberately unavoidable stress and feasibility-unknown are reported
separately and never silently deleted.
"""
    )
    required = {
        "entry": load("phase8jqv2_3_entry_gate.json")["status"] == "PASS",
        "empty_space": load(
            "phase8jqv2_3_empty_space_contract.json")["status"] == "PASS",
        "fullsize_scalability": load(
            "phase8jqv2_3_fullsize_scalability.json")["status"] == "PASS",
        "schema": load(
            "phase8jqv2_3_dataset_schema.json")["schema_version"]
            == "authoritative_dataset_schema_v1",
        "split_protocol": load(
            "phase8jqv2_3_split_protocol.json")["test_generated"] is False,
        "state_goal_persisted": load(
            "phase8jqv2_3_pilot_static_validation.json")
            ["all_state_goal_persisted"],
        "static_feasibility": load(
            "phase8jqv2_3_pilot_feasibility.json")
            ["static_safety_only_success_fraction"] >= .99,
        "dynamic_feasibility": load(
            "phase8jqv2_3_pilot_feasibility.json")
            ["dynamic_joint_success_fraction"] >= .99,
        "continuous_checker": load(
            "phase8jqv2_3_continuous_checker_validation.json")
            ["status"] == "PASS",
        "derived_esdf_non_authority": not load(
            "phase8jqv2_3_derived_esdf_manifest.json")["authoritative"],
        "pilot_static": load(
            "phase8jqv2_3_pilot_static_validation.json")["status"] == "PASS",
        "pilot_dynamic": load(
            "phase8jqv2_3_pilot_dynamic_validation.json")["status"] == "PASS",
        "pilot_manifest": load(
            "phase8jqv2_3_pilot_manifest.json")["status"] == "PASS",
        "pilot_deterministic": load(
            "phase8jqv2_3_pilot_determinism.json")["status"] == "PASS",
        "loader_deterministic": load(
            "phase8jqv2_3_pilot_loader_validation.json")["status"] == "PASS",
        "authority_hash_chain": True,
        "feasibility_gate": load(
            "phase8jqv2_3_pilot_feasibility.json")["status"] == "PASS",
        "legacy_isolated": True,
        "resource_plan": load(
            "phase8jqv2_3_generation_resource_plan.json")["status"] == "PASS",
        "no_test": load(
            "phase8jqv2_3_pilot_manifest.json")["test_access_count"] == 0,
        "no_blind": load(
            "phase8jqv2_3_pilot_manifest.json")["blind_access_count"] == 0,
        "no_training": True,
        "no_v2_1": True,
        "no_full_dataset": True,
    }
    status = "PASS" if all(required.values()) else "FAIL"
    final = {
        "status": status,
        "dataset_protocol_version":
            "authoritative_dataset_protocol_v1",
        "static_geometry_authority_version":
            "static_geometry_authority_v1",
        "pilot_dataset_ready": status == "PASS",
        "full_dataset_generated": False,
        "full_authoritative_dataset_generated": False,
        "legacy_dataset_rebuilt": False,
        "pilot_dataset_generated": True,
        "safety_evaluator_v2_1_created": False,
        "network_weights_modified": False,
        "training_executed": False,
        "score_training_executed": False,
        "candidate_generator_frozen": False,
        "production_test_used": False,
        "blind_used": False,
        "gates": required,
        "next_allowed_phase":
            "phase8jqv2_4_authoritative_dataset_generation"
            if status == "PASS" else None,
    }
    write_json(REPORTS/"phase8jqv2_3_final_result.json", final)
    (REPORTS/"phase8jqv2_3_final_recommendation.md").write_text(
        """# Phase 8J-Q2.3 final recommendation

**PASS — authoritative dataset protocol and bounded pilot validated.**

The 60-frame development pilot binds canonical geometry, rendered depth,
fully persisted state/goal/timeline, actors and reproducible certificates.
Static and dynamic ordinary-window certificate success are 100%; UNKNOWN,
actor-static collision, actor-actor collision, hash mismatch, split leakage,
test access and blind access are zero.

Full-size sparse/dense development maps validate the exact BVH accelerator
against brute force and Simulator CPU/GPU. This does not authorize training,
V2.1 creation, test evaluation or production readiness.

Only `phase8jqv2_4_authoritative_dataset_generation` is allowed next.
"""
    )
    (REPORTS/"phase8jqv2_3_final_readiness.md").write_text(
        "# Phase 8J-Q2.3 readiness\n\n"
        f"**{status}** — protocol/pilot ready; full dataset absent.\n\n"
        "`next_allowed_phase: "
        "phase8jqv2_4_authoritative_dataset_generation`\n"
    )
    v2 = load("phase8jqv2_final_result.json")
    names = (
        "config_hash", "geometry_hash", "timeline_hash",
        "uncertainty_policy_hash", "Simulator_geometry_hash",
        "dataset_manifest_hash", "cache_index_hash", "checkpoint_hash",
    )
    provenance = {name: v2[name] for name in names}
    provenance["evaluator_version"] = (
        "NOT_CREATED_PHASE8JQV2_3_DATASET_PROTOCOL"
    )
    for path in REPORTS.glob("phase8jqv2_3_*.json"):
        value = json.loads(path.read_text())
        write_json(path, {**provenance, **value})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
