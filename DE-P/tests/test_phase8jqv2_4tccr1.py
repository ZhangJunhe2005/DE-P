"""Phase 8J-Q2.4-TCCR1 frozen-boundary and evidence tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from controller.dynamic_safety_shadow_adapter_v1 import (
    ShadowSafetyConfig, contract_active, evaluate_shadow,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_01_socr1_entry_is_preserved():
    assert load("phase8jqv2_4tccr1_entry_gate.json")["socr1_status"] == "FAIL_EVIDENCE_INCOMPLETE"


def test_02_eosr_witnesses_frozen():
    assert load("phase8jqv2_4tccr1_entry_gate.json")["eosr1_witness_count"] == 6


def test_03_track_manager_hash_frozen():
    entry = load("phase8jqv2_4tccr1_entry_gate.json")
    assert entry["track_manager_hash_matches_socr1"]


def test_04_static_checkpoint_frozen():
    result = load("phase8jqv2_4tccr1_regression.json")
    assert result["static_checkpoint_hash_unchanged"]


def test_05_real_detector_ran():
    result = load("phase8jqv2_4tccr1_final_result.json")
    assert result["real_detector_tracker_telemetry"] == "PASS"


def test_06_real_track_manager_ran():
    assert load("phase8jqv2_4tccr1_final_result.json")["successful_tracks"] >= 30


def test_07_confidence_not_artificial():
    assert not load("phase8jqv2_4tccr1_confidence_component_contract.json")["confidence_artificially_set"]


def test_08_confidence_components_complete():
    assert load("phase8jqv2_4tccr1_confidence_component_contract.json")["telemetry_fields_complete"]


def test_09_velocity_uncertainty_recorded():
    assert load("phase8jqv2_4tccr1_velocity_uncertainty_analysis.json")["dynamic_velocity_uncertainty"]["sample_count"] > 0


def test_10_covariance_recorded():
    row = json.loads((ROOT / "diagnostics/phase8jqv2_4tccr1/runtime_telemetry.jsonl").read_text().splitlines()[0])
    assert len(row["covariance_diagonal"]) == 6


def test_11_dynamic_reason_recorded():
    row = json.loads((ROOT / "diagnostics/phase8jqv2_4tccr1/runtime_telemetry.jsonl").read_text().splitlines()[0])
    assert isinstance(row["dynamic_reason"], str)


def test_12_attention_reason_recorded():
    row = json.loads((ROOT / "diagnostics/phase8jqv2_4tccr1/runtime_telemetry.jsonl").read_text().splitlines()[0])
    assert isinstance(row["attention_reason"], str)


def check_state_semantic(key):
    assert load("phase8jqv2_4tccr1_state_semantics.json")[key]


def test_17_gap1_real_trace():
    assert load("phase8jqv2_4tccr1_gap1_state_results.json")["events"] >= 6


def test_18_gap2_real_trace():
    assert load("phase8jqv2_4tccr1_gap2_classification.json")["events"] > 0


def test_19_no_target_negative():
    report = load("phase8jqv2_4tccr1_real_pre_gap_confidence_distribution.json")
    assert report["negative_sequences"] >= 10


def test_20_static_fov_negative():
    report = load("phase8jqv2_4tccr1_real_pre_gap_confidence_distribution.json")
    assert report["negative_false_attention_frames"] == 0


def test_21_static_disocclusion_control_present():
    summary = load("../diagnostics/phase8jqv2_4tccr1/case_summary.json")
    names = {case["case_id"] for case in summary["cases"]}
    assert any("disocclusion" in name for name in names)


def test_22_ordinary_dynamic_present():
    report = load("phase8jqv2_4tccr1_real_pre_gap_confidence_distribution.json")
    assert report["category_counts"]["ordinary_dynamic"] >= 5


def test_23_multi_target_mode_present():
    modes = load("phase8jqv2_4tccr1_mode_comparison.json")["scenarios"]
    assert modes["multi_target"]["real_tracker_state"]


def test_24_checkpoint_strict_load():
    result = load("phase8jqv2_4tccr1_static_checkpoint_loading.json")
    assert result["strict_load"] and not result["missing_keys"] and not result["unexpected_keys"]


def test_25_static_yopo_inference():
    assert load("phase8jqv2_4tccr1_static_yopo_compatibility.json")["finite"]


def test_26_candidate_schema():
    assert load("phase8jqv2_4tccr1_static_yopo_compatibility.json")["candidate_count"] == 15


def test_27_shadow_adapter_runs():
    assert load("phase8jqv2_4tccr1_dynamic_safety_shadow_adapter.json")["status"] == "PASS"


def test_28_shadow_does_not_modify_command():
    candidates = np.zeros((1, 2, 3))
    result = evaluate_shadow(candidates, [0.1, 0.2], [])
    assert not result["formal_control_modified"]


def test_29_no_runtime_gt():
    candidates = np.zeros((1, 2, 3))
    assert not evaluate_shadow(candidates, [0.1, 0.2], [])["runtime_gt_used"]


def base_track(**updates):
    value = {
        "track_exists": True, "is_confirmed": True,
        "is_dynamic": False, "attention_authorized": False,
        "confidence": .4, "missed_count": 1,
        "previously_dynamic": True,
        "state_covariance": np.eye(6) * .01,
    }
    value.update(updates)
    return value


def test_30_coasting_cannot_birth():
    active, _ = contract_active(
        base_track(track_exists=False), "C2_recent_dynamic_coasting",
        ShadowSafetyConfig(),
    )
    assert not active


def test_31_coasting_requires_previous_dynamic():
    active, _ = contract_active(
        base_track(previously_dynamic=False), "C2_recent_dynamic_coasting",
        ShadowSafetyConfig(),
    )
    assert not active


def test_32_covariance_inflates_radius():
    track = base_track(
        position_world=np.array([0., 0., 0.]),
        velocity_world=np.zeros(3), track_id=1,
    )
    result = evaluate_shadow(np.zeros((1, 2, 3)), [.1, .2], [track])
    radii = [row["uncertainty_inflated_safety_radius_m"] for row in result["candidate_rows"]]
    assert radii[0] > .7


def test_33_missed_count_bound():
    active, _ = contract_active(
        base_track(missed_count=4), "C2_recent_dynamic_coasting",
        ShadowSafetyConfig(),
    )
    assert not active


def test_34_long_gap_terminates():
    report = load("phase8jqv2_4tccr1_coasting_safety_evaluation.json")
    assert report["long_gap_c2_is_bounded"]


def test_35_false_veto_separate():
    report = load("phase8jqv2_4tccr1_coasting_safety_evaluation.json")
    assert report["negative_shadow_veto_count"] == 0


def test_36_missed_veto_field_separate():
    report = load("phase8jqv2_4tccr1_mode_comparison.json")
    assert "limitation" in report


def check_mode(key):
    assert load("phase8jqv2_4tccr1_mode_comparison.json")[key]


def test_40_dynamic_dataset_conclusion():
    result = load("phase8jqv2_4tccr1_dynamic_dataset_requirement.json")
    assert result["classification"] == "not_required_for_engineering_baseline"


def check_no_forbidden_data(key):
    assert not load("phase8jqv2_4tccr1_final_result.json")[key]


def test_45_no_optimizer():
    assert not load("phase8jqv2_4tccr1_final_result.json")["optimizer_step_executed"]


def test_46_no_training():
    assert not load("phase8jqv2_4tccr1_final_result.json")["training_started"]


def test_47_legacy_default_unchanged():
    assert not load("phase8jqv2_4tccr1_regression.json")["legacy_default_changed"]


def test_48_static_shapes():
    shapes = load("phase8jqv2_4tccr1_static_checkpoint_loading.json")["output_shapes"]
    assert shapes == {"feature": [1, 64, 3, 5], "endstate": [1, 9, 3, 5], "score": [1, 3, 5]}


def test_49_route_b_selected():
    result = load("phase8jqv2_4tccr1_final_result.json")
    assert result["status"] == "PASS" and result["route"] == "B"


def test_50_reports_complete():
    required = [
        "entry_gate", "frozen_artifacts", "confidence_component_contract",
        "real_pre_gap_confidence_distribution", "confidence_reachability",
        "velocity_uncertainty_analysis", "state_semantics",
        "planner_track_consumption", "gap1_state_results",
        "gap2_classification", "static_yopo_compatibility",
        "static_checkpoint_loading", "dynamic_safety_shadow_adapter",
        "mode_comparison", "contract_candidates",
        "coasting_safety_evaluation", "dynamic_dataset_requirement",
        "engineering_baseline_readiness", "runtime", "determinism",
        "regression", "final_result",
    ]
    assert all((REPORTS / f"phase8jqv2_4tccr1_{name}.json").is_file() for name in required)


class TestPhase8JTCCR1(unittest.TestCase):
    """Methods are attached below to keep each contract item independent."""


def _wrap(function, *arguments):
    def method(self):
        function(*arguments)
    return method


for _name, _function in tuple(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(TestPhase8JTCCR1, _name, _wrap(_function))

for _number, _key in enumerate((
    "track_existence", "identity_continuity",
    "dynamic_classification", "safety_relevance",
), 13):
    setattr(
        TestPhase8JTCCR1, f"test_{_number:02d}_state_{_key}",
        _wrap(check_state_semantic, _key),
    )

for _number, _key in enumerate((
    "mode_a_present", "mode_b_present", "mode_c_present",
), 37):
    setattr(
        TestPhase8JTCCR1, f"test_{_number:02d}_{_key}",
        _wrap(check_mode, _key),
    )

for _number, _key in enumerate((
    "new_maps_generated", "formal_data_generated", "holdout_accessed",
    "production_test_accessed", "blind_accessed",
), 41):
    setattr(
        TestPhase8JTCCR1, f"test_{_number:02d}_forbidden_{_key}",
        _wrap(check_no_forbidden_data, _key),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
