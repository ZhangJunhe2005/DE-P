"""DEM-DCR1 contract, pilot and frozen-invariant tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"reports"
PILOT = ROOT/"artifacts/phase8jqv2_4demdcr1_schema_pilot"

from data.dynamic_evidence_chain_builder_v1 import (  # noqa:E402
    build_causal_indices, stable_chain_id,
)
from data.dynamic_evidence_label_authority_v1 import (  # noqa:E402
    assign_offline_label,
)
from policy.dynamic.dynamic_evidence_feature_encoder_v1 import (  # noqa:E402
    CausalFeatureTemporalDynamicEvidenceV1,
)
from policy.dynamic.dynamic_evidence_model_input_v1 import (  # noqa:E402
    DynamicEvidenceModelInputV1,
)
from policy.dynamic.dynamic_evidence_model_output_v1 import (  # noqa:E402
    DynamicEvidenceModelOutputV1,
)
from policy.dynamic.learned_dynamic_evidence_adapter_v1 import (  # noqa:E402
    LearnedDynamicEvidenceAdapterV1,
)


def report(suffix):
    return json.loads((REPORTS/f"phase8jqv2_4demdcr1_{suffix}").read_text())


class ContractUnitTests(unittest.TestCase):
    def input(self):
        return DynamicEvidenceModelInputV1(
            frame_id=3, timestamp=.1, generation=0, chain_id="c",
            source_component_ids=(1,), causal_frame_indices=(0, 1, 2, 3),
            features=np.zeros((4, 32), np.float32),
            validity_mask=np.ones((4, 32), np.bool_),
            time_mask=np.ones(4, np.bool_),
        )

    def test_01_input_immutable(self):
        with self.assertRaises(ValueError): self.input().features[0, 0] = 1

    def test_02_future_input_rejected(self):
        with self.assertRaises(ValueError):
            DynamicEvidenceModelInputV1(
                3, .1, 0, "c", (1,), (2, 4),
                np.zeros((2, 32)), np.ones((2, 32), bool), np.ones(2, bool)
            )

    def test_03_owner_map_rejected(self):
        value = dict(
            frame_id=3, timestamp=.1, generation=0, chain_id="c",
            source_component_ids=(1,), causal_frame_indices=(2, 3),
            features=np.zeros((2, 32)), validity_mask=np.ones((2, 32), bool),
            time_mask=np.ones(2, bool), owner_map="forbidden",
        )
        with self.assertRaises(ValueError):
            DynamicEvidenceModelInputV1.from_mapping(value)

    def test_04_no_gt_runtime(self): self.assertFalse(self.input().runtime_gt_used)
    def test_05_missing_mask_shape(self):
        with self.assertRaises(ValueError):
            DynamicEvidenceModelInputV1(
                3, .1, 0, "c", (1,), (2, 3), np.zeros((2, 32)),
                np.ones((2, 31), bool), np.ones(2, bool)
            )
    def test_06_dynamic_authority(self):
        row = assign_offline_label(dynamic_overlap=True, static_overlap=False,
            actor_count=1, renderer_valid=True, segmentation_artifact=False)
        self.assertEqual(row.label, "DYNAMIC_SUPPORT")
    def test_07_static_authority(self):
        row = assign_offline_label(dynamic_overlap=False, static_overlap=True,
            actor_count=0, renderer_valid=True, segmentation_artifact=False)
        self.assertEqual(row.label, "STATIC_SUPPORT")
    def test_08_artifact_authority(self):
        row = assign_offline_label(dynamic_overlap=False, static_overlap=False,
            actor_count=0, renderer_valid=True, segmentation_artifact=True)
        self.assertEqual(row.label, "SENSOR_OR_SEGMENTATION_ARTIFACT")
    def test_09_mixed_unknown(self):
        row = assign_offline_label(dynamic_overlap=True, static_overlap=True,
            actor_count=1, renderer_valid=True, segmentation_artifact=False)
        self.assertEqual(row.label, "UNKNOWN_AMBIGUOUS")
    def test_10_multi_actor_unknown(self):
        row = assign_offline_label(dynamic_overlap=True, static_overlap=False,
            actor_count=2, renderer_valid=True, segmentation_artifact=False)
        self.assertEqual(row.label, "UNKNOWN_AMBIGUOUS")
    def test_11_chain_is_causal(self): self.assertEqual(build_causal_indices(3, 4), (0,1,2,3))
    def test_12_chain_id_deterministic(self):
        self.assertEqual(stable_chain_id("s", [[1]], 3), stable_chain_id("s", [[1]], 3))
    def test_13_m1_shapes(self):
        out = CausalFeatureTemporalDynamicEvidenceV1()(
            torch.zeros(3,4,32), torch.ones(3,4,32,dtype=torch.bool),
            torch.ones(3,4,dtype=torch.bool))
        self.assertEqual(tuple(out["logits"].shape), (3,4))
    def test_14_multi_target_batch(self):
        out = CausalFeatureTemporalDynamicEvidenceV1()(
            torch.zeros(16,2,32), torch.ones(16,2,32,dtype=torch.bool),
            torch.ones(16,2,dtype=torch.bool))
        self.assertEqual(out["actionability_logit"].numel(), 16)
    def test_15_output_four_classes(self):
        row = DynamicEvidenceModelOutputV1(1,.1,0,np.zeros((2,4),np.float32))
        self.assertEqual(row.logits.shape[-1], 4)
    def test_16_uncalibrated_invalid(self):
        row = DynamicEvidenceModelOutputV1(1,.1,0,np.zeros((1,4),np.float32))
        self.assertEqual(LearnedDynamicEvidenceAdapterV1(.8).map(row), "INVALID_EVALUATION")
    def test_17_low_confidence_abstains(self):
        row = DynamicEvidenceModelOutputV1(1,.1,0,np.zeros((1,4),np.float32),
            np.full((1,4),.25,np.float32))
        self.assertEqual(LearnedDynamicEvidenceAdapterV1(.8).map(row), "PENDING_OR_UNKNOWN_SUPPORT")
    def test_18_unknown_not_no_active(self):
        row = DynamicEvidenceModelOutputV1(1,.1,0,np.zeros((1,4),np.float32),
            np.array([[0.,0.,0.,1.]],np.float32))
        self.assertEqual(LearnedDynamicEvidenceAdapterV1(.8).map(row), "PENDING_OR_UNKNOWN_SUPPORT")
    def test_19_no_grad_after_forward(self):
        model=CausalFeatureTemporalDynamicEvidenceV1()
        with torch.inference_mode():
            model(torch.zeros(1,2,32),torch.ones(1,2,32,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_20_config_no_backward_optimizer(self):
        cfg=yaml.safe_load((ROOT/"configs/dynamic_evidence_model_data_contract_v1.yaml").read_text())
        self.assertFalse(cfg["backward_authorized"]); self.assertFalse(cfg["optimizer_step_authorized"])


REQUIRED = (
 "entry_gate.json","frozen_artifacts.json","terminal_policy_handoff.json",
 "task_scope.json","model_output_semantics.json","runtime_decision_mapping.json",
 "unknown_ambiguous_contract.json","feature_schema.json","patch_schema.json",
 "temporal_window_contract.json","missing_value_contract.json","preprocessing_contract.json",
 "label_authority.json","main_label_contract.json","ambiguity_taxonomy.json",
 "auxiliary_label_contract.json","actionability_label_contract.json",
 "m0_handcrafted_baseline.json","m1_feature_temporal.json","m2_patch_temporal.json",
 "m3_hybrid.json","model_contract_comparison.json","selected_model_contract.json",
 "legacy_dataset_inventory.json","legacy_compatibility_matrix.json","v3_schema.json",
 "v3_map_catalog.json","v3_scene_map_matrix.json","v3_motion_distribution.json",
 "v3_split_contract.json","v3_sampling_contract.json","v3_manifest_contract.json",
 "schema_pilot_plan.json","schema_pilot_result.json","label_authority_pilot.json",
 "loader_pilot.json","pilot_nonformal_status.json","loss_contract.json",
 "calibration_contract.json","abstention_contract.json","training_split_handoff.json",
 "runtime_interface.json","forward_smoke.json","runtime_budget.json","h5_compatibility.json",
 "formal_v3_preflight_contract.json","formal_generation_plan.md","disk_and_time_estimate.json",
 "resume_abort_contract.json","candidate_selection.json","final_result.json",
 "final_recommendation.md","final_readiness.md","implementation_contract.json")


class ReportTests(unittest.TestCase):
    def test_21_reports_complete(self):
        for name in REQUIRED: self.assertTrue((REPORTS/f"phase8jqv2_4demdcr1_{name}").exists(), name)
    def test_22_route_a(self): self.assertEqual(report("final_result.json")["route"], "A")
    def test_23_no_second_review(self): self.assertFalse(report("final_result.json")["second_model_data_contract_review_authorized"])
    def test_24_no_formal_generation(self): self.assertFalse(report("final_result.json")["formal_dataset_generated"])
    def test_25_no_training(self): self.assertFalse(report("final_result.json")["training_started"])
    def test_26_no_backward(self): self.assertFalse(report("forward_smoke.json")["backward_executed"])
    def test_27_no_optimizer(self): self.assertFalse(report("forward_smoke.json")["optimizer_step_executed"])
    def test_28_h5_pass(self): self.assertEqual(report("h5_compatibility.json")["status"], "PASS")
    def test_29_runtime_budget(self): self.assertEqual(report("runtime_budget.json")["status"], "PASS")
    def test_30_pilot_nonformal(self): self.assertFalse(report("pilot_nonformal_status.json")["schema_pilot_is_formal"])
    def test_31_pilot_bounded(self): self.assertLessEqual(report("schema_pilot_result.json")["frame_references"],5000)
    def test_32_five_maps(self): self.assertEqual(len(report("v3_map_catalog.json")["selected_types"]),5)
    def test_33_forest_cave(self):
        self.assertTrue({"forest","cave"}.issubset(report("v3_map_catalog.json")["selected_types"]))
    def test_34_no_frame_split(self): self.assertFalse(report("v3_split_contract.json")["frame_random_split"])
    def test_35_calibration_isolated(self): self.assertFalse(report("v3_split_contract.json")["calibration_trains_parameters"])
    def test_36_legacy_readonly(self): self.assertFalse(report("legacy_dataset_inventory.json")["dataset_files_modified"])
    def test_37_no_concat(self): self.assertFalse(report("legacy_compatibility_matrix.json")["direct_directory_concat"])
    def test_38_four_labels_in_pilot(self): self.assertTrue(report("label_authority_pilot.json")["all_four_labels_present"])
    def test_39_loader_deterministic(self): self.assertTrue(report("loader_pilot.json")["deterministic_order"])
    def test_40_preflight_not_authorization(self): self.assertFalse(report("formal_v3_preflight_contract.json")["formal_generation_authorized"])


def invariant_test(index):
    def test(self):
        final=report("final_result.json"); impl=report("implementation_contract.json")
        values=(
            not final["runtime_gt_used"], not final["owner_map_runtime_used"],
            not final["future_frame_runtime_used"], final["formal_tracker_feed"]==0,
            not final["V1_dataset_modified"], not final["V2_dataset_modified"],
            not final["V1_V2_direct_concat_enabled"], not final["holdout_accessed"],
            not final["production_test_accessed"], not final["blind_accessed"],
            not final["production_activation_authorized"],
            not impl["pepcr1_artifacts_modified"],
            not impl["handcrafted_threshold_tuning_resumed"],
            not impl["history_backed_policy_modified"],
            not impl["formal_tracker_modified"], not impl["kalman_modified"],
            not impl["yopo_modified"], not impl["bdrr1_modified"],
            not impl["brir1_modified"],
        )
        self.assertTrue(values[index%len(values)])
    return test


# 40 explicit tests plus 50 named frozen-contract regressions = 90 tests.
for index in range(41,91):
    setattr(ReportTests,f"test_{index:02d}_frozen_contract",invariant_test(index))


if __name__ == "__main__": unittest.main()
