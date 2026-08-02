from __future__ import annotations

import json
from pathlib import Path
import unittest

from authoritative_dataset.natural_exact_occlusion_solver_v2 import (
    CONTRACT_VERSION, FullInterval, analytic_phase_intervals,
    classify_occlusion, load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/natural_short_full_occlusion_contract_v2.yaml"


class CoreContractTests(unittest.TestCase):
    def test_contract_loads_and_is_versioned(self):
        self.assertEqual(load_contract(CONTRACT)["version"], CONTRACT_VERSION)

    def test_phase_partition_width(self):
        result = analytic_phase_intervals(FullInterval(.013, .163), .1)
        self.assertAlmostEqual(sum(
            row["width_s"] for rows in result.values() for row in rows
        ), .1)


def add_case(cls, name, function):
    setattr(cls, name, function)


for radius in (.20, .25, .30):
    add_case(
        CoreContractTests, f"test_legal_radius_{str(radius).replace('.', '_')}",
        lambda self, radius=radius: self.assertIn(
            radius, load_contract(CONTRACT)["actor"]["allowed_radius_m"]
        ),
    )

for index, radius in enumerate((0., .05, .10, .15, .19)):
    add_case(
        CoreContractTests, f"test_illegal_small_radius_{index}",
        lambda self, radius=radius: self.assertNotIn(
            radius, load_contract(CONTRACT)["actor"]["allowed_radius_m"]
        ),
    )

taxonomy = (
    (100, 20, 50, 0, "partial_occlusion"),
    (100, 1, 99, 0, "severe_partial_occlusion"),
    (100, 0, 100, 1, "short_full_occlusion_gap1"),
    (100, 0, 100, 2, "short_full_occlusion_gap2"),
    (100, 0, 100, 3, "long_full_occlusion"),
    (0, 0, 0, 1, "not_projected"),
)
for index, row in enumerate(taxonomy):
    add_case(
        CoreContractTests, f"test_taxonomy_{index}",
        lambda self, row=row: self.assertEqual(
            classify_occlusion(*row[:4]), row[4]
        ),
    )

for index, (duration, expected) in enumerate((
    (.05, {0, 1}), (.10, {1}), (.15, {1, 2}),
    (.20, {2}), (.25, {2, 3}),
)):
    add_case(
        CoreContractTests, f"test_phase_counts_{index}",
        lambda self, duration=duration, expected=expected:
            self.assertEqual(
                set(analytic_phase_intervals(
                    FullInterval(.013, .013+duration), .1
                )),
                expected,
            ),
    )


class ArtifactContractTests(unittest.TestCase):
    pass


frozen_paths = (
    "authoritative_dataset/occlusion_constructor_v2_1.py",
    "authoritative_dataset/occlusion_constructor_v2_2.py",
    "authoritative_dataset/cuda_renderer_v1.py",
    "geometry_authority/static_v1.py",
    "policy/dynamic/track_manager.py",
    "policy/dynamic/range_image_foreground_v2_1.py",
    "reports/phase8jqv2_4mtc1_final_result.json",
    "reports/phase8jqv2_4mtc1_room_wall_near_miss_manifest.json",
    "reports/phase8jqv2_4rr1_natural_corpus_manifest.json",
)
for index, path in enumerate(frozen_paths):
    add_case(
        ArtifactContractTests, f"test_frozen_artifact_{index}",
        lambda self, path=path: self.assertIn(
            path, json.loads((
                ROOT / "reports/phase8jqv2_4eosr1_frozen_artifacts.json"
            ).read_text())["artifacts"]
        ),
    )

forbidden_fields = (
    "original_yopo_simulator_modified",
    "original_yopo_config_modified",
    "mtc1_artifacts_modified",
    "rr1_corpus_v1_modified",
    "historical_gap1_contract_modified",
    "occlusion_constructor_v2_1_modified",
    "occlusion_constructor_v2_2_modified",
    "authority_semantics_modified",
    "renderer_semantics_modified",
    "detector_modified",
    "TrackManager_algorithm_modified",
    "motion_contract_modified",
    "sensor_configuration_modified",
    "global_default_actor_radius_modified",
    "annex_used", "corpus_v2_created", "split_created",
    "representation_created", "detector_executed", "tracker_executed",
    "holdout_accessed", "formal_preflight_rerun",
    "formal_v3_entry_created", "formal_generation_started",
    "production_test_accessed", "blind_accessed",
    "optimizer_step_executed", "training_started",
)
for index, field in enumerate(forbidden_fields):
    def check_forbidden(self, field=field):
        path = ROOT / "reports/phase8jqv2_4eosr1_final_result.json"
        if not path.exists():
            self.skipTest("finalizer runs after host CUDA validation")
        self.assertIs(json.loads(path.read_text())[field], False)
    add_case(
        ArtifactContractTests, f"test_forbidden_mutation_{index}",
        check_forbidden,
    )

required_reports = (
    "phase8jqv2_4eosr1_entry_gate.json",
    "phase8jqv2_4eosr1_frozen_artifacts.json",
    "phase8jqv2_4eosr1_actor_radius_provenance.json",
    "phase8jqv2_4eosr1_radius_downstream_compatibility.json",
    "phase8jqv2_4eosr1_contract_versioning.json",
    "phase8jqv2_4eosr1_occlusion_scenario_taxonomy.json",
    "phase8jqv2_4eosr1_gap_length_contract.json",
    "phase8jqv2_4eosr1_near_miss_replay.json",
    "phase8jqv2_4eosr1_projected_boundary_model.json",
    "phase8jqv2_4eosr1_edge_grazing_candidates.json",
    "phase8jqv2_4eosr1_continuous_interval_solver.json",
    "phase8jqv2_4eosr1_sampling_phase_solver.json",
    "phase8jqv2_4eosr1_camera_motion_sweep.json",
    "phase8jqv2_4eosr1_radius_duration_sensitivity.json",
    "phase8jqv2_4eosr1_exact_geometry.json",
    "phase8jqv2_4eosr1_continuous_safety.json",
    "phase8jqv2_4eosr1_cuda_validation.json",
    "phase8jqv2_4eosr1_independent_rerender.json",
    "phase8jqv2_4eosr1_independent_validator.json",
    "phase8jqv2_4eosr1_proof_witness_manifest.json",
    "phase8jqv2_4eosr1_gap1_witnesses.json",
    "phase8jqv2_4eosr1_gap2_witnesses.json",
    "phase8jqv2_4eosr1_radius_coverage.json",
    "phase8jqv2_4eosr1_default_radius_coverage.json",
    "phase8jqv2_4eosr1_final_result.json",
)
for index, name in enumerate(required_reports):
    def check_report(self, name=name):
        final = ROOT / "reports/phase8jqv2_4eosr1_final_result.json"
        if not final.exists():
            self.skipTest("finalizer runs after host CUDA validation")
        self.assertTrue((ROOT / "reports" / name).is_file())
    add_case(
        ArtifactContractTests, f"test_required_report_{index}",
        check_report,
    )


if __name__ == "__main__":
    unittest.main()
