#!/usr/bin/env python3
"""Build immutable Phase 8F readiness reports from frozen evaluation JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / name).read_text())


def dump(name, payload):
    (REPORTS / name).write_text(json.dumps(payload, indent=2) + "\n")


def metric_row(name, values):
    return (f"| {name} | {values['visible_detection_precision']:.3f} | "
            f"{values['visible_detection_recall']:.3f} | "
            f"{values['active_track_precision']:.3f} | "
            f"{values['active_track_recall']:.3f} | "
            f"{values['never_observed_false_positive_match_count']} |")


def main():
    signal = load("phase8f_range_signal_analysis.json")
    quality = load("phase8f_perception_quality.json")
    test = load("phase8f_test_once_quality.json")
    train = quality["split_metrics"]["train"]
    valid = quality["split_metrics"]["valid"]
    test_metrics = test["split_metrics"]["test"]
    same_hash = quality["implementation_sha256"] == test["implementation_sha256"]

    signal["answers"] = {
        "dynamic_actor_pixel_capture": {
            "train_development_range_seed_recall": signal["selected_metrics"]["recall"],
            "full_train_range_seed_recall": train["pixel_stage_metrics"]["range_residual_seed"]["recall"],
            "full_valid_range_seed_recall": valid["pixel_stage_metrics"]["range_residual_seed"]["recall"],
        },
        "static_pixel_trigger_fraction_train_development":
            signal["selected_metrics"]["static_trigger_fraction"],
        "multi_history_consensus": (
            "Support=2 was selected by the frozen train-only objective. The persisted top-12 "
            "candidate table contains only support=2 candidates; a numeric support=1 comparison "
            "was not retained and is not reconstructed from valid/test."
        ),
        "held_out_recall_diagnosis": (
            "The signal is present but weaker on full valid: range-seed recall is 0.229. "
            "Component recall is 0.202, so most loss precedes tracking and some additional "
            "loss occurs during constrained completion."
        ),
        "hardest_scenarios_train_development": ["temporal_separation", "head_on"],
        "first_frame_visible_delay": (
            "No actor with first_visible_frame=0 produced a retained delay sample in the fixed "
            "signal manifest. Overall full-valid first seed/component/dynamic-attention means are "
            f"{valid['first_visible_to_first_seed_mean_s']:.3f}/"
            f"{valid['first_visible_to_first_component_mean_s']:.3f}/"
            f"{valid['first_visible_to_dynamic_attention_mean_s']:.3f} s."
        ),
    }
    signal["frozen_full_evaluation_implementation_sha256"] = quality["implementation_sha256"]
    dump("phase8f_range_signal_analysis.json", signal)

    gate = {
        "status": "FAIL",
        "perception_ready": False,
        "train_valid_gate": quality["status"],
        "test_once_gate": test["status"],
        "implementation_sha256": quality["implementation_sha256"],
        "test_used_same_frozen_implementation": same_hash,
        "full_sequence_counts": {"train": train["sequence_count"],
                                 "valid": valid["sequence_count"],
                                 "test": test_metrics["sequence_count"]},
        "hard_failures": test["hard_failures"],
        "test_was_used_for_parameter_selection": False,
        "test_rerun_allowed": False,
        "dynamic_cpu_tests": {"status": "PASS", "count": 94},
        "ros_import_without_master": {
            "status": "PASS", "raw_depth_shape": [90, 160],
            "depth_callback_uses_update_depth": True,
        },
        "live_ros_simulator_matrix": "NOT_RUN_NO_MASTER_OR_SENSOR_PUBLISHERS",
        "next_allowed_phase": None,
    }
    dump("phase8f_perception_gate.json", gate)

    design = f"""# Phase 8F causal range-image foreground

## Why temporal-voxel was sparse

The retained `temporal_voxel` mode receives flattened `[N,3]` points. Its
observed-free-space test can mark a moving object's newly occupied leading
edge, but pixel adjacency, interior and trailing contour are already lost.
Global DBSCAN cannot reconstruct a full object from those sparse seeds;
lowering DBSCAN thresholds or unconstrained dilation previously reintroduced
static edges. The old mode remains available and is still the YAML default.

## Implemented path

`DepthFrame` preserves raw float32 metric depth, validity, sampled optical
points and their raw `(u,v)` pixels. Formal depth paths now call
`DynamicPerception.update_depth`; point-cloud users retain `update`.

The independent `range_image_hybrid` mode performs bounded, causal history
reprojection with ego-motion compensation and z-buffering. A closer-than-
history residual with train-selected absolute/relative thresholds and
multi-history support is combined with sparse prior-free-space evidence.
Pixels consistently explained by history are hard negatives. Components grow
only from a hard seed, only through positive closer evidence, and cannot cross
configured range or 3-D neighbour discontinuities. Each image component is
converted directly into a `ClusterObservation`; global DBSCAN is not used in
this mode. `foreground_support` is evidence-derived rather than fixed at 1.

No future frame, actor GT, static PLY or ESDF is used at runtime. Actor GT is
used only by the evaluator. Because the recorded dataset has no instance mask,
pixel metrics use a documented conservative projected/depth proxy.

## Frozen results

- Full train component precision/recall: {train['pixel_stage_metrics']['completed_component']['precision']:.3f}/{train['pixel_stage_metrics']['completed_component']['recall']:.3f}.
- Full valid component precision/recall: {valid['pixel_stage_metrics']['completed_component']['precision']:.3f}/{valid['pixel_stage_metrics']['completed_component']['recall']:.3f}.
- Full valid complete perception mean/P95: {valid['complete_perception_latency_ms']['mean']:.1f}/{valid['complete_perception_latency_ms']['p95']:.1f} ms (10 Hz period: 100 ms).
- History is bounded to {valid['bounded_capacity']['range_history_frames']} frames; world memory remains capped at {valid['bounded_capacity']['background_max_voxels']} voxels.
- New Phase 8F tests: 25/25; combined dynamic regression selection: 94/94.

## Known limitations

Small targets remain hardest: full-valid active recall for projected area below
25 pixels is {valid['stratified_active_recall']['pixel_area_bin']['0-25']['recall']:.3f}.
The one-time frozen test failed only the never-observed hard constraint (13
matches across three sequences). Per protocol, this result was not used to
tune or rerun the model. Live ROS/Simulator validation was not available;
Noetic import and callback binding passed without a Master.
"""
    (REPORTS / "phase8f_range_image_foreground.md").write_text(design)

    readiness = f"""# Phase 8F final readiness

Final status: **FAIL** (`perception_ready: false`).

The complete frozen train+valid Gate passed, but the single permitted test run
failed `never_observed_false_matches = 0`: it produced 13 matches. All 13 are
recorded with sequence, frame, actor and track IDs in
`phase8f_test_once_quality.json`. The implementation hash was identical for
train/valid and test: `{quality['implementation_sha256']}`.

| Split | Visible P/R | Active P/R | Never-observed matches |
|---|---:|---:|---:|
| train | {train['visible_detection_precision']:.3f}/{train['visible_detection_recall']:.3f} | {train['active_track_precision']:.3f}/{train['active_track_recall']:.3f} | 0 |
| valid | {valid['visible_detection_precision']:.3f}/{valid['visible_detection_recall']:.3f} | {valid['active_track_precision']:.3f}/{valid['active_track_recall']:.3f} | 0 |
| test (once) | {test_metrics['visible_detection_precision']:.3f}/{test_metrics['visible_detection_recall']:.3f} | {test_metrics['active_track_precision']:.3f}/{test_metrics['active_track_recall']:.3f} | 13 |

No test-driven parameter change or second test run was performed. Therefore
Phase 8F does **not** authorize candidate scoring, ranking loss, gated fusion,
estimated-context training, production launcher changes, or long training.
The next step requires a separately approved phase and a new untouched
evaluation protocol; this report does not declare `next_allowed_phase`.
"""
    (REPORTS / "phase8f_final_readiness.md").write_text(readiness)

    limitations = """# Phase 8F range-signal limitations

The deterministic range signal is usable: full train/valid tracking Gates pass
and range-seed recall is non-zero on held-out maps. It is not sufficient for a
final production PASS under the current protocol because the one-time test run
contains 13 matches to actors classified as never observed. These occur in
three test sequences and are not no-target failures; their details are frozen
in the test report. The dataset does not contain per-pixel actor instance
masks, so pixel precision/recall relies on a conservative geometry/depth proxy.

No neural foreground refiner was introduced. No threshold or association rule
was changed after observing test. A future attempt must use a newly approved
protocol rather than rerunning or tuning against this test split.
"""
    (REPORTS / "phase8f_range_signal_limitations.md").write_text(limitations)


if __name__ == "__main__":
    main()
