# Phase 8 dynamic-training readiness report

Date: 2026-07-22  
Project: `/home/zjh/YOPO/DE-P`  
Final decision: **`production_ready: false`**

The production launcher is intentionally locked. The final host-GPU preflight has one
unresolved hard Gate: on the fixed risk set, average recorded-future dynamic loss rose
from `0.0076160` to `0.0084942` after 30 optimization steps. This is a failure, not a
warning. Production data recording is also awaiting capacity approval and no production
dynamic checkpoint exists.

## Implemented hardening

- `DynamicObstacleBatch` now optionally carries world-frame
  `future_positions_world [B,M,N,3]`, `future_valid_mask`,
  `future_visibility_mask`, and strictly increasing `future_timestamps [B,N]`, plus
  `observable_mask` and `ever_observed_in_history`.
- `dynamic_loss.target_source` is strict: `recorded_future_gt` (formal default) or
  `constant_velocity` (fallback/ablation). Both use the same quintic candidate samples;
  their absolute label delta is logged.
- The sequence loader linearly interpolates only label-side future annotations onto the
  30 sampler timestamps. It masks sequence ends and actor lifecycle gaps. Future data is
  never inserted into depth, observation, attention, `DynamicContext`, or network forward.
- A target is supervised only after it has been visible in the history window. A briefly
  occluded target is propagated from its last visible state while the track is valid;
  behind-camera/future-first targets are excluded.
- Context modes are `ground_truth`, deterministic `noisy_ground_truth`, and `estimated`.
  Estimated mode creates a fresh `DynamicPerception` per window, warms it only with
  historical depth/pose/intrinsics/timestamps, and optionally caches by perception config,
  sequence hash, split, frame, timestamps, and tracker version. No state crosses samples,
  batches, workers, or splits.
- YAML curriculum stages select context source and ratio by epoch. The current stage and
  full sampler weights are included in checkpoint metadata.
- Training uses a deterministic `WeightedRandomSampler` only for train. Validation/test
  remain unweighted. Six categories are recognized and per-epoch emitted counts are stored
  in `status.json`.
- Static safety maps now use an explicit external `map_id -> ESDF index` catalog instead
  of treating `map_id` as a list index. `map_id=700` was tested with the phase-7 control
  PLY and returned finite distance/cost. Dynamic actors never enter the static PLY.
- Late-freeze training keeps frozen BatchNorm statistics in eval mode while trainable
  BatchNorm remains train mode. This fixed a deterministic train/eval drift found by Gate.
- Checkpoints use temporary-file + atomic rename, SHA-256 sidecars, strict corruption and
  resume-metadata checks, and include model, optimizer, scheduler and optional scaler state.

## Existing generator audit and multi-map preflight

The Simulator's existing `dataset_generator.cpp` uses `mocka::Maps`, writes
`pointcloud-<id>.ply`, `pose-<id>.csv`, and matching depth frames; `SafetyLoss` builds its
ESDF from those PLY files. The generator is config-driven for map count/seed, but its pose
sampling still uses a nondeterministic `random_device`, checks local obstacle clearance
rather than start-goal graph reachability, and destructively prepares its output directory.
Those constraints are recorded rather than hidden.

`tools/generate_dynamic_map_dataset.py` therefore has two bounded roles: it builds the
non-production preflight dataset from existing compatible YOPO maps, and produces a
plan-only production manifest until explicit capacity approval. It rejects blank and
implausibly dense PLYs, records hashes/bounds/density/separate actor seeds, and declares
map-level splits. It does not invent a new static-map format.

The bounded dataset contains 4 maps and 13 sequences:

| Split | Maps | Windows | Categories present |
|---|---:|---:|---|
| train | 0, 1 | 147 | all six categories |
| valid | 2 | 63 | no-target, temporally-separated, multi-target |
| test | 3 | 63 | crossing/head-on, low-risk, occluded-but-tracked |

Map sets and sequence/actor seeds are disjoint across splits. This dataset is explicitly
`bounded_synthetic_preflight_only`; it is not presented as a formal Simulator recording.

## Capacity plan

The immutable phase-7 pilot has 24 sequences, 1,440 frames and 85,076,694 bytes:

- 3,543,881 bytes/sequence and 59,065 bytes/frame;
- proposed 12 train + 3 valid + 3 test maps, 12 sequences/map, 216 sequences total;
- estimated dataset: 765,478,269 bytes (0.713 GiB);
- estimated context cache: 61,238,261 bytes (0.057 GiB);
- simulated duration 0.36 h, recording wall estimate 0.54 h, validation estimate 32.4 min.

The machine had about 257 GiB free during preflight. The plan remains
`AWAITING_USER_APPROVAL`; no production-scale generation or recording was started. See
`reports/phase8_dataset_capacity_plan.json`.

## Verification results

### CPU and schema

- Full suite: 127 tests passed, 1 pre-existing expected failure.
- Additional future/observability/map tests: linear GT vs constant velocity, waypoint
  reversal, delayed start, interpolation boundary, lifecycle/end mask, strict timestamps,
  no-target zero, future-first exclusion, occlusion retention, noisy-GT determinism,
  estimated-context cache invalidation contract, and map split isolation passed.
- Legacy CPU golden and corrected architecture/checkpoint tests passed.
- Shell syntax and Python compilation passed.
- Formal pointcloud training remains forbidden; official dynamic source is depth.

### Host RTX 5070 Ti preflight

Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU, capability `(12,0)`.

Small-set result after the frozen-BN correction:

| Metric | Before | After | Result |
|---|---:|---:|---|
| total loss | 4.91455 | 2.96102 | PASS |
| score loss | 1.30405 | 0.57049 | PASS |
| recorded-future dynamic loss | 0.007616 | 0.008494 | **FAIL** |
| static cost | 2.26727 | 1.57119 | PASS |
| Spearman(score, risk) | 0.24885 | 0.35190 | PASS |
| Kendall(score, risk) | 0.17883 | 0.23941 | PASS |
| minimum dynamic distance | 0.49038 m | 0.57954 m | improved |
| no-target dynamic loss | 0 | 0 | PASS |

Parameters changed (`max_abs_delta=0.003127`), all values stayed finite, and exact
checkpoint restore passed. GPU allocation was stable (`418,693,120` peak, zero measured
growth after warm-up); CPU RSS grew 270,336 bytes and file descriptors did not grow.

The independent bounded run completed 8 train steps and 16 validation batches on train
maps 0/1 and validation map 2. It produced TensorBoard, CSV, status, latest/best/epoch
checkpoints and matching SHA-256 sidecars under
`runs/dynamic/20260722-100819-phase8_preflight_v1/`. Validation dynamic loss was nonzero
(`0.0078810`). The watchdog and checkpoint checksum passed. Exact resume was separately
verified by the small-set Gate.

## Managed production and evaluation entry points

- Configs: `configs/train_dynamic_preflight.yaml`,
  `configs/train_dynamic_production.yaml`, `configs/eval_gate_e_random_maps.yaml`.
- Lifecycle scripts: `scripts/phase8_preflight.sh`,
  `train_dynamic_production.sh`, `resume_dynamic_production.sh`,
  `monitor_dynamic_training.sh`, and `stop_dynamic_training.sh`.
- The managed runner creates unique `runs/dynamic/<run_id>/` directories containing the
  resolved config, command, environment, DE-P/Simulator git state, log, metrics, status,
  TensorBoard and atomic checkpoints. SIGINT/SIGTERM stop after the current batch.
- The launcher requires both a preflight JSON `PASS` and explicit confirmation/`--yes`.
  Because the current Gate is `FAIL`, it cannot launch production.
- Offline held-out comparison is prepared by
  `tools/evaluate_random_dynamic_dataset.py` and its shell wrapper. It compares static and
  dynamic corrected checkpoints by category and writes JSON, CSV, and a regret plot.
- Gate-E config/script are prepared but deliberately do not execute the ROS matrix until
  formal held-out data and a production dynamic checkpoint exist.

## Required answers

1. **Recorded future GT loss?** Yes, with strict aligned tensors, interpolation and masks.
2. **Future leakage remaining?** No known path; tests prove future-label mutation cannot
   change attention/context and future-first actors are excluded.
3. **Estimated context?** Yes, history-only, fresh tracker per window, versioned cache.
4. **Never-observed behind-camera actor excluded?** Yes, from context and current loss.
5. **Previously observed occluded actor retained?** Yes, within tracker validity.
6. **Multiple static maps?** Yes for bounded preflight (four compatible YOPO maps); formal
   production maps are not yet recorded.
7. **Map-isolated train/valid/test?** Yes in the preflight manifest; formal manifest pending.
8. **Sampler balanced?** Weighted deterministic train sampler supports all six categories;
   validation/test are unweighted and emitted train counts are recorded.
9. **Small-set overfit passed?** **No.** Total/score passed, but average dynamic loss failed.
10. **One-epoch bounded run passed?** Yes, including independent-map validation artifacts.
11. **Resume passed?** Yes, exact parameters and optimizer metadata; checksum verified.
12. **Static regression passed?** Yes for the bounded threshold and CPU golden checks.
13. **Score/risk sanity passed?** Correlations and oracle/top-1 checks did not regress;
    dynamic-loss descent did not pass.
14. **Memory stable?** Yes over the bounded measurement; long production stability is unmeasured.
15. **Launch/resume/monitor/stop scripts generated?** Yes.
16. **Production training not automatically started?** Correct; it remains locked.
17. **Held-out offline evaluation ready?** The code/config/output formats are ready; execution
    awaits formal test maps and a dynamic checkpoint.
18. **Gate-E random-map scripts ready?** Preparation is ready; the full ROS matrix is locked
    until its required data/checkpoint exist.
19. **Pointcloud formal training forbidden?** Yes, explicitly in manifests/configs/scripts.
20. **Final production_ready?** **false** because a key small-set dynamic-loss Gate failed,
    production data approval/recording is pending, and no production checkpoint exists.

## Remaining blockers and recommended next action

Do not start production training. The next engineering task is to diagnose why minimizing
the combined trajectory objective improves minimum clearance and risk ranking while raising
the mean recorded-future dynamic cost across all 15 candidates. Candidate checks are:

1. report per-candidate/per-time gradients and split risk-only vs static/guidance gradients;
2. inspect max-over-obstacle aggregation and whether improvements are concentrated in top-1
   while other primitives become riskier;
3. define a configured Gate metric that includes both top-1 collision risk and distributional
   risk without weakening the existing mean-cost failure;
4. rerun the same immutable preflight before approving the 0.713-GiB production plan.

Production-scale data generation and training require separate user approval after this Gate
passes.
