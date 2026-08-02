# Phase 6 — Dynamic loss and sequence-training closure

Date: 2026-07-21  
Project: `/home/zjh/YOPO/DE-P`  
Simulator: `/home/zjh/YOPO/Simulator`

## 13.1 Current Implementation Baseline

After phase 5, the effective ROS path was depth-driven planning with optional, synchronized dynamic perception. `DynamicContext` modulated the corrected or legacy CNN output through the parameter-free operation `feature * (1 + alpha * attention)`. Static fallback was preserved. The trainer, however, still accepted only independent static `DEPDataset` frames, used a hard-coded batch size in reshapes, did not pass dynamic context or dynamic obstacles, and defined but did not apply gradient clipping. `DEPLoss` had only static smoothness, ESDF safety, and guidance terms.

Phase 6 changes are split between the two workspaces:

- Simulator: `src/CMakeLists.txt`, `src/package.xml`, `src/config/config.yaml`, `src/src/test_simulator_cuda.cpp`, and `src/readme.md`.
- Sensor/ROS boundary: `config/traj_opt.yaml`, `policy/dynamic/types.py`, `policy/dynamic/ros_bridge.py`, and `test_dep_ros.py`.
- Dynamic loss: `loss/trajectory_sampler.py`, `loss/dynamic_types.py`, `loss/dynamic_safety_loss.py`, `loss/loss_function.py`, and the minimal device-neutral changes in `loss/safety_loss.py` and `loss/smoothness_loss.py`.
- Sequence training: `policy/dynamic_training_config.py`, `policy/dynamic_sequence_dataset.py`, `policy/dynamic_collate.py`, `policy/dynamic/context.py`, `policy/dep_trainer.py`, and `train_dep.py`.
- Data and validation tools: `tools/generate_synthetic_dynamic_sequences.py`, `tools/validate_dynamic_dataset.py`, phase-6 unit tests, and the four host validation entry files.

No old checkpoint was overwritten. No long training and no large dataset recording were started.

## 13.2 Simulator PointCloud Frame Semantics

The old `/lidar_points` implementation published body-FLU point values but labeled the message `odom`. It is retained only as a deprecated compatibility topic; therefore that legacy topic still has intentionally documented frame/content inconsistency and is rejected by the strict DE-P pointcloud consumer.

The CUDA Simulator publisher now emits:

| Topic | Numeric content | `frame_id` | Status |
|---|---|---|---|
| `/lidar_points` | body FLU | historical `odom` | deprecated compatibility only |
| `/lidar_points_body` | body FLU | `quadrotor` | explicit diagnostic |
| `/lidar_points_optical` | camera optical RDF | `camera_optical` | corrected pointcloud input |
| `/lidar_points_odom` | fixed world coordinates | `world` | explicit diagnostic |

The corresponding TF chain is `world -> quadrotor -> camera_optical`. Depth and CameraInfo use the same timestamp and `camera_optical` frame. The ROS node resolves the camera pose at the sensor timestamp through TF when configured.

Full-access Gate A result:

- body-to-optical median geometry error: `3.37e-7 m`;
- body-to-world median geometry error: `8.94e-8 m`;
- depth and CameraInfo: `160×90`, matching timestamp/frame and intrinsics;
- depth dynamic maximum confirmed speed in the static scene: `0.0785 m/s`, zero dynamic tracks;
- optical pointcloud dynamic maximum speed in the same scene: `4.61 m/s`, five false dynamic tracks.

Thus sensor geometry passed, but the full 360-degree pointcloud clustering/tracking path did not pass the perception-equivalence gate. The official phase-6 dynamic sensor and training source remains **depth**. Corrected optical pointcloud is not authorized for formal training yet.

## 13.3 Depth Geometry

The raw sensor geometry is explicitly `160×90`, with `fx=fy=80`, `cx=80`, `cy=45`. CameraInfo describes that raw image. Dynamic depth back-projection uses the raw depth and those raw intrinsics. A separate nearest-neighbor resize produces the CNN tensor `160×96`; the resize does not alter or masquerade as sensor geometry.

Axis, round-trip, translation, 90-degree rotation, fixed-world-point, CameraInfo, encoding, synchronization, projection, and resize-separation tests all pass.

## 13.4 Dynamic Collision Loss

`DynamicObstacleBatch` is a strict, padded batch boundary containing world-frame position, velocity, 3×3 position covariance, radius, track timestamp, sample timestamp, confidence, validity mask, and dynamic mask. Shape, finiteness, covariance symmetry/PSD, radius, confidence, timestamp, and boolean-mask semantics are validated.

The shared `QuinticTrajectorySampler` exactly preserves the historical fifth-order coefficient map and 30-point time grid used by `SafetyLoss`. Its output is grouped as `[B, 15, T, 3]` in the fixed world frame.

For each active obstacle, the loss predicts

```text
p_obstacle(t) = p0 + v0 * (track_age + t)
```

and approximates covariance growth as

```text
P_position(t) = P0 + I * covariance_growth_rate * dt².
```

The conservative safe radius is UAV radius + obstacle radius + configured covariance sigma times the square root of the largest position-covariance eigenvalue. Penetration is penalized by a squared softplus. Samples receive exponential time discount. Multiple obstacles use `max`, making the result invariant to padding or duplicated objects rather than growing with obstacle count.

Tests cover collision/far separation, temporal separation, head-on and crossing motion, radius/covariance changes, stale and future timestamps, empty/padded/static objects, extreme values, max aggregation, and finite trajectory gradients that push away from an obstacle. All pass.

## 13.5 DEPLoss and Score Label

`DEPLossOutput.trajectory_cost` is now:

```text
weighted smoothness
+ weighted static ESDF safety
+ weighted guidance
+ weighted future dynamic safety.
```

The score target is exactly `trajectory_cost.detach()`, so dynamic cost changes both trajectory optimization and score supervision while labels remain detached. With `dynamic_loss.enabled=false`, the original three-value API and static values are preserved exactly. With no dynamic input or no active target, dynamic cost is exactly zero.

TensorBoard records smooth, static safety, dynamic safety, guidance, score, total, minimum dynamic distance, risky-trajectory ratio, obstacle count, fallback ratio, pre/post clipping gradient norms, and static/dynamic batch indicators.

## 13.6 Dynamic Sequence Dataset

The formal versioned layout is:

```text
dynamic_dataset/
  dataset_manifest.yaml
  splits/{train,valid,test}.txt
  sequences/<sequence_id>/
    metadata.yaml
    frames.csv
    depth/*.npy
    dynamic_objects/*.json
```

The loader rejects unknown/missing manifest, metadata, frame, or object fields; non-depth official sources; legacy `/lidar_points`; non-contiguous frame indices; non-increasing timestamps; split overlap; and random-seed leakage between splits.

Each sample is a configurable continuous history window and returns raw-derived depth history, current CNN depth, timestamps, camera positions, actual synchronized 9-D state, pose, fixed goal, map ID, current attention, current GT objects, sequence mask, sequence ID, and frame index. Depth arrays are loaded on demand through a bounded per-dataset LRU cache; the dataset is not preloaded.

For `context_source=ground_truth`, attention uses only the current frame and its history. Future GT never enters network context. Future motion is represented only through the loss label. `context_source=estimated` is deliberately rejected with `NotImplementedError` until per-window tracker warm-up is implemented; it is not falsely advertised as active.

The deterministic smoke generator produces four schema-identical synthetic sequences (sphere crossing, empty, head-on cylinder, and an occluded crossing), ten frames each, with camera motion. Occluded GT is retained with zero visibility for supervision bookkeeping but is excluded from current network attention. Split validation produced 14 train, 7 validation, and 7 test windows with no leakage. This is **synthetic smoke only**, not formal recorded training data.

## 13.7 Training

`train_dep.py` now exposes real arguments for dataset mode, epochs, batch size, learning rate, dynamic root/context/loss, freezing, resume, save interval, gradient norm, seed, and workers. It intentionally defaults to one epoch instead of silently launching ten.

`DepTrainer` supports `static`, `dynamic`, and `mixed`. Every reshape uses the actual `B=depth.shape[0]`; incomplete batches are no longer silently skipped. A dynamic batch passes an independent `[B,1,3,5]` context to `DepNetwork.inference` and a separately padded obstacle batch to `DEPLoss`. Mixed mode schedules static and dynamic loaders according to `static_batch_ratio` without requiring a shared file format.

The current attention fusion has no trainable fusion parameters. The report therefore does not claim “fusion-only training.” The `late` policy freezes early MobileNet blocks and trains the final three feature blocks, output 1×1 convolution, and `DepHead`, initialized from the phase-3 corrected checkpoint. `none` trains all parameters. Further unfreezing is a future long-training decision.

`clip_grad_norm_` runs after backward and before optimizer step when the configured maximum is positive. AdamW uses fused mode on supported CUDA and safely falls back on CPU/unsupported builds.

Dynamic checkpoints contain model, optimizer, optional scheduler, epoch, global step, and JSON-safe metadata: architecture, role, dynamic perception/loss/training configs, dataset mode/version/manifest hash, sensor and three frames, Simulator and DE-P revisions, epoch, step, seed, freeze policy, optimizer, and scheduler. Resume requires this full format; an old weight-only checkpoint cannot be mislabeled as a dynamic training resume. Legacy and corrected static weights remain strictly loadable.

## 13.8 Tests

### CPU and schema

- Phase-6 focused unit subset: 19 tests, PASS.
- Dynamic sequence/split tests: 7 tests, PASS, including occluded-GT context isolation.
- Synthetic dataset validator: PASS for train/valid/test.
- Exact shared sampler regression: PASS.
- Static safety monotonicity, continuity, gradient, signed SDF, and OOB tests: PASS.

### Full-access GPU

Dynamic loss validation on `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, capability 12.0:

- corrected checkpoint strict load: PASS;
- context `[1,1,3,5]`, endstate `[1,9,3,5]`, score `[1,3,5]`;
- dynamic loss mean `0.02663368`;
- 145 finite network gradient tensors;
- minimum dynamic distance `0.9232 m`;
- risky trajectory ratio `0.2667`;
- peak allocated memory `29,260,800 bytes`.

Mixed training smoke, three optimizer steps:

- moving dynamic loss `1.73696e-5`;
- empty dynamic loss exactly `0`;
- pre-clip norm `23.9702`, post-clip norm `0.09999999`;
- trained parameter maximum change `3.0085e-5`;
- validation BatchNorm statistics unchanged;
- mixed schedule included both batch types;
- checkpoint metadata, optimizer save, exact model restore, and resume: PASS;
- peak allocated memory `418,715,648 bytes`.

The first resume attempt correctly exposed YAML `ScalarFloat` objects as unsafe under PyTorch 2.7 `weights_only=True`; checkpoint metadata was converted to plain JSON scalar types, then the same host test passed.

### Regression and ROS

- Exact prompt baseline with ROS sourced and GPU visible: 116 tests, exit code 0 (one retained expected failure).
- Legacy CPU golden: exact zero difference at `1e-7` tolerance.
- Legacy and corrected strict checkpoint loading: PASS.
- Host sensor semantics Gate A: PASS for official depth; corrected optical geometry PASS but pointcloud perception-equivalence FAIL as described above.
- Simulator incremental `catkin_make`: PASS.
- Final process audit found no retained roscore, rosmaster, Simulator, Controller, or DE-P node.

A device-coupling regression was exposed because CPU tests ran while CUDA was visible. Lattice angles/rotations, smoothness matrices, zero costs, SDF bounds, and cropped SDF tensors now follow the actual input tensor's device/dtype. Formulas and golden values are unchanged; the complete GPU-visible baseline then passed.

## 13.9 Limitations

- This phase completed only deterministic unit tests and short synthetic GPU smoke; no formal long training was run.
- No avoidance rate, reach rate, collision rate, path-length, acceleration, or jerk performance conclusion is available.
- The current Simulator contains static map obstacles and vehicle motion but no controlled moving-obstacle actor, dynamic-object ground truth publisher, radius/type query, or repeatable crossing/head-on/multi-target scenario interface. Real dynamic scenario count is zero.
- No formal dynamic sequence dataset has been recorded. Synthetic scale is four sequences × ten frames; it exists only in temporary test directories and is removed after validation.
- Pointcloud geometry is corrected, but pointcloud tracking on the 360-degree cloud produced false dynamics and is not approved as a formal training source. Depth is the only approved source.
- Estimated-track training context is not enabled yet.
- The dynamic checkpoint proven by smoke was temporary and deleted with its temporary test directory; old persistent checkpoints were not overwritten.
- Full ROS closed-loop Gate E was not claimed. Its required dynamic-trained persistent checkpoint and real no-target/crossing/head-on/multi-target Simulator scenarios do not exist. Starting a static scene with a converted or three-step synthetic weight would not satisfy that gate.

## Final required answers

1. **Does old `/lidar_points` still have frame/content inconsistency?** Yes. It is deliberately retained as deprecated compatibility, warned, and rejected by formal consumers.
2. **Which topic is used by formal pointcloud mode?** `/lidar_points_optical`, frame `camera_optical`; however it is not approved for formal training until perception equivalence passes.
3. **What is the formal training sensor?** Depth only: raw depth + CameraInfo + synchronized odometry + TF extrinsics.
4. **Is CameraInfo actually published and synchronized?** Yes, by the CUDA Simulator publisher with matching timestamp/frame; Gate A verified it.
5. **Are raw depth and CNN resize separated?** Yes: raw `160×90` geometry and a separate `160×96` CNN tensor.
6. **Are dynamic positions and candidate trajectories in one fixed frame?** Yes, DynamicCollisionLoss compares both in the configured fixed `world` frame.
7. **Does DynamicCollisionLoss enter trajectory loss?** Yes.
8. **Does it enter score labels?** Yes, before the total label is detached.
9. **Is an empty dynamic scene exactly zero?** Yes, unit and host training smoke both verify exact zero.
10. **Does a spatial crossing at a different time receive lower cost?** Yes, covered by temporal-separation tests.
11. **Is network input free of future information?** Yes for the implemented GT-current context; future GT is confined to loss supervision.
12. **Is there a real continuous sequence dataset implementation?** Yes, a strict lazy continuous-window schema/loader exists. Only synthetic smoke data has been instantiated so far.
13. **Are splits isolated by sequence?** Yes, including random-seed leakage checks.
14. **Does the dataset avoid full preload?** Yes, it uses bounded lazy LRU loading.
15. **Does training actually pass DynamicContext?** Yes; the host mixed smoke exercised this path.
16. **Is parameter-free attention described accurately?** Yes. It has no trainable fusion parameters.
17. **Is gradient clipping active?** Yes; host evidence is `23.9702 -> 0.1000`.
18. **Does the corrected dynamic checkpoint format contain full metadata?** Yes, and save/resume was verified. The smoke artifact was temporary, not retained as a formal model.
19. **Did full-access GPU smoke pass?** Yes, both loss and mixed training/resume.
20. **Did complete ROS sensor semantics pass?** Official depth passed. Corrected optical geometry passed, but pointcloud perception equivalence did not; formal pointcloud remains disabled.
21. **Was only short smoke run?** Yes. No long training was started.
22. **Did SafetyLoss, eval, requirements layering, and static checkpoints regress?** No. The 116-test ROS/GPU-visible baseline passed; legacy CPU golden and both checkpoint variants strictly load.

## Gate summary and next decision

- Gate A: **PASS for official depth**; optical geometry pass, pointcloud training authorization fail.
- Gate B: **PASS**.
- Gate C: **PASS for schema/loader and synthetic smoke**; no formal recording yet.
- Gate D: **PASS**.
- Gate E: **NOT SATISFIED**, for explicitly documented missing real dynamic scenarios and formal dynamic checkpoint.

Recommended next step is to design Simulator moving actors plus synchronized dynamic-object ground truth, record a small formal depth dataset, validate it, then request approval for a bounded corrected mixed training run. Only after producing a retained dynamic checkpoint should the full ROS scenario matrix be executed.
