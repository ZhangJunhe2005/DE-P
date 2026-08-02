# Phase 7 — Simulator dynamic scenarios and formal depth pilot dataset

## 1. Simulator audit before the Phase-7 actor path

The CUDA Simulator loaded either a generated `mocka::Maps` point cloud or a PLY selected by `config/config.yaml`, voxelized it once into `raycast::GridMap`, and ray-cast both depth and lidar against that static grid. `/sim/odom` supplied vehicle pose. Depth publication was wall-scheduled by `ros::Time::now()`, while depth and CameraInfo headers used the triggering odometry stamp. There was no independent movable entity, mesh actor, dynamic collision system, synchronized object GT, or custom message generation. Marker-only visualization would therefore not have affected depth.

The static depth path is `odomCallback -> renderDepthCallback -> renderDepthImage(GridMap, camera, T_wc)`. Camera native rays are forward-left-up and published images use the corresponding optical RDF frame. The static lidar and depth renderers share `GridMap`; they do not share a mutable scene graph. Runtime map-object pose mutation was not supported. The package initially had no structured dynamic-object messages.

Phase 7 keeps this static path intact and inserts dynamic rendering only after the CUDA static depth image is returned:

```text
static point cloud -> GridMap -> CUDA static depth ----+
                                                     min-depth overlay -> /depth_image
scenario YAML -> world-frame actor state -> analytic ray intersections ----+
                                      +-> synchronized world-frame GT
                                      +-> independent UAV collision diagnostic
```

Actors are never appended to the static cloud, PLY, `GridMap`, ESDF, lidar output, or SafetyLoss map. The controlled static PLY SHA-256 remained
`288a00863f0f28b202ebd4b34235abd22521b1ec3ffa743102f5f412825292b2`
before and after smoke and recording.

## 2. Dynamic actor implementation

Implementation files are `Simulator/src/include/dynamic_actor.hpp` and `Simulator/src/src/dynamic_actor.cpp`. Configuration is strict YAML. Invalid/missing geometry, non-finite values, unsupported types, invalid time intervals, zero IDs, and duplicate IDs fail immediately.

Supported shapes:

- sphere, using analytic ray/sphere intersection;
- finite vertical cylinder, including side and end-cap intersections.

Supported world-frame trajectories:

- stationary;
- linear;
- delayed linear through an explicit `start_time`;
- waypoint ping-pong at configured speed.

IDs are configured and stable. Lifecycle is explicit: before `start_time` and after `end_time`, an actor is published inactive and is neither rendered nor collision-tested. Linear position is continuous at lifecycle boundaries; ping-pong position is continuous at reversal points, with the intentional instantaneous velocity sign change. Given the same YAML and seed, sampling is deterministic; there is currently no random perturbation after loading.

Scenario time is derived from odometry simulation stamps relative to the first accepted odometry stamp. Odom stamps must be strictly increasing. Wall time is used only to schedule sensor publication, not to evaluate trajectories. Configurations are selected at process start with `~dynamic_scenario_file`; there is no live service to reload or manually teleport actors.

## 3. Sensor, visibility, and collision semantics

For every depth frame, actors are sampled at exactly the depth/odom timestamp. Rays use the same camera pose and intrinsics as static rendering. An actor replaces a pixel only when its surface is closer than the existing static depth, so static geometry occludes actors. Leaving the FOV or moving behind the camera stops pixel contribution. `visible` means at least one final depth pixel belongs to that actor; `occluded` means active, projected-center-in-image, but no final actor pixel remains. Partial edge visibility is valid even when the projected center is outside the image.

Collision is a synchronized geometry diagnostic using a configurable UAV radius:

- sphere actor versus spherical UAV;
- finite vertical-cylinder actor versus spherical UAV.

It does not modify the static ESDF. Lidar intentionally remains static-only in this stage.

Simulator depth saturation is exactly `camera_max_depth`. Phase 7 excludes exactly saturated pixels when converting depth to DynamicPerception points; they represent no hit, not an obstacle surface. CPU and Torch regression tests cover this rule.

## 4. Ground-truth messages and topics

`sensor_simulator/DynamicObjectState.msg` carries header, stable ID, type, active/visible/occluded/inside-image/collision flags, world position/velocity/acceleration, geometry, projected center, expected/observed depth, depth error, and rendered-pixel count. `DynamicObjectStateArray.msg` carries header, scenario ID, seed, scenario time, UAV collision, and objects.

- GT topic: `/dynamic_objects/ground_truth`;
- collision topic: `/dynamic_objects/uav_collision`;
- optional debug markers: `/dynamic_objects/markers`;
- formal frame: `world`;
- stamp: identical to depth, CameraInfo, and triggering odometry;
- no-target behavior: a stamped array with zero objects, not a missing message.

Marker output is diagnostic only and is not used as a label. CMake/package metadata now generate the custom messages and declare `message_runtime`.

## 5. Configured scenario families

Independent YAML configurations cover:

- no-target/static control;
- left-to-right and right-to-left crossing with varied speed/start time, including time-offset second actors;
- head-on cylinders with varied speed, offset, distance, and delay;
- multi-target mixtures of crossing, head-on, and waypoint ping-pong actors.

The smoke configurations live under `Simulator/src/config/dynamic_scenarios`. The 24 exact pilot YAMLs are preserved under `data/phase7_dynamic_pilot/scenario_configs`; scenarios require no C++ constant edits.

## 6. Full-access four-scenario smoke

Host: NVIDIA GeForce RTX 5070 Ti Laptop GPU, CUDA build 12.8. Each scene collected 180 synchronized raw `160x90` depth/CameraInfo/odom/GT frames. Lidar was disabled. Result: `HOST_DYNAMIC_SCENARIO_SMOKE_RESULT.status=PASS`.

| scene | active objects max | visible depth frames | collision frames | max velocity error | max DynamicPerception dynamic tracks |
|---|---:|---:|---:|---:|---:|
| no-target | 0 | 0 | 0 | 0 | 0 |
| crossing | 2 | 62 | 11 | 1.43e-5 m/s | 1 |
| head-on | 1 | 53 | 15 | 2.86e-5 m/s | 1 |
| multi-target | 3 | 57 | 11 | 3.36e-5 m/s | 2 |

All four had zero synchronization error at message-stamp precision, stable configured IDs, valid raw depth, and passing projection checks. No-target false dynamics were exactly zero. Logs for the final run are under `/tmp/dep-phase7-smoke-Q7GutI`. All started ROS processes were stopped.

## 7. Formal recorder and schema

`tools/record_dynamic_sequences.py` uses an `ApproximateTimeSynchronizer` over raw depth, CameraInfo, odometry, and structured GT. It records only `sensor_source: depth`; pointcloud paths are empty. It writes lossless float32 `.npy` depth at the original `160x90`; resizing to `160x96` remains an online Dataset operation.

Each frame records timestamp, paths, body pose, velocity, finite-difference acceleration, goal, map ID, scenario ID, seed, and three depth-relative synchronization offsets. Object JSON stores current active GT plus visibility, geometry, collision, projection, and depth diagnostics. Metadata contains frames/rate, scenario, commits, config hash, static-map hash, intrinsics/extrinsics, sync slop and observed errors.

Metadata is created with `completion_status: incomplete`. Only reaching the requested finite frame count writes `frames.csv`, updates metadata to `complete`, and allows the finalizer to add the sequence to a split. The recorder refuses overwrite and automatically stops at the target count.

## 8. Pilot dataset

Location: `data/phase7_dynamic_pilot` (about 91 MiB).

- 24 complete sequences;
- 1,440 raw depth frames;
- 6 sequences and 360 frames per scenario family;
- 60 frames per sequence at 10 Hz;
- observed durations 5.90–6.03 s;
- 24 unique seeds;
- split by sequence and seed: train 16, valid 4, test 4;
- every split contains all four scene families;
- no seed crosses splits;
- source is exclusively depth.

Scene statistics:

| scene | visible frames | collision frames | simultaneous active max | GT velocity max error | projected depth-neighborhood max error |
|---|---:|---:|---:|---:|---:|
| no-target | 0 | 0 | 0 | 0 | 0 |
| crossing | 153 | 10 | 2 | 6.46e-6 m/s | 0 m |
| head-on | 177 | 27 | 1 | 1.21e-5 m/s | 0 m |
| multi-target | 185 | 52 | 3 | 9.75e-6 m/s | 2.86e-4 m |

In multi-target overlap, the projected center pixel can belong to a nearer actor; its maximum stored center-pixel diagnostic is 4.28 m. The required neighborhood alignment metric searches the actor projection footprint and is 0.000286 m, distinguishing partial inter-actor occlusion from geometric misalignment.

Across dynamic families the validator found 1,244 active-but-invisible object frames, including 1,116 frames whose actor center was geometrically behind the camera and 15 fully occluded multi-target frames. All had no final actor depth support as required.

Preview: `reports/phase7_pilot_preview.png`. Complete machine-readable validation: `reports/phase7_pilot_validation.json`.

## 9. Schema, temporal, coordinate, and loss validation

`tools/validate_phase7_pilot_dataset.py` passes all checks:

- manifest, metadata, frames, object JSON, file existence, dtype and shape;
- contiguous frame indices and strictly increasing timestamps;
- exact recorded synchronization offsets (maximum 0 in this run);
- world/camera frame metadata and `160x90` CameraInfo geometry;
- stable IDs and position/velocity finite-difference consistency;
- no-target empty GT, both crossing directions/config variation, approaching head-on motion, and at least three simultaneous multi-target actors;
- unchanged static-map hash and no lidar source/path;
- formal loader produces `[1,96,160]` current depth, `[1,1,3,5]` DynamicContext, and valid padded DynamicObstacleBatch;
- no future annotation enters network context. DynamicCollisionLoss uses current state and constant-velocity extrapolation over future sample times.

Offline straight-candidate loss scan covered 342 windows per family:

| scene | positive windows | maximum dynamic cost |
|---|---:|---:|
| no-target | 0 | exactly 0 |
| crossing | 139 | 7.9739 |
| head-on | 159 | 6.8461 |
| multi-target | 167 | 10.0457 |

Thus no-target is exact zero and every risk family contains nonzero risk windows. These are data/loss acceptance calculations, not model training.

## 10. Regression results

- Simulator `catkin_make`: PASS, including custom message generation and CUDA executable.
- Python unit tests after sourcing ROS: 117 PASS, 1 expected failure.
- legacy checkpoint strict load and CPU golden: PASS, max differences 0.
- corrected architecture, DynamicPerception GPU, dynamic network GPU, and DynamicCollisionLoss GPU: PASS.
- bounded phase-6 mixed-training regression: PASS, exactly 3 optimizer steps, temporary artifacts removed.
- ROS sensor semantics: PASS; depth ego-motion produced 0 dynamic tracks.
- ROS pointcloud diagnostic still produced up to 6 false dynamic tracks.
- All test-started ROS processes were cleaned up.

## 11. Pointcloud boundary and remaining limitations

`official_sensor_source: depth` and `formal_pointcloud_training_allowed: false` remain mandatory. Dynamic actors are not inserted into lidar in this stage. Existing pointcloud problems remain isolated: 360-degree surfaces, large-plane centroid drift, occlusion-induced centroid changes, sampling instability, and timing sensitivity. The legacy `/lidar_points` frame/content mismatch remains retained only for compatibility; corrected body/optical/world topics exist.

Other limits:

- actors are analytic sphere/cylinder geometry, not arbitrary mesh actors;
- scenario config is selected at startup, not hot-reloaded;
- collision is diagnostic and does not physically stop/bounce the UAV;
- multi-actor center-pixel diagnostics require neighborhood interpretation during overlap;
- the seed is persisted for reproducibility, but actor YAMLs are presently deterministic and do not sample randomness;
- this pilot uses a controlled sparse boundary map for sensor/GT alignment, not a diversity-complete navigation benchmark;
- no dynamic model checkpoint was trained or retained, and Gate E is not claimed complete.

## 12. Required final answers

1. **Does Simulator now have moving actors that truly enter depth?** Yes, analytic sphere/cylinder actors overwrite nearer pixels in the raw sensor depth path.
2. **Is synchronized world-frame GT published?** Yes, on `/dynamic_objects/ground_truth` with the depth/CameraInfo/odom stamp.
3. **Are object IDs stable?** Yes, configured IDs persist across all lifecycle frames.
4. **Are no-target, crossing, head-on, and multi-target supported?** Yes, by independent YAML configurations.
5. **Is fixed-seed reproduction supported?** Yes; exact YAML and seed are persisted, and current sampling is deterministic.
6. **Are dynamic objects excluded from static ESDF?** Yes; the static PLY/GridMap hash is unchanged and actors exist only in the depth overlay/collision/GT path.
7. **Are GT, depth, odom, and CameraInfo synchronized?** Yes; recorded maximum header-stamp error is 0.
8. **Were at least 24 formal depth pilot sequences recorded?** Yes: 24 complete sequences, 1,440 frames.
9. **Did schema, time, frame, projection, and split validation pass?** Yes; see `phase7_pilot_validation.json`.
10. **Is pointcloud still forbidden for formal training?** Yes.
11. **Was model training started without authorization?** No. Only the explicitly required three-step regression smoke ran; no long/formal training or dynamic checkpoint creation occurred.

Phase 7 stops here. Bounded mixed training has not been started, and Gate E has not been declared complete.
