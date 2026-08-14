# Route-A V4.9.1 reproducibility

The release tag `route-a-v4.9.1-recovery-subgoal-v6` freezes the latest
avoidance stack:

- MobileNetV3 V4.8.3 static policy and its exact checkpoint;
- V4.9 causal dynamic tracking and motion-preserving dynamic safety filter;
- universal bounded-scan deadlock recovery;
- V4.9.1 certified 3.5--5.0 m temporary recovery target;
- mission-goal restoration, yaw reacquisition and re-arm lifecycle V6;
- stale-command and goal-generation race protection.

The policy remains the translation owner.  Recovery changes the temporary
conditioning goal only after a network candidate has passed the existing
static and predicted-dynamic safety evaluation.  Runtime safety never invents
a replacement trajectory.

## Checkout and algorithm verification

```bash
git clone --branch route-a-v4.9.1-recovery-subgoal-v6 \
  https://github.com/ZhangJunhe2005/DE-P.git DE-P-workspace
cd DE-P-workspace/DE-P
conda activate yopo
bash scripts/verify_route_a_v4_9_1_checkout.sh
```

This verifies the frozen checkpoint SHA-256 and all focused V4.9/V4.9.1
recovery tests.  It does not start ROS, modify a dataset or execute training.
The machine-readable identity is
`configs/route_a_v4_9_1_release_manifest.yaml`.

## Full RViz replay

The repository tracks the 5.4 MB policy checkpoint.  Generated point clouds,
canonical occupancies and ROS build products are intentionally not committed.
They are too large and are environment products rather than source code.  For
the exact current fixture, place the existing V4.6 assets under the relative
paths listed by `configs/dep_interactive_demo_scenes_v4_6.json`, and keep the
ROS workspaces adjacent:

```text
DE-P-workspace/DE-P
DE-P-workspace/Controller
DE-P-workspace/Simulator
```

The four point-cloud hashes and required release identities are recorded in
`configs/route_a_v4_9_1_release_manifest.yaml`.  Validate all runtime assets
without starting ROS master:

```bash
bash scripts/verify_route_a_v4_9_1_checkout.sh --runtime-assets
```

Then launch one scene (the path is no longer tied to `/home/zjh`):

```bash
bash scripts/route_a_v4_9_1_dynamic_rviz_host.sh pillar
```

Valid scene names are `cave`, `forest`, `pillar` and `wall`.  The launcher
defaults to 16 uniformly distributed 3-D actors and seed 9098.  A different
asset contract or an unpacked copy of the same checkpoint may be selected
without editing source:

```bash
DEP_V491_SCENES_CONFIG=/absolute/path/scenes.json \
DEP_V491_CHECKPOINT=/absolute/path/best.pth \
  bash scripts/route_a_v4_9_1_dynamic_rviz_host.sh forest
```

The checkpoint override must still match the frozen release SHA-256.  This
prevents a superficially identical launch from silently running another model.
