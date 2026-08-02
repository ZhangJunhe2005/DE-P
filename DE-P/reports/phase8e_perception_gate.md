# Phase 8E perception Gate status

## Decision

**FAIL — stop before Phase 8E items 6–10.**

The causal perception implementation materially reduces the original no-target
failure, but the frozen train/valid development subset does not satisfy all hard
precision/recall requirements. Therefore candidate/selection decomposition,
safety-first score labels, ranking loss, gated fusion, estimated-context
shakedown, and production-launcher updates were not started.

No test-split result was used for tuning in this phase.

## Corrected evaluator semantics

- Visible GT: active, inside image, not occluded, visibility greater than zero.
- Visible precision: only estimated tracks currently projected into the image.
- Active tracking: visible actors plus previously observed actors inside the
  configured occlusion window.
- Correctly retained occluded tracks are excluded from visible false positives.
- Every dynamic track in a no-target sequence remains a hard false positive.
- Never-observed actors are reported separately.
- No-target attention statistics are hard metrics; they are not hidden by the
  occlusion correction.

## Root-cause diagnosis before the fix

The complete no-target diagnostic scanned 36 sequences / 2,160 frames and
recorded 35,233 false dynamic-track records (16.31 per frame on average).

- Static-map oracle proximity within 0.25 m: 0.997649 mean point fraction.
- Dominant mechanism classification: KF response to drifting centroids 75.70%,
  viewpoint centroid drift 20.96%, split/merge 2.40%, wrong association 0.89%,
  innovation-only 0.057%.
- Geometry: compact/sparse 42.60%, other 35.45%, wall/plane 16.54%, ground 5.42%.

The static-map oracle exists only in the offline diagnostic and is not imported
or used by runtime perception.

## Implemented perception changes

- Causal world-frame temporal background and observed-free-space voxel model.
- Per-stream reset, time-gap reset, age expiry, and bounded capacity.
- No future frames, actor GT, test parameters, or static-map oracle at runtime.
- Robust trimmed centroid and extent-dependent covariance floor.
- Cluster extent/eigenvalue/shape/voxel descriptors.
- One-to-one Hungarian association with Euclidean, Mahalanobis, bbox/voxel
  overlap, point-count ratio, shape, and physical-motion gates.
- Linear constant-velocity KF with velocity covariance floor.
- Robust median velocity history, acceleration/outlier rejection, split/merge
  handling, and coherent-motion evidence.
- Explainable bounded track confidence; attention requires confirmed dynamic
  state and confidence threshold.
- Visible/active/occluded/no-target evaluator v2.

## Best retained train/valid development-subset result

Source: `reports/phase8e_perception_preview_free_space_dbscan05.json`.
The subset is deterministic: one sequence per scenario from train and valid.

| Metric | Train | Valid | Hard requirement |
|---|---:|---:|---:|
| visible precision | 0.7368 | 0.2500 | >= 0.5 |
| visible recall | 0.1931 | 0.0290 | >= 0.2 |
| active precision | 0.7222 | 0.2727 | >= 0.5 |
| active recall | 0.2167 | 0.0313 | >= 0.2 |
| position RMSE (m) | 0.3451 | 0.3808 | <= 1.0 |
| velocity RMSE (m/s) | 0.5785 | 0.2557 | <= 1.5 |
| no-target FP frame fraction | 0.1167 | 0.0000 | <= 0.1 |
| no-target attention nonzero fraction | 0.0833 | 0.0000 | <= 0.05 |
| attention IoU | 0.2859 | 0.2396 | >= 0.05 |
| never-observed false matches | 0 | 0 | 0 |

The retained version favors safety over apparent recall. Two rejected
experiments are recorded separately:

- Local foreground dilation improved recall but raised no-target FP to
  0.32–0.55.
- Persistent historical geometry evidence improved recall but raised no-target
  FP to 0.28–0.43.

Both were reverted and are not present in the retained implementation.

## Test status

- Dynamic-module CPU suite: 61 tests passed.
- New focused tests: temporal causality/capacity, static suppression,
  split/merge, velocity outlier rejection, and occlusion semantics all pass.
- Full generic discovery in the sandbox stops at `test_checkpoint_paths`
  because `rospy` is unavailable there; this is an environment import failure,
  not a dynamic-module regression. No ROS master was started.

## Remaining blocker

The retained causal free-space seeds are precise but too sparse on held-out
valid maps, especially when actors are already visible near sequence start.
Simply lowering minimum cluster size or expanding seeds also promotes static
free-space artifacts. The next perception iteration needs a bounded image-space
or range-image temporal component that preserves full moving-object silhouettes
while keeping the observed-free-space hard negative check.

Until this is solved and the full frozen train/valid/test perception Gate passes:

- do not run estimated-context shakedown;
- do not implement or evaluate score/fusion changes against noisy estimated attention;
- do not update the production launcher;
- do not start long production training.
