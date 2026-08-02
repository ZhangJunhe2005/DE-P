# Phase 8F causal range-image foreground

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

- Full train component precision/recall: 0.329/0.212.
- Full valid component precision/recall: 0.325/0.202.
- Full valid complete perception mean/P95: 37.8/58.7 ms (10 Hz period: 100 ms).
- History is bounded to 4 frames; world memory remains capped at 200000 voxels.
- New Phase 8F tests: 25/25; combined dynamic regression selection: 94/94.

## Known limitations

Small targets remain hardest: full-valid active recall for projected area below
25 pixels is 0.090.
The one-time frozen test failed only the never-observed hard constraint (13
matches across three sequences). Per protocol, this result was not used to
tune or rerun the model. Live ROS/Simulator validation was not available;
Noetic import and callback binding passed without a Master.
