# Route A V4.2.3 — EGO-inspired quintic feasibility

This version does not copy EGO-Planner's B-spline control-point equations into
YOPO.  The two representations are different.  It transfers the physical
contract and adapts it to the exact quintic used by DE-P.

Reference implementation:

- `bspline_optimizer.cpp::calcFeasibilityCost`: differentiable velocity and
  acceleration violation costs;
- `uniform_bspline.cpp::checkFeasibility`: feasibility ratio
  `max(v/v_max, sqrt(a/a_max))`;
- `uniform_bspline.cpp::lengthenTime` and
  `planner_manager.cpp::reparamBspline`: temporal stretching and refitting.

Official source: <https://github.com/ZJU-FAST-Lab/ego-planner>

## DE-P adaptation

For every one of the 15 candidates, the exact quintic is sampled over the full
1.7 s horizon.  Let

```text
r_v(t) = ||v(t)|| / 6
r_a(t) = ||a(t)|| / 6
r_time = max(1, max_t r_v(t), sqrt(max_t r_a(t)))
```

Training receives gradients from:

1. dense vector-norm barriers beginning at 90% of the hardware limit;
2. per-axis barriers, corresponding to EGO's directional feasibility terms;
3. trajectory-relative normal acceleration, with an earlier 75% soft margin
   at high speed, so sharp bends learn to slow down rather than stop;
4. the continuous time-dilation residual `(r_time - 1)^2`;
5. mean, worst-third CVaR, and best-three coverage aggregation across all
   candidates.

The continuous candidate feasibility cost is detached only when used as a
score label.  Its direct loss remains differentiable and reaches terminal
position, velocity and acceleration in the trajectory head.  Consequently an
all-unsafe batch still has a useful gradient even though binary safe-vs-unsafe
ranking has no pair.

## Runtime boundary

Runtime score values are not modified.  A checkpoint is qualified only when
validation selects hardware-feasible trajectories and has at least three
feasible candidates on average.  The runtime 6 m/s and 6 m/s^2 shield remains
the final hardware backstop; frequent projection or braking is a failed
training/deployment Gate, not an accepted operating mode.
