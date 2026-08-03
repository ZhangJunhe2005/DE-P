# Route-A V4.2.3 local-planner deadlock root cause

## Finding

The observed stop is not one defect and cannot be repaired by reducing the
runtime clearance threshold.  It is the expected terminal state of three
contracts that do not close a recovery loop:

1. The 15 primitive anchors cover only five horizontal directions centred at
   approximately -36, -18, 0, 18 and 36 degrees.  The bounded network residual
   adds at most 15 degrees, so the intended translational proposal envelope is
   forward-only (approximately +/-51 degrees).  Position radius is nonnegative.
2. The only V1 action when all candidates are rejected is a finite-horizon
   braking trajectory.  Braking changes neither the camera sector nor the
   candidate action set, so the same observation can reproduce zero candidates
   indefinitely.
3. The sensor is a forward monocular depth camera.  A safe reverse/U-turn
   cannot be inferred from the current frame alone because rear free space is
   unobserved.

## Dataset audit

The audit sampled every twelfth derived depth row, covering 25,000 train and
5,000 validation frames across all sequences.

| Metric | Train | Validation |
|---|---:|---:|
| Minimum depth below 0.65 m | 0.664% | 0.760% |
| Minimum depth below 1.0 m | 3.556% | 3.600% |
| >=25% pixels below 2 m | 3.056% | 3.740% |
| >=50% pixels below 2 m | 0.180% | 0.240% |
| >=50% pixels below 3 m | 0.860% | 1.120% |

The actual training observation adapter (`route_a_wide_state_v1`) was also
replayed for all 300,000/60,000 identities:

| Metric | Train | Validation |
|---|---:|---:|
| Absolute goal yaw above 45 degrees | 2.473% | 2.463% |
| Absolute goal yaw above 51 degrees | 1.086% | 1.090% |
| Goal behind the vehicle | 0% | 0% |
| Backward body-X velocity | 0.00067% | 0.00167% |

All raw UAV references are continuously certified, monotonic forward paths.
There is no closed-loop demonstration of approaching a dead end, braking,
changing camera heading, and leaving through a different sector.

## Resolution boundary

More near-obstacle frames can help the policy avoid entering a trap earlier,
but they cannot add a reverse or camera-scan action to the existing 15-output
head.  Fully blocked frames also create an unsatisfiable anti-hover objective
when every representable translation is unsafe.

`deadlock_recovery_v1` therefore keeps normal translation under the network,
but changes the observation when the forward action set collapses:

1. require 15 consecutive zero-feasible replans;
2. brake within the existing 6 m/s and 6 m/s^2 limits;
3. if physically inside the observed collision floor, move to a bounded recent
   flown breadcrumb (1.2 m path history target);
4. yaw-scan at 45 deg/s toward the freer image side;
5. require five consecutive replans with feasible candidates before returning
   translation authority to the network.

This is a recovery controller, not a claim that the neural network learned a
U-turn.  A future preventive-training dataset should retain pre-deadlock frames
with lateral openings, but should not label impossible fully blocked
forward-only translations as learnable actions.
