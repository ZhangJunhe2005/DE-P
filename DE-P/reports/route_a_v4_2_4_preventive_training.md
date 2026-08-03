# Route-A V4.2.4 Preventive Safety Training

## Root cause established before modification

- V4.2.3 kept a 10 m local objective in a 1.7 s segment. The implied 5.88
  m/s mean progress leaves almost no 6 m/s hardware envelope for lateral
  avoidance or braking.
- Its final validation selected a hardware-unsafe candidate on about 94% of
  frames and produced only about 0.49 hard-feasible candidates out of 15 on
  average. Runtime projection and the safety shield therefore carried most of
  the closed-loop feasibility burden.
- The frozen V4.2 static corpus contains near-obstacle observations, but the
  original loss does not emphasize them and the desired clearance remains a
  collision-scale objective rather than an anticipation-scale objective.
- Moving actor futures cannot be inferred reliably by a single-frame static
  checkpoint. Route A therefore keeps motion prediction in a causal,
  deterministic safety layer rather than training with simulator identity or
  future ground truth.

## V4.2.4 changes

- Seven metre local objective; 3.5 m endpoint and 2 m/s speed anti-hover gates
  remain mandatory.
- Speed-aware desired static clearance: 1.2 m at rest, increasing by 0.12 s
  times speed and capped at 2.0 m.
- Differentiable clearance mean/CVaR/best-candidate coverage loss and
  clear-vs-near score ranking.
- Near-field depth occupancy raises only the loss weight, capped at 4x. It does
  not provide actor identity or labels to the CNN.
- Kinodynamic loss weight increased and its soft speed/normal-acceleration
  margins moved earlier.
- FP32 training is mandatory after the initial AMP dry-run dropped four of
  eight optimizer steps. The FP32 rerun applied 8/8 steps with zero overflows.
- `dynamic_safety` runtime mode extrapolates confirmed causal dynamic tracks
  using measured position/velocity only. It does not feed attention into the
  statically trained CNN and does not use simulator ground truth.

## Frozen data decision

No raw dataset regeneration is required. V4.2.4 reads the existing immutable
300,000 train and 60,000 validation frames. A separate moving-actor learning
claim is not introduced.

## Acceptance gate

- hardware unsafe selection <= 5%
- static unsafe selection <= 3%
- mean feasible candidates >= 3/15
- mean selected time dilation <= 1.02
- anticipatory unsafe selection <= 20%
- mean selected clearance >= 1.2 m
- mean clear candidates >= 3/15
- endpoint distance >= 3.5 m, endpoint speed >= 2 m/s, hover <= 20%

The checkpoint must pass every gate before `best.pth` is published. Otherwise
it remains diagnostic and must not replace the V4.2.2/V4.2.3 comparison set.
