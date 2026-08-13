# Route A V4.8.5 Recovery Freeze

## Frozen boundary

This freeze combines the unchanged V4.8.3 epoch-2 static YOPO checkpoint with
the V4.8.5 scene-agnostic, motion-verified deadlock-recovery handoff. It does
not retrain the network and does not modify candidate geometry, static/dynamic
collision checks, temporal foreground extraction, tracking, Kalman prediction,
or dynamic actor occupancy.

The checkpoint remains a local ignored artifact and is identified by SHA-256:

`22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60`

## Recovery semantics

A five-replan same-sector candidate is now a provisional handoff. The scan
chain is reset only after the network produces at least 0.50 m of measured
translation within 1.20 s, or after the forward centre view improves by at
least 0.50 m for five consecutive replans. A failed provisional handoff brakes
and resumes the prior scan direction and offset at the next 60/90/120-degree
level. No map name or map type enters this decision.

## Evidence

- Automated checks: 81 passed, 1 expected xfail, 10 subtests passed.
- Cave, forest, pillar and wall launch preflights returned zero.
- The V4.8.3 checkpoint loaded strictly.
- Actor-free Pillar run `20260813T065954Z-pillar` arrived with zero collision.
- Two 16-actor Pillar runs (`20260813T070046Z-pillar` and
  `20260813T070239Z-pillar`) arrived with zero collision.
- In the first dynamic run, twelve provisional handoffs failed validation and
  resumed scanning; the run still arrived. This directly exercises the repair.
- Run `20260813T070345Z-pillar` reached an actual 116.4-degree scan but recorded
  three dynamic collisions and did not arrive. All three were dynamic rather
  than static collisions.
- The operator accepted V4.8.5 as materially more stable and less likely to
  deadlock.

## Qualification

The V4.8.3 static policy and V4.8.5 recovery handoff are frozen as the rollback
baseline for subsequent dynamic-avoidance work. Dynamic avoidance and overall
production qualification remain false. The unresolved scope is explicitly the
dynamic perception/prediction/selection path, not the frozen static policy or
the recovery handoff.

The machine-readable authority is
`configs/route_a_v4_8_5_recovery_freeze.yaml`.
