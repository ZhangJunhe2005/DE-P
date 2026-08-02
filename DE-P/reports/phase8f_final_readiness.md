# Phase 8F final readiness

Final status: **FAIL** (`perception_ready: false`).

The complete frozen train+valid Gate passed, but the single permitted test run
failed `never_observed_false_matches = 0`: it produced 13 matches. All 13 are
recorded with sequence, frame, actor and track IDs in
`phase8f_test_once_quality.json`. The implementation hash was identical for
train/valid and test: `fc9c5370a1da929cfb8cf267b0c11176ec15e3aeb291b8e5117ec6fe753abf04`.

| Split | Visible P/R | Active P/R | Never-observed matches |
|---|---:|---:|---:|
| train | 0.808/0.464 | 0.784/0.427 | 0 |
| valid | 0.909/0.534 | 0.887/0.473 | 0 |
| test (once) | 0.914/0.545 | 0.912/0.497 | 13 |

No test-driven parameter change or second test run was performed. Therefore
Phase 8F does **not** authorize candidate scoring, ranking loss, gated fusion,
estimated-context training, production launcher changes, or long training.
The next step requires a separately approved phase and a new untouched
evaluation protocol; this report does not declare `next_allowed_phase`.
