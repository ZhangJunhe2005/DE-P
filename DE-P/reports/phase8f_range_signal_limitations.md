# Phase 8F range-signal limitations

The deterministic range signal is usable: full train/valid tracking Gates pass
and range-seed recall is non-zero on held-out maps. It is not sufficient for a
final production PASS under the current protocol because the one-time test run
contains 13 matches to actors classified as never observed. These occur in
three test sequences and are not no-target failures; their details are frozen
in the test report. The dataset does not contain per-pixel actor instance
masks, so pixel precision/recall relies on a conservative geometry/depth proxy.

No neural foreground refiner was introduced. No threshold or association rule
was changed after observing test. A future attempt must use a newly approved
protocol rather than rerunning or tuning against this test split.
