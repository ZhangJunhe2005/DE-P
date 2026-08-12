# Route A V4.8 recovery-capacity shakedown preparation

Status: `READY_FOR_USER_LAUNCH`; training has not been started by this change.

V4.8 is a bounded five-epoch diagnostic shakedown. It reuses the complete,
frozen V4.7 raw and derived datasets, and the preparation entry resolves the
newest completed V4.7 `checkpoints/best.pth` with a strict checkpoint hash.
No dataset generation, post-training H5 conversion, production activation, or
qualification Gate is introduced.

The frozen contract is:

- training contract:
  `route_a_static_yopo_training_v4_8_recovery_capacity_shakedown_v1`
- experiment role: `recovery_capacity_five_epoch_shakedown`
- observation: `route_a_recovery_state_v4_8`, seed `84710` (the V4.7
  original-YOPO sampler seed, preserving the ordinary 80% byte-for-byte)
- validation: `route_a_v4_8_recovery_capacity_v1`, ungated
- epochs: `5`; learning rates: backbone `5e-8`, candidate `1e-6`, score `2e-6`
- V4.5.10 objective disabled; V4.8 inherits its values and adds bounded
  safe-sector coverage (`0.05` weight; rays `0.5..4.0 m`, 12 samples;
  openness center/softness `0.45/0.10 m`; matching-lattice axial progress
  target `3.0 m`; shortfall softness `0.35 m`)
- qualification Gate count: `0`

The RViz entry retains the V4.7 fixed scene fixture and
`v4_7_balanced_dynamic` runtime profile. Only the pillar scene enables
`bounded_scan_v3`, matching the prior fixed comparison protocol.

User launch commands:

```bash
bash scripts/route_a_v4_8_train_host.sh --dry-run
bash scripts/route_a_v4_8_train_host.sh
bash scripts/route_a_v4_8_training_status.sh
bash scripts/route_a_v4_8_summary.sh
bash scripts/route_a_v4_8_rviz_host.sh pillar
```
