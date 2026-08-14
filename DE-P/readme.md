# DE-P

The frozen, reproducible avoidance release is Route-A V4.9.1 recovery
subgoal lifecycle V6.  Its source, exact V4.8.3 policy checkpoint, runtime
contract, verification command, ROS workspace layout and generated map-asset
requirements are documented in
[`docs/route_a_v4_9_1_reproducibility.md`](docs/route_a_v4_9_1_reproducibility.md).

After cloning, the algorithm-only verification does not require ROS master or
the generated map dataset:

```bash
conda activate yopo
bash scripts/verify_route_a_v4_9_1_checkout.sh
```

TODO:

1. Enable AMP (Automatic Mixed Precision) training. 
Ensure that the CUDA versions of the virtual environment and system are consistent.
