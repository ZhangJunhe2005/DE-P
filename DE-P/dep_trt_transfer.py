"""
    将dep模型转换为Tensorrt
    prepare:
        1 pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2 git clone https://github.com/NVIDIA-AI-IOT/torch2trt
          cd torch2trt
          python setup.py install
"""

import os
import argparse
import json
import time
import numpy as np
import torch
from config.config import cfg
from policy.backbone_variant import BACKBONE_VARIANTS, resolve_backbone_variant
from policy.checkpoint_utils import load_dep_checkpoint
from policy.dep_network import DepNetwork


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--dir", type=str, default=None, help="output file name")
    parser.add_argument("--backbone-variant", choices=BACKBONE_VARIANTS, default=None,
                        help="override config/traj_opt.yaml backbone_variant")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit matching PyTorch checkpoint path")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    variant = resolve_backbone_variant(args.backbone_variant)
    print(f"Backbone variant: {variant}")
    checkpoint_dir = f"DEP_{args.trial}" if variant == "legacy" else f"DEP_corrected_{args.trial}"
    weight = (os.path.abspath(os.path.expanduser(args.checkpoint)) if args.checkpoint else
              os.path.join(base_dir, "saved", checkpoint_dir, f"epoch{args.epoch}.pth"))
    output = args.dir or ("dep_trt.pth" if variant == "legacy" else "dep_trt_corrected.pth")

    print("Loading Network...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = DepNetwork(backbone_variant=variant)
    load_dep_checkpoint(policy, weight, variant)
    policy = policy.to(device)
    policy.eval()

    # The inputs should be consistent with training
    depth = np.zeros(shape=[1, 1, 96, 160], dtype=np.float32)
    obs = np.zeros(shape=[1, 9, cfg["vertical_num"], cfg["horizon_num"]], dtype=np.float32)
    depth_in = torch.from_numpy(depth).to(device)
    obs_in = torch.from_numpy(obs).to(device)

    print("TensorRT Transfer...")
    try:
        from torch2trt import torch2trt
    except ImportError as exc:
        raise RuntimeError("TensorRT conversion skipped: torch2trt is not installed") from exc
    model_trt = torch2trt(policy, [depth_in, obs_in], fp16_mode=True)
    torch.save(model_trt.state_dict(), output)
    with open(output + ".metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump({"backbone_variant": variant, "source_checkpoint": os.path.abspath(weight)}, metadata_file, indent=2)


    print("Evaluation...")
    # Warm Up...
    traj_trt, score_trt = model_trt(depth_in, obs_in)
    traj, score = policy(depth_in, obs_in)
    torch.cuda.synchronize()

    # PyTorch Latency
    torch_start = time.time()
    traj, score = policy(depth_in, obs_in)
    torch.cuda.synchronize()
    torch_end = time.time()

    # TensorRT Latency
    trt_start = time.time()
    traj_trt, score_trt = model_trt(depth_in, obs_in)
    torch.cuda.synchronize()
    trt_end = time.time()

    # Transfer Error
    traj_error = torch.mean(torch.abs(traj - traj_trt))
    score_error = torch.mean(torch.abs(score - score_trt))

    print(f"Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, "
          f"TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, "
          f"Transfer Trajectory Error: {traj_error.item():.6f},"
          f"Transfer Score Error: {score_error.item():.6f}")
