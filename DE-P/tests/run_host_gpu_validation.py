"""One-batch DE-P GPU validation for a normal Ubuntu host terminal."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dep_network import DepNetwork
from policy.models.backbone import DepBackbone
from policy.state_transform import state_body2world
from tests.baseline_helpers import finite_tensor, get_single_map_loss, load_limited_dataset, reset_lattice_singleton, seed_everything


def fail(message: str, code: int = 2) -> int:
    print(f"GPU VALIDATION FAILED: {message}", file=sys.stderr)
    return code


def benchmark_inference(model, depth, observation, warmup=20, iterations=100):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model.inference(depth, observation.clone())
        torch.cuda.synchronize()
        elapsed_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.inference(depth, observation.clone())
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
    return statistics.fmean(elapsed_ms), float(np.percentile(elapsed_ms, 95))


def main() -> int:
    print("torch_version:", torch.__version__)
    print("torch_cuda_build:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return fail(
            "CUDA is unavailable in this host process. Run nvidia-smi and verify that the yopo "
            "environment is using its existing CUDA-enabled PyTorch build."
        )

    seed_everything(0)
    device = torch.device("cuda:0")
    print("gpu_name:", torch.cuda.get_device_name(device))
    print("compute_capability:", torch.cuda.get_device_capability(device))

    checkpoint = ROOT / "saved" / "DEP_0" / "epoch10.pth"
    if not checkpoint.is_file():
        return fail(f"checkpoint not found: {checkpoint}")

    reset_lattice_singleton()
    model = DepNetwork().to(device)
    state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        return fail(f"strict checkpoint mismatch: {incompatible}")
    print("checkpoint_strict_load: PASS", checkpoint)

    backbone = DepBackbone(64).to(device).eval()
    fixed_depth = torch.zeros(1, 1, 96, 160, device=device)
    fixed_observation = torch.zeros(1, 9, device=device)
    with torch.inference_mode():
        feature = backbone(fixed_depth)
        endstate, score = model.eval().inference(fixed_depth, fixed_observation.clone())
    if tuple(feature.shape) != (1, 64, 3, 5) or not finite_tensor(feature):
        return fail(f"invalid GPU backbone output: shape={tuple(feature.shape)}")
    if tuple(endstate.shape) != (1, 9, 3, 5) or tuple(score.shape) != (1, 3, 5):
        return fail(f"invalid GPU inference shapes: {tuple(endstate.shape)}, {tuple(score.shape)}")
    if not finite_tensor(endstate) or not finite_tensor(score):
        return fail("GPU inference contains NaN or Inf")
    print("gpu_forward: PASS")

    average_ms, p95_ms = benchmark_inference(model, fixed_depth, fixed_observation)

    dataset, data_load_seconds = load_limited_dataset(mode="train", image_count=10)
    seed_everything(0)
    depth_np, pos_np, rot_np, obs_np, map_id_value = dataset[0]
    loss_module = get_single_map_loss()
    if loss_module.device.type != "cuda":
        return fail(f"loss module was created on {loss_module.device}, not CUDA")

    depth = torch.from_numpy(depth_np).unsqueeze(0).to(device)
    pos = torch.from_numpy(pos_np).unsqueeze(0).to(device)
    rot = torch.from_numpy(rot_np).unsqueeze(0).to(device)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    map_id = torch.tensor([map_id_value], dtype=torch.long, device=device)
    traj_num = 15

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, fused=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs[:, 6:9], obs[:, 0:3], obs[:, 3:6])
    start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)
    train_endstate, train_score = model.inference(depth, obs)
    endstate_flat = train_endstate.permute(0, 2, 3, 1).reshape(traj_num, 9)
    score_flat = train_score.reshape(traj_num)
    pos_expanded = pos.repeat_interleave(traj_num, dim=0)
    rot_expanded = rot.repeat_interleave(traj_num, dim=0)
    start_state_w = start_state_w.repeat_interleave(traj_num, dim=0)
    goal_w = goal_w.repeat_interleave(traj_num, dim=0)
    end_pos_w, end_vel_w, end_acc_w = state_body2world(
        pos_expanded, rot_expanded,
        endstate_flat[:, 0:3], endstate_flat[:, 3:6], endstate_flat[:, 6:9],
    )
    end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)
    smooth, safety, guidance = loss_module(start_state_w, end_state_w, goal_w, map_id)
    trajectory_loss = (smooth + safety + guidance).mean()
    score_label = (smooth + safety + guidance).detach()
    score_loss = F.smooth_l1_loss(score_flat, score_label)
    total = trajectory_loss + score_loss
    total.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(finite_tensor(gradient) for gradient in gradients):
        return fail("one-batch training produced a missing, NaN, or Inf gradient")
    for value, label in ((smooth, "smooth"), (safety, "safety"), (guidance, "guidance"),
                         (score_loss, "score_loss"), (total, "total")):
        if not finite_tensor(value):
            return fail(f"{label} contains NaN or Inf")
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    if peak_memory <= 0:
        return fail("reported peak CUDA memory is not positive")

    summary = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "checkpoint": str(checkpoint),
        "backbone_shape": list(feature.shape),
        "endstate_shape": list(endstate.shape),
        "score_shape": list(score.shape),
        "limited_data_load_seconds": data_load_seconds,
        "smooth_mean": float(smooth.mean().detach().cpu()),
        "safety_mean": float(safety.mean().detach().cpu()),
        "guidance_mean": float(guidance.mean().detach().cpu()),
        "score_loss": float(score_loss.detach().cpu()),
        "gradient_tensor_count": len(gradients),
        "peak_gpu_memory_bytes": peak_memory,
        "inference_average_ms": average_ms,
        "inference_p95_ms": p95_ms,
    }
    print("HOST_GPU_VALIDATION_RESULT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

