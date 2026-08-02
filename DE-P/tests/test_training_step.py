import time
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from policy.state_transform import state_body2world
from tests.baseline_helpers import finite_tensor, get_single_map_loss, load_limited_dataset, make_runtime_network, seed_everything


class OneTrainingStepTests(unittest.TestCase):
    def test_one_batch_forward_backward_and_step(self):
        seed_everything(0)
        dataset, data_load_seconds = load_limited_dataset(mode="train", image_count=10)
        seed_everything(0)
        depth_np, pos_np, rot_np, obs_np, map_id_value = dataset[0]

        seed_everything(0)
        model, device = make_runtime_network()
        model.train()
        loss_module = get_single_map_loss()
        self.assertEqual(loss_module.device.type, device.type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, fused=True)

        depth = torch.from_numpy(depth_np).unsqueeze(0).to(device)
        pos = torch.from_numpy(pos_np).unsqueeze(0).to(device)
        rot = torch.from_numpy(rot_np).unsqueeze(0).to(device)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        map_id = torch.tensor([map_id_value], dtype=torch.long, device=device)
        traj_num = 15

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs[:, 6:9], obs[:, 0:3], obs[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)
        endstate, score = model.inference(depth, obs)
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(traj_num, 9)
        score_flat = score.reshape(traj_num)
        pos_expanded = pos.repeat_interleave(traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(traj_num, dim=0)
        goal_w = goal_w.repeat_interleave(traj_num, dim=0)
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded,
            rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)
        smooth, safety, guidance = loss_module(start_state_w, end_state_w, goal_w, map_id)
        trajectory_loss = (smooth + safety + guidance).mean()
        score_label = (smooth + safety + guidance).detach()
        score_loss = F.smooth_l1_loss(score_flat, score_label)
        total = trajectory_loss + score_loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(finite_tensor(gradient) for gradient in gradients))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_start
        peak_memory = (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda"
                       else "NOT MEASURED IN CODEX SANDBOX")
        batch_norm_training = [module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]

        metrics = {
            "device": str(device),
            "data_load_seconds": round(data_load_seconds, 6),
            "forward_seconds": round(forward_seconds, 6),
            "backward_and_step_seconds": round(backward_seconds, 6),
            "trajectory_loss": float(trajectory_loss.detach().cpu()),
            "score_loss": float(score_loss.detach().cpu()),
            "smooth": float(smooth.mean().detach().cpu()),
            "safety": float(safety.mean().detach().cpu()),
            "guidance": float(guidance.mean().detach().cpu()),
            "peak_gpu_memory_bytes": peak_memory,
            "batch_norm_all_training": all(batch_norm_training),
            "gradient_tensor_count": len(gradients),
        }
        print("BASELINE_TRAIN_STEP", metrics)
        self.assertTrue(finite_tensor(total))
        self.assertTrue(all(batch_norm_training))


if __name__ == "__main__":
    unittest.main()
