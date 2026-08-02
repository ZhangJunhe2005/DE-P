from __future__ import annotations

import contextlib
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT.parent / "dataset"


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numeric_image_key(name: str) -> int:
    return int(Path(name).stem.split("_")[1])


def load_limited_dataset(mode: str = "train", image_count: int = 10):
    """Construct the real DEPDataset while exposing only a few real files."""
    import policy.dep_dataset as dataset_module

    data_dir = DATASET_ROOT.resolve()
    map_dir = (DATASET_ROOT / "0").resolve()
    real_scandir = os.scandir
    real_listdir = os.listdir
    real_loadtxt = np.loadtxt

    map_entries = [entry for entry in real_scandir(data_dir) if entry.is_dir() and entry.name == "0"]
    image_names = sorted(
        [name for name in real_listdir(map_dir) if Path(name).suffix == ".png"],
        key=_numeric_image_key,
    )[:image_count]

    def limited_scandir(path):
        if Path(path).resolve() == data_dir:
            return iter(map_entries)
        return real_scandir(path)

    def limited_listdir(path):
        if Path(path).resolve() == map_dir:
            return list(image_names)
        return real_listdir(path)

    def limited_loadtxt(path, *args, **kwargs):
        values = real_loadtxt(path, *args, **kwargs)
        if Path(path).resolve() == (DATASET_ROOT / "pose-0.csv").resolve():
            return values[:image_count]
        return values

    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(dataset_module.os, "scandir", side_effect=limited_scandir))
        stack.enter_context(mock.patch.object(dataset_module.os, "listdir", side_effect=limited_listdir))
        stack.enter_context(mock.patch.object(dataset_module.np, "loadtxt", side_effect=limited_loadtxt))
        dataset = dataset_module.DEPDataset(mode=mode, val_ratio=0.1)
    return dataset, time.perf_counter() - start


def reset_lattice_singleton():
    from policy.primitive import LatticePrimitive

    LatticePrimitive._instance = None


def make_cpu_network():
    """Build a CPU network even on hosts where CUDA is visible."""
    from policy.dep_network import DepNetwork

    reset_lattice_singleton()
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        model = DepNetwork().cpu()
    return model


def make_runtime_network():
    from policy.dep_network import DepNetwork

    reset_lattice_singleton()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DepNetwork().to(device), device


@lru_cache(maxsize=1)
def get_single_map_loss():
    """Build the production DEPLoss using only pointcloud-0.ply."""
    from loss.loss_function import DEPLoss
    from loss.safety_loss import SafetyLoss

    ply = str(DATASET_ROOT / "pointcloud-0.ply")
    with mock.patch.object(SafetyLoss, "read_sorted_ply_files", return_value=[ply]):
        return DEPLoss()


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())
