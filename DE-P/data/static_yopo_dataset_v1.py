"""Read-only mmap dataset for the frozen P0 static YOPO reference view."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.static_yopo_manifest_v1 import validate_batch_allowlist
from data.static_yopo_preprocessing_v1 import DEFAULT_STATIC_YOPO_PREPROCESSOR_V1


class StaticYOPODatasetV1(Dataset):
    _FIELDS = (
        "sample_id", "sequence_id", "frame_index", "source_split", "map_uuid",
        "suite", "depth_path", "depth_hash", "observation", "position_world",
        "quaternion_xyzw", "map_id",
    )

    def __init__(self, root, split="train", preprocessor=None, mmap_cache_size=8):
        if split not in {"train", "validation"}:
            raise ValueError("only train/validation sample contents may be opened")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.index_root = self.root / "indices" / split
        self.arrays = {
            field: np.load(self.index_root / f"{field}.npy", mmap_mode="r")
            for field in self._FIELDS
        }
        lengths = {len(value) for value in self.arrays.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"index field length mismatch: {lengths}")
        self.preprocessor = preprocessor or DEFAULT_STATIC_YOPO_PREPROCESSOR_V1
        self.mmap_cache_size = int(mmap_cache_size)
        if self.mmap_cache_size < 1:
            raise ValueError("mmap_cache_size must be positive")
        self._depth_cache = OrderedDict()

    def __len__(self):
        return len(self.arrays["sample_id"])

    @staticmethod
    def _text(value):
        return bytes(value).decode("utf-8")

    def _depth_array(self, relative_path):
        path = str((self.root / relative_path).resolve())
        if path in self._depth_cache:
            value = self._depth_cache.pop(path)
            self._depth_cache[path] = value
            return value
        value = np.load(path, mmap_mode="r")
        if value.dtype != np.float32 or value.ndim != 3 or value.shape[1:] != (96, 160):
            raise RuntimeError(f"invalid frozen static depth array: {path} {value.shape} {value.dtype}")
        self._depth_cache[path] = value
        while len(self._depth_cache) > self.mmap_cache_size:
            _, old = self._depth_cache.popitem(last=False)
            mmap = getattr(old, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return value

    def __getitem__(self, index):
        frame = int(self.arrays["frame_index"][index])
        relative = self._text(self.arrays["depth_path"][index])
        depth = self.preprocessor(self._depth_array(relative)[frame])
        quat = np.asarray(self.arrays["quaternion_xyzw"][index], dtype=np.float32)
        x, y, z, w = quat
        rotation = np.asarray([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        batch = {
            "depth": torch.from_numpy(depth),
            "observation": torch.from_numpy(
                np.array(self.arrays["observation"][index], dtype=np.float32, copy=True)
            ),
            "position_world": torch.from_numpy(
                np.array(self.arrays["position_world"][index], dtype=np.float32, copy=True)
            ),
            "rotation_world_from_body": torch.from_numpy(rotation),
            "map_id": torch.tensor(int(self.arrays["map_id"][index]), dtype=torch.long),
            "sample_id": self._text(self.arrays["sample_id"][index]),
        }
        validate_batch_allowlist(batch)
        return batch
