"""Read-only map-type views over a frozen static YOPO dataset."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class StaticYOPOMapTypeSubsetV1(Dataset):
    """Select complete samples by authoritative map type without copying data."""

    def __init__(self, base, map_type_by_map_id, selected_map_type):
        self.base = base
        self.split = str(base.split)
        self.selected_map_type = str(selected_map_type)
        available = sorted(set(map_type_by_map_id.values()))
        if self.selected_map_type not in available:
            raise ValueError(
                f"unknown map type {self.selected_map_type!r}; "
                f"available={available}"
            )
        map_ids = np.asarray(base.arrays["map_id"])
        keep = np.asarray([
            map_type_by_map_id[int(map_id)] == self.selected_map_type
            for map_id in map_ids
        ], dtype=bool)
        self.indices = np.flatnonzero(keep)
        if not len(self.indices):
            raise RuntimeError(
                f"empty {self.split} subset for {self.selected_map_type}"
            )
        # Downstream contracts only inspect map_id.  Keep this one indexed
        # vector rather than materializing all string/path mmap fields.
        self.arrays = {"map_id": map_ids[self.indices]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[int(self.indices[index])]
