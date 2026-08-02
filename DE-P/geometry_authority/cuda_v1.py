"""CUDA reference for the Simulator Static Authority V1 contract."""

from __future__ import annotations

import numpy as np
import torch

from .static_v1 import (
    BACKEND_VERSION,
    CONTACT_TOLERANCE_M,
    DEFAULT_UAV_RADIUS_M,
    EMPTY_GAP_M,
    CollisionBatch,
    StaticAuthorityMap,
)


class CudaStaticAuthorityMap:
    """Exact occupancy query using CUDA tensors from the canonical artifact."""

    def __init__(self, root, device="cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the GPU authority backend")
        self.reference = StaticAuthorityMap(root)
        self.device = torch.device(device)
        self.origin = torch.as_tensor(
            self.reference.origin, dtype=torch.float64, device=self.device
        )
        self.bounds_max = torch.as_tensor(
            self.reference.bounds_max,
            dtype=torch.float64,
            device=self.device,
        )
        occupied = torch.as_tensor(
            self.reference.occupied_indices,
            dtype=torch.float64,
            device=self.device,
        )
        self.voxel_min = self.origin + occupied * self.reference.resolution
        self.voxel_max = self.voxel_min + self.reference.resolution
        self.occupied = occupied.to(torch.int64)

    def query(self, centers, radius=DEFAULT_UAV_RADIUS_M, chunk_size=2048):
        centers = torch.as_tensor(
            np.asarray(centers), dtype=torch.float64, device=self.device
        )
        single = centers.ndim == 1
        centers = centers.reshape(-1, 3)
        radii = torch.as_tensor(
            np.asarray(radius), dtype=torch.float64, device=self.device
        ).expand(len(centers))
        if not torch.isfinite(centers).all() or not torch.isfinite(radii).all():
            raise ValueError("query must be finite")
        if torch.any(radii < 0):
            raise ValueError("radius must be non-negative")
        oob = torch.any(
            (centers - radii[:, None] < self.origin - CONTACT_TOLERANCE_M)
            | (
                centers + radii[:, None]
                > self.bounds_max + CONTACT_TOLERANCE_M
            ),
            dim=1,
        )
        collision = oob.clone()
        gap = torch.full(
            (len(centers),), EMPTY_GAP_M,
            dtype=torch.float64, device=self.device,
        )
        nearest = torch.full(
            (len(centers), 3), -1, dtype=torch.int64, device=self.device
        )
        if len(self.occupied):
            for start in range(0, len(centers), chunk_size):
                stop = min(len(centers), start + chunk_size)
                delta = torch.maximum(
                    torch.maximum(
                        self.voxel_min[None]
                        - centers[start:stop, None],
                        centers[start:stop, None]
                        - self.voxel_max[None],
                    ),
                    torch.zeros((), dtype=torch.float64, device=self.device),
                )
                distance = torch.linalg.vector_norm(delta, dim=2)
                values = torch.min(distance, dim=1).values
                choices = torch.argmax(
                    (
                        distance
                        <= values[:, None] + CONTACT_TOLERANCE_M
                    ).to(torch.int8),
                    dim=1,
                )
                gap[start:stop] = values - radii[start:stop]
                nearest[start:stop] = self.occupied[choices]
            collision |= gap <= CONTACT_TOLERANCE_M
        nearest[~collision] = -1
        voxel_bounds = torch.full(
            (len(centers), 2, 3), torch.nan,
            dtype=torch.float64, device=self.device,
        )
        valid = torch.all(nearest >= 0, dim=1)
        voxel_bounds[valid, 0] = (
            self.origin + nearest[valid] * self.reference.resolution
        )
        voxel_bounds[valid, 1] = (
            voxel_bounds[valid, 0] + self.reference.resolution
        )
        count = torch.full(
            (len(centers),), len(self.occupied),
            dtype=torch.int64, device=self.device,
        )
        arrays = [
            value.detach().cpu().numpy()
            for value in (
                collision, gap, nearest, voxel_bounds, oob, count
            )
        ]
        result = CollisionBatch(
            *arrays, BACKEND_VERSION,
            self.reference.metadata["artifact_manifest_hash"],
        )
        if not single:
            return result
        return CollisionBatch(
            *(value[0] if isinstance(value, np.ndarray) else value
              for value in result.__dict__.values())
        )
