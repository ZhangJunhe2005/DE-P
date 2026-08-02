"""CUDA renderer for canonical occupancy and analytic dynamic spheres."""

from __future__ import annotations

import numpy as np
import torch


RENDERER_VERSION = "canonical_occupancy_cuda_raycast_v1"


class CudaAuthorityRenderer:
    def __init__(self, sensor, device="cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA renderer requested but CUDA is unavailable")
        self.device = torch.device(device)
        self.sensor = sensor
        height, width = int(sensor["height"]), int(sensor["width"])
        fx, fy, cx, cy = sensor["intrinsics"]
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        rays = torch.stack((
            torch.ones_like(u), (u-float(cx))/float(fx),
            (v-float(cy))/float(fy)), dim=-1)
        self.body_rays = torch.nn.functional.normalize(rays, dim=-1)
        self._map_key = None
        self._occupancy = None

    def _load_map(self, backend):
        key = (
            backend.map.metadata["map_uuid"],
            backend.map.metadata["occupancy_hash"],
        )
        if key != self._map_key:
            self._occupancy = torch.as_tensor(
                backend.map.flat, dtype=torch.bool, device=self.device)
            self._map_key = key

    def _render_static(self, backend, positions, yaws):
        self._load_map(backend)
        positions_array = np.asarray(positions, dtype=np.float32)
        yaws_array = np.asarray(yaws, dtype=np.float32)
        repeated_camera = (
            len(positions_array) > 1
            and np.all(positions_array == positions_array[0])
            and np.all(yaws_array == yaws_array[0]))
        positions = torch.as_tensor(
            positions_array, dtype=torch.float32, device=self.device)
        ray_positions = positions[:1] if repeated_camera else positions
        ray_yaws = torch.as_tensor(
            yaws_array[:1] if repeated_camera else yaws_array,
            dtype=torch.float32, device=self.device)
        c, s = torch.cos(ray_yaws), torch.sin(ray_yaws)
        rotation = torch.zeros(
            (len(ray_yaws), 3, 3),
            dtype=torch.float32, device=self.device)
        rotation[:, 0, 0] = c
        rotation[:, 0, 1] = s
        rotation[:, 1, 0] = -s
        rotation[:, 1, 1] = c
        rotation[:, 2, 2] = 1
        rays = torch.einsum("hwc,fcd->fhwd", self.body_rays, rotation)
        shape = rays.shape[:-1]
        maximum = float(self.sensor["max_depth_m"])
        static = torch.full(
            shape, maximum, dtype=torch.float32, device=self.device)
        active = torch.ones(shape, dtype=torch.bool, device=self.device)
        origin = torch.as_tensor(
            backend.map.origin, dtype=torch.float32, device=self.device)
        dimensions = torch.as_tensor(
            backend.map.dimensions, dtype=torch.int64, device=self.device)
        resolution = float(backend.map.resolution)
        yz = int(backend.map.dimensions[1] * backend.map.dimensions[2])
        z = int(backend.map.dimensions[2])
        for distance in torch.arange(
            float(self.sensor["ray_step_m"]), maximum + 1e-6,
            float(self.sensor["ray_step_m"]), device=self.device,
        ):
            points = ray_positions[:, None, None, :] + rays*distance
            indices = torch.floor(
                (points-origin)/resolution).to(torch.int64)
            inside = torch.all(
                (indices >= 0) & (indices < dimensions), dim=-1)
            valid = active & inside
            linear = (
                indices[..., 0]*yz + indices[..., 1]*z + indices[..., 2]
            ).clamp(0, self._occupancy.numel()-1)
            hit = valid & self._occupancy[linear]
            static[hit] = distance
            active &= inside & ~hit
        if repeated_camera:
            rays = rays.expand(len(positions), -1, -1, -1)
            static = static.expand(len(positions), -1, -1).clone()
        return positions, rays, static

    def render_with_actor_diagnostics(
        self, backend, positions, yaws, actor_positions, actor_radii,
        return_owner_map=False, return_actor_near_depth=False,
    ):
        """Render depth and report visibility provenance for every actor.

        ``projected`` is independent of static geometry.  ``visible`` requires
        the actor to be the nearest dynamic surface and nearer than static
        occupancy.  These definitions make full static occlusion distinguishable
        from field-of-view and range exits without metadata hints.
        """
        positions, rays, static = self._render_static(
            backend, positions, yaws)
        maximum = float(self.sensor["max_depth_m"])
        frame_count = len(positions)
        actor_positions = np.asarray(actor_positions, dtype=np.float32)
        actor_count = (
            int(actor_positions.shape[1])
            if actor_positions.ndim == 3 else 0
        )
        height, width = static.shape[-2:]
        if actor_count:
            centers = torch.as_tensor(
                actor_positions, dtype=torch.float32, device=self.device)
            radii = torch.as_tensor(
                actor_radii, dtype=torch.float32, device=self.device)
            offset = (
                centers[:, :, None, None, :]
                - positions[:, None, None, None, :]
            )
            projection = torch.sum(
                offset*rays[:, None, :, :, :], dim=-1)
            perpendicular2 = (
                torch.sum(offset*offset, dim=-1)
                - projection*projection
            )
            radius2 = radii[None, :, None, None] ** 2
            intersects = (
                (projection > 0) & (perpendicular2 <= radius2)
            )
            near = projection - torch.sqrt(torch.clamp(
                radius2-perpendicular2, min=0))
            projected = intersects & (near > 0) & (near <= maximum)
            near_valid = torch.where(
                projected, near, torch.full_like(near, float("inf")))
            nearest_actor_depth, owner = torch.min(near_valid, dim=1)
            any_actor = torch.isfinite(nearest_actor_depth)
            visible_owner = any_actor & (nearest_actor_depth < static)
            actor_indices = torch.arange(
                actor_count, device=self.device)[None, :, None, None]
            visible = (
                projected
                & (owner[:, None, :, :] == actor_indices)
                & visible_owner[:, None, :, :]
            )
            static_blocked = projected & (
                static[:, None, :, :] <= near)
            composed = torch.minimum(static, nearest_actor_depth)
            composed = torch.where(any_actor, composed, static)
            actor_mask = visible.any(dim=1)
            center_distance = torch.linalg.norm(offset[:, :, 0, 0, :], dim=-1)
            center_forward = torch.sum(
                offset[:, :, 0, 0, :]
                * rays[:, None, height//2, width//2, :], dim=-1)
            behind = center_forward + radii[None, :] <= 0
            beyond = center_distance - radii[None, :] > maximum
            projected_count = projected.flatten(2).sum(dim=2)
            outside = (
                (projected_count == 0) & ~behind & ~beyond
            )
            visible_count = visible.flatten(2).sum(dim=2)
            blocked_count = static_blocked.flatten(2).sum(dim=2)
        else:
            composed = static.clone()
            actor_mask = torch.zeros_like(static, dtype=torch.bool)
            shape = (frame_count, 0)
            projected_count = torch.zeros(
                shape, dtype=torch.int64, device=self.device)
            visible_count = projected_count.clone()
            blocked_count = projected_count.clone()
            outside = torch.zeros(
                shape, dtype=torch.bool, device=self.device)
            behind = outside.clone()
            beyond = outside.clone()
            owner = torch.full(
                (frame_count, height, width), -1,
                dtype=torch.int64, device=self.device)
            near_valid = torch.empty(
                (frame_count, 0, height, width),
                dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        result = {
            "composed_depth": composed.cpu().numpy().astype("<f4"),
            "static_depth": static.cpu().numpy().astype("<f4"),
            "actor_pixel_count": actor_mask.flatten(1).sum(
                dim=1).cpu().numpy().astype(np.int64),
            "per_actor_projected_pixel_count":
                projected_count.cpu().numpy().astype(np.int64),
            "per_actor_visible_pixel_count":
                visible_count.cpu().numpy().astype(np.int64),
            "per_actor_static_blocked_pixel_count":
                blocked_count.cpu().numpy().astype(np.int64),
            "per_actor_outside_fov":
                outside.cpu().numpy().astype(bool),
            "per_actor_behind_camera":
                behind.cpu().numpy().astype(bool),
            "per_actor_beyond_max_depth":
                beyond.cpu().numpy().astype(bool),
        }
        if return_owner_map:
            owner_output = torch.where(
                visible_owner, owner, torch.full_like(owner, -1)
            ) if actor_count else owner
            result["nearest_actor_owner"] = (
                owner_output.cpu().numpy().astype(np.int32))
        if return_actor_near_depth:
            result["actor_near_depth"] = (
                near_valid.cpu().numpy().astype("<f4"))
        return result

    def render(self, backend, positions, yaws, actor_positions, actor_radii):
        """Backward-compatible composed/static depth and merged pixel count."""
        result = self.render_with_actor_diagnostics(
            backend, positions, yaws, actor_positions, actor_radii)
        return (
            result["composed_depth"],
            result["static_depth"],
            result["actor_pixel_count"],
        )
