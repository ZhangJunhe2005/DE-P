"""Canonical Static Geometry Authority V1.

The occupancy bitset, not a PLY or ESDF, is the physical geometry authority.
Serialization is little-endian, x-major/y-middle/z-fast and LSB-first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import uuid

import numpy as np


AUTHORITY_VERSION = "static_geometry_authority_v1"
BACKEND_VERSION = "static_sphere_voxel_aabb_v1"
MAP_NAMESPACE = "static_authority_dev_v1"
RESOLUTION_M = 0.1
DEFAULT_UAV_RADIUS_M = 0.3
CONTACT_TOLERANCE_M = 1e-6
EMPTY_GAP_M = float(np.finfo(np.float32).max)
RAW_HEADER = struct.Struct("<8sQ")
RAW_MAGIC = b"SGARAW1\0"


class AuthorityArtifactError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _point_indices(points, origin, resolution):
    scaled = (points.astype(np.float64) - origin) / float(resolution)
    return np.floor(scaled + 1e-12).astype(np.int64)


def _pack_occupancy(occupied, dimensions):
    flat = np.zeros(int(np.prod(dimensions)), dtype=np.uint8)
    linear = (
        occupied[:, 0] * dimensions[1] * dimensions[2]
        + occupied[:, 1] * dimensions[2] + occupied[:, 2]
    )
    flat[np.unique(linear)] = 1
    return np.packbits(flat, bitorder="little").tobytes(), flat


def _manifest_payload(metadata):
    return {
        "authority_version": metadata["authority_version"],
        "map_uuid": metadata["map_uuid"],
        "map_namespace": metadata["map_namespace"],
        "generator_seed": metadata["generator_seed"],
        "map_id": metadata["map_id"],
        "generator_source_hash": metadata["generator_source_hash"],
        "generator_config_hash": metadata["generator_config_hash"],
        "source_cloud_hash": metadata["source_cloud_hash"],
        "occupancy_hash": metadata["occupancy_hash"],
        "builder_implementation_hash": metadata[
            "builder_implementation_hash"
        ],
        "parent_git_commit": metadata["parent_git_commit"],
    }


def build_authority_artifact(
    output,
    points,
    *,
    map_uuid,
    generator_seed,
    map_id,
    generator_source_hash,
    generator_config_hash,
    parent_git_commit,
    map_namespace=MAP_NAMESPACE,
    resolution=RESOLUTION_M,
    frame_id="odom",
    build_timestamp="1970-01-01T00:00:00Z",
    bounds_min=None,
    bounds_max=None,
):
    """Atomically build or verify one immutable authority artifact."""
    output = Path(output)
    points = np.asarray(points, dtype="<f4")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if not np.isfinite(points).all():
        raise ValueError("source cloud must be finite")
    if abs(float(resolution) - RESOLUTION_M) > 1e-12:
        raise ValueError("V1 resolution is frozen at 0.1 m")
    uuid.UUID(str(map_uuid))

    if bounds_min is None and bounds_max is None:
        if not len(points):
            raise ValueError("empty synthetic cloud requires explicit bounds")
        origin = points.astype(np.float64).min(axis=0)
        indices = _point_indices(points, origin, resolution)
        dimensions = indices.max(axis=0) + 1
        origin_policy = "component_wise_source_cloud_minimum"
    else:
        origin = np.asarray(bounds_min, dtype=np.float64)
        maximum = np.asarray(bounds_max, dtype=np.float64)
        if origin.shape != (3,) or maximum.shape != (3,):
            raise ValueError("explicit bounds must be xyz")
        dimensions = np.rint(
            (maximum - origin) / resolution
        ).astype(np.int64)
        if np.any(dimensions <= 0) or not np.allclose(
            origin + dimensions * resolution, maximum, atol=1e-9
        ):
            raise ValueError("bounds must define positive 0.1 m cells")
        indices = _point_indices(points, origin, resolution)
        if len(indices) and (
            np.any(indices < 0) or np.any(indices >= dimensions)
        ):
            raise ValueError("source point outside explicit bounds")
        origin_policy = "explicit_synthetic_fixture_bounds"
    occupancy_bytes, flat = _pack_occupancy(indices, dimensions)
    raw_bytes = RAW_HEADER.pack(RAW_MAGIC, len(points)) + points.tobytes("C")
    builder_hash = sha256(Path(__file__))
    source_cloud_hash = hashlib.sha256(raw_bytes).hexdigest()
    occupancy_hash = hashlib.sha256(occupancy_bytes).hexdigest()
    bounds_max = origin + dimensions * resolution
    metadata = {
        "authority_version": AUTHORITY_VERSION,
        "backend_version": BACKEND_VERSION,
        "map_uuid": str(map_uuid),
        "map_namespace": str(map_namespace),
        "generator_seed": int(generator_seed),
        "map_id": str(map_id),
        "generator_source_hash": str(generator_source_hash),
        "generator_config_hash": str(generator_config_hash),
        "source_cloud_file": "raw_cloud.bin",
        "source_cloud_hash": source_cloud_hash,
        "source_point_count": int(len(points)),
        "source_cloud_dtype": "little_endian_float32_xyz",
        "occupancy_file": "occupancy.bin",
        "occupancy_resolution_m": RESOLUTION_M,
        "occupancy_origin": origin.tolist(),
        "occupancy_origin_policy": origin_policy,
        "occupancy_dimensions": dimensions.tolist(),
        "occupancy_hash": occupancy_hash,
        "occupied_voxel_count": int(flat.sum()),
        "bounds_min": origin.tolist(),
        "bounds_max": bounds_max.tolist(),
        "frame_id": frame_id,
        "axis_order": "xyz",
        "storage_order": "x_major_y_middle_z_fast",
        "endianness": "little",
        "bit_packing": "lsb_first",
        "voxel_coordinate": "floor((point-origin)/resolution)",
        "voxel_center_convention": "origin+(index+0.5)*resolution",
        "voxel_geometry": "closed_axis_aligned_cube",
        "occupied_rule": "source_point_count_greater_than_zero",
        "uav_shape": "sphere",
        "uav_radius_m": DEFAULT_UAV_RADIUS_M,
        "contact_tolerance_m": CONTACT_TOLERANCE_M,
        "tangent_is_collision": True,
        "sensor_raycast_oob_policy": "legacy_backend_specific",
        "static_collision_oob_policy": "occupied_fail_closed",
        "offline_planner_oob_policy": "occupied_fail_closed",
        "build_timestamp": build_timestamp,
        "builder_implementation_hash": builder_hash,
        "parent_git_commit": str(parent_git_commit),
        "visualization_ply_authoritative": False,
    }
    metadata["artifact_manifest_hash"] = hashlib.sha256(
        canonical_json(_manifest_payload(metadata))
    ).hexdigest()
    metadata_bytes = canonical_json(metadata) + b"\n"
    manifest = {
        "schema": AUTHORITY_VERSION,
        "map_uuid": str(map_uuid),
        "files": {
            "raw_cloud.bin": source_cloud_hash,
            "occupancy.bin": occupancy_hash,
            "occupancy_metadata.json": hashlib.sha256(
                metadata_bytes
            ).hexdigest(),
        },
        "artifact_manifest_hash": metadata["artifact_manifest_hash"],
    }
    manifest_bytes = canonical_json(manifest) + b"\n"

    if output.exists():
        loaded = StaticAuthorityMap(output)
        if (
            loaded.metadata["generator_source_hash"]
            != str(generator_source_hash)
            or loaded.metadata["generator_config_hash"]
            != str(generator_config_hash)
            or loaded.metadata["source_cloud_hash"] != source_cloud_hash
            or loaded.metadata["occupancy_hash"] != occupancy_hash
            or loaded.metadata["builder_implementation_hash"] != builder_hash
        ):
            raise AuthorityArtifactError(
                "existing authority artifact input/hash mismatch"
            )
        return loaded.metadata

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _atomic_bytes(staging / "raw_cloud.bin", raw_bytes)
        _atomic_bytes(staging / "occupancy.bin", occupancy_bytes)
        _atomic_bytes(
            staging / "occupancy_metadata.json", metadata_bytes
        )
        _atomic_bytes(staging / "authority_manifest.json", manifest_bytes)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


@dataclass(frozen=True)
class CollisionBatch:
    collision: np.ndarray
    minimum_gap_m: np.ndarray
    contacted_voxel_index: np.ndarray
    contacted_voxel_bounds: np.ndarray
    out_of_bounds: np.ndarray
    queried_voxel_count: np.ndarray
    backend_version: str
    map_authority_hash: str


class StaticAuthorityMap:
    def __init__(
        self, root, *, expected_map_uuid=None,
        expected_generator_config_hash=None,
        expected_generator_source_hash=None,
    ):
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "occupancy_metadata.json").read_text()
        )
        manifest = json.loads(
            (self.root / "authority_manifest.json").read_text()
        )
        if self.metadata["authority_version"] != AUTHORITY_VERSION:
            raise AuthorityArtifactError("authority version mismatch")
        expected = {
            "map_uuid": expected_map_uuid,
            "generator_config_hash": expected_generator_config_hash,
            "generator_source_hash": expected_generator_source_hash,
        }
        for key, value in expected.items():
            if value is not None and self.metadata[key] != str(value):
                raise AuthorityArtifactError(f"{key} mismatch")
        for name, expected in manifest["files"].items():
            if sha256(self.root / name) != expected:
                raise AuthorityArtifactError(f"{name} hash mismatch")
        if self.metadata["artifact_manifest_hash"] != hashlib.sha256(
            canonical_json(_manifest_payload(self.metadata))
        ).hexdigest():
            raise AuthorityArtifactError("artifact manifest hash mismatch")
        self.origin = np.asarray(
            self.metadata["occupancy_origin"], dtype=np.float64
        )
        self.dimensions = np.asarray(
            self.metadata["occupancy_dimensions"], dtype=np.int64
        )
        self.resolution = float(
            self.metadata["occupancy_resolution_m"]
        )
        bits = np.frombuffer(
            (self.root / "occupancy.bin").read_bytes(), dtype=np.uint8
        )
        count = int(np.prod(self.dimensions))
        self.flat = np.unpackbits(
            bits, bitorder="little", count=count
        ).astype(bool)
        self.occupied_indices = np.argwhere(
            self.flat.reshape(tuple(self.dimensions))
        ).astype(np.int64)
        self.bounds_min = self.origin
        self.bounds_max = (
            self.origin + self.dimensions * self.resolution
        )

    def query(self, centers, radius=DEFAULT_UAV_RADIUS_M):
        centers = np.asarray(centers, dtype=np.float64)
        single = centers.ndim == 1
        centers = np.atleast_2d(centers)
        radii = np.asarray(radius, dtype=np.float64)
        radii = np.broadcast_to(radii, (len(centers),))
        if (
            centers.shape[1:] != (3,)
            or not np.isfinite(centers).all()
            or not np.isfinite(radii).all()
            or np.any(radii < 0)
        ):
            raise ValueError("finite [N,3] centers and non-negative radii required")
        tolerance = CONTACT_TOLERANCE_M
        oob = np.any(
            (centers - radii[:, None] < self.bounds_min - tolerance)
            | (centers + radii[:, None] > self.bounds_max + tolerance),
            axis=1,
        )
        collision = oob.copy()
        gap = np.full(len(centers), EMPTY_GAP_M, dtype=np.float64)
        nearest = np.full((len(centers), 3), -1, dtype=np.int64)
        query_count = np.zeros(len(centers), dtype=np.int64)
        occupied = self.occupied_indices
        if len(occupied):
            bmin = self.origin + occupied * self.resolution
            bmax = bmin + self.resolution
            chunk = 1024
            for start in range(0, len(centers), chunk):
                stop = min(len(centers), start + chunk)
                delta = np.maximum(
                    np.maximum(
                        bmin[None] - centers[start:stop, None],
                        centers[start:stop, None] - bmax[None],
                    ),
                    0.0,
                )
                distance = np.sqrt(np.sum(delta * delta, axis=2))
                minimum_distance = np.min(distance, axis=1)
                # Occupied indices are linear-index sorted. Near-equal
                # contacts select the lowest index deterministically.
                choice = np.argmax(
                    distance
                    <= minimum_distance[:, None] + CONTACT_TOLERANCE_M,
                    axis=1,
                )
                gap[start:stop] = minimum_distance - radii[start:stop]
                nearest[start:stop] = occupied[choice]
                query_count[start:stop] = len(occupied)
            collision |= gap <= tolerance
        nearest[~collision] = -1
        voxel_bounds = np.full((len(centers), 2, 3), np.nan)
        valid = np.all(nearest >= 0, axis=1)
        voxel_bounds[valid, 0] = (
            self.origin + nearest[valid] * self.resolution
        )
        voxel_bounds[valid, 1] = voxel_bounds[valid, 0] + self.resolution
        result = CollisionBatch(
            collision, gap, nearest, voxel_bounds, oob, query_count,
            BACKEND_VERSION, self.metadata["artifact_manifest_hash"],
        )
        if not single:
            return result
        return CollisionBatch(
            *(value[0] if isinstance(value, np.ndarray) else value
              for value in result.__dict__.values())
        )
