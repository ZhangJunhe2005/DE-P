#!/usr/bin/env python3
"""One-time pre-finalization repair for ephemeral V3 map references."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v2"
V3 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v3"
EXPECTED_PRE_REPAIR = (
    "05a4537dc472637c464df3695a97a67f6cf09143fc4940e5beeebf5450460760"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_path(value: str) -> str:
    path = (V2 / "maps" / Path(value).name).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    manifest_path = V3 / "manifests/dataset_manifest.json"
    current = sha256_file(manifest_path)
    if current != EXPECTED_PRE_REPAIR:
        raise RuntimeError(
            f"unexpected pre-repair V3 manifest: {current}; refusing mutation"
        )

    yaml = YAML(typ="safe")
    catalog_path = V3 / "map_catalog.yaml"
    catalog = yaml.load(catalog_path)
    authority_path = V3 / "manifests/map_authority.json"
    authority = json.load(open(authority_path))
    before = {
        "dataset_manifest_sha256": current,
        "map_catalog_sha256": sha256_file(catalog_path),
        "map_authority_sha256": sha256_file(authority_path),
    }
    changed = 0
    for entry in catalog["maps"]:
        replacement = stable_path(entry["static_ply"])
        changed += replacement != entry["static_ply"]
        entry["static_ply"] = replacement
    for entry in authority["maps"]:
        entry["static_ply"] = stable_path(entry["static_ply"])
    if changed != len(catalog["maps"]):
        raise RuntimeError("expected every published catalog path to be ephemeral")

    from io import StringIO

    output = StringIO()
    YAML().dump(catalog, output)
    atomic_text(catalog_path, output.getvalue())
    atomic_json(authority_path, authority)

    manifest = json.load(open(manifest_path))
    manifest["map_authority_hash"] = sha256_file(authority_path)
    manifest["map_catalog_hash"] = sha256_file(catalog_path)
    manifest["stable_map_reference_repair"] = (
        "SMGSS_TR1_PRE_FINALIZATION_EPHEMERAL_PATH_REPAIR"
    )
    atomic_json(manifest_path, manifest)
    repaired_manifest = sha256_file(manifest_path)
    complete_path = V3 / "generation_state/BUILD_COMPLETE.json"
    complete = json.load(open(complete_path))
    complete["manifest_sha256"] = repaired_manifest
    complete["pre_finalization_reference_repair_count"] = 1
    atomic_json(complete_path, complete)

    after = {
        "status": "PASS",
        "scope": "map references only; no samples, split, depth, or source changed",
        "paths_repaired": changed,
        "before": before,
        "after": {
            "dataset_manifest_sha256": repaired_manifest,
            "map_catalog_sha256": sha256_file(catalog_path),
            "map_authority_sha256": sha256_file(authority_path),
        },
    }
    atomic_json(
        ROOT / "reports/phase8jqv2_5smgsstr1_stable_map_reference_repair.json",
        after,
    )
    print(json.dumps(after, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
