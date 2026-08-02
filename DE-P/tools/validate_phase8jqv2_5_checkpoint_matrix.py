#!/usr/bin/env python3
"""Strict-load every declared Q2.5 checkpoint without inference or training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.checkpoint_utils import (
    detect_checkpoint_variant,
    detect_head_variant,
    load_dep_checkpoint,
)
from policy.dep_network import DepNetwork
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source = json.loads(
        (ROOT / "reports/phase8i_checkpoint_matrix.json").read_text()
    )
    rows = []
    for record in source["checkpoints"]:
        path = Path(record["path"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        variant = detect_checkpoint_variant(payload)
        head = detect_head_variant(payload)
        model = DepNetwork(
            backbone_variant=variant,
            head_variant=head,
        )
        loaded = load_dep_checkpoint(model, path, variant)
        actual_hash = sha256(path)
        rows.append({
            "name": record["name"],
            "path": str(path),
            "sha256": actual_hash,
            "declared_sha256": record["sha256"],
            "hash_match": actual_hash == record["sha256"],
            "backbone_variant": variant,
            "head_variant": head,
            "strict_load": loaded["strict"],
            "missing_keys": loaded["missing_keys"],
            "unexpected_keys": loaded["unexpected_keys"],
        })
        del model, payload
    status = all(
        row["hash_match"] and row["strict_load"]
        and not row["missing_keys"] and not row["unexpected_keys"]
        for row in rows
    )
    report = {
        "status": "PASS" if status else "FAIL",
        "dataset_version": "phase8_authoritative_v2",
        "root_manifest_hash":
            "56d7118862afc22d8df8672d747c930bc4fdcfcb1fe0795c19e5ceef47989461",
        "strict_load_results": rows,
        "phase8b": source["phase8b"],
        "phase8b_substituted": False,
        "network_weights_modified_on_disk": False,
        "optimizer_step_executed": False,
        "production_test_used": False,
        "blind_used": False,
    }
    output = ROOT / "reports/phase8jqv2_5_checkpoint_matrix.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        **{key: value for key, value in report.items()
           if key != "strict_load_results"},
        "strict_load_count": len(rows),
    }, indent=2))
    raise SystemExit(0 if status else 2)


if __name__ == "__main__":
    main()
