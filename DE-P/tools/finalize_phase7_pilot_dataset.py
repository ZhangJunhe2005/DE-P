#!/usr/bin/env python3
"""Finalize split files only after all Phase-7 sequences are complete."""

import argparse
import csv
from pathlib import Path

from ruamel.yaml import YAML


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return YAML(typ="safe").load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    with args.matrix.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    splits = {name: [] for name in ("train", "valid", "test")}
    seen_seeds = set()
    for row in rows:
        metadata = load_yaml(root / "sequences" / row["sequence_id"] / "metadata.yaml")
        if metadata.get("completion_status") != "complete" or int(metadata["frame_count"]) != 60:
            raise RuntimeError(f"incomplete sequence: {row['sequence_id']}")
        seed = int(metadata["random_seed"])
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate seed: {seed}")
        seen_seeds.add(seed)
        splits[row["split"]].append(row["sequence_id"])
    (root / "splits").mkdir()
    for split, sequence_ids in splits.items():
        (root / "splits" / f"{split}.txt").write_text(
            "".join(f"{item}\n" for item in sequence_ids), encoding="utf-8"
        )
    manifest = {
        "dataset_version": "dep_dynamic_sequence_v1", "sensor_source": "depth",
        "time_unit": "second", "distance_unit": "meter",
        "splits": {name: f"splits/{name}.txt" for name in splits},
    }
    yaml = YAML()
    with (root / "dataset_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(manifest, stream)
    print({"status": "PASS", "sequences": len(rows),
           "splits": {name: len(values) for name, values in splits.items()}})


if __name__ == "__main__":
    main()
