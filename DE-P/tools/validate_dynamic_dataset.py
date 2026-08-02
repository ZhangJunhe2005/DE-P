#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.dynamic_sequence_dataset import DynamicSequenceDataset, validate_dataset_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest, splits = validate_dataset_splits(args.root)
    details = {}
    for split in ("train", "valid", "test"):
        dataset = DynamicSequenceDataset(args.root, split=split)
        sample = dataset[0]
        details[split] = {
            "sequences": len(splits[split]), "windows": len(dataset),
            "current_depth_shape": list(sample["current_depth"].shape),
            "first_sequence": sample["sequence_id"],
        }
    print(json.dumps({"status": "PASS", "manifest": manifest, "splits": details}, indent=2))


if __name__ == "__main__":
    main()
