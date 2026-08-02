#!/usr/bin/env python3
"""Compare two independently committed generator outputs byte-for-byte."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    compared = ["pointcloud-0.ply", "pose-0.csv", "start_goal-0.csv",
                "reachability_metadata.json"]
    hashes = {}
    for name in compared:
        left, right = digest(args.first / name), digest(args.second / name)
        hashes[name] = {"first": left, "second": right, "match": left == right}
    status = "PASS" if all(item["match"] for item in hashes.values()) else "FAIL"
    print(json.dumps({"status": status, "files": hashes}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
