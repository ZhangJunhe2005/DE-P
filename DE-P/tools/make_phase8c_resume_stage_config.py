#!/usr/bin/env python3
"""Derive the one-epoch first half of the designated resume audit run."""

import argparse
from pathlib import Path

from ruamel.yaml import YAML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = YAML(typ="safe").load(args.source)
    config["epochs"] = 1
    yaml = YAML()
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


if __name__ == "__main__":
    main()
