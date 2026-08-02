#!/usr/bin/env python3
"""Create the one-epoch first stage for the designated resume audit."""

import argparse
from pathlib import Path

from ruamel.yaml import YAML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = YAML(typ="safe").load(args.source)
    if config["run_kind"] != "bounded_shakedown" or int(config["epochs"]) != 2:
        raise ValueError("resume audit source must be a two-epoch bounded shakedown")
    config["epochs"] = 1
    config["config_version"] = f"{config['config_version']}_resume_stage1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.dump(config, stream)


if __name__ == "__main__":
    main()
