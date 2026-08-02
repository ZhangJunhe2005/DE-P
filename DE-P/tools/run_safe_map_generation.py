#!/usr/bin/env python3
"""Non-destructive production wrapper around Simulator dataset_generator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--generator-arg", action="append", default=[])
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"Existing output target: {output}")
        if not args.overwrite:
            raise FileExistsError("refusing existing output; --overwrite is required")
        if not args.yes and input("Type OVERWRITE to continue: ") != "OVERWRITE":
            raise RuntimeError("overwrite not confirmed")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    generated = staging / "generated"
    try:
        subprocess.run([
            str(args.generator.resolve()), "--save-path", str(generated),
            *args.generator_arg,
        ], check=True)
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("validate_static_map_reachability.py")),
            str(generated),
        ], check=True)
        metadata = generated / "generation_metadata.yaml"
        reachability = generated / "reachability_metadata.json"
        if not metadata.is_file() or json.loads(reachability.read_text()).get("status") != "PASS":
            raise RuntimeError("complete generator/reachability metadata is required")
        manifest = {"completion_status": "complete", "atomic_commit": True,
                    "reachability_metadata": reachability.name,
                    "production_recording_started": False}
        (generated / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            backup = output.with_name(f"{output.name}.backup-{time.strftime('%Y%m%d-%H%M%S')}")
            print(f"Recoverable overwrite backup: {backup}")
            os.replace(output, backup)
        os.replace(generated, output)
        staging.rmdir()
    except Exception:
        (staging / "INCOMPLETE").write_text("generation did not commit\n")
        raise
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
