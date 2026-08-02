#!/usr/bin/env python3
"""Create a compact depth/GT preview from recorded Phase-7 data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.sequence / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    chosen = [frames[index] for index in np.linspace(0, len(frames) - 1, 6).astype(int)]
    figure, axes = plt.subplots(2, 3, figsize=(12, 5.2), constrained_layout=True)
    for axis, frame in zip(axes.flat, chosen):
        depth = np.load(args.sequence / frame["depth_path"], allow_pickle=False)
        objects = json.loads((args.sequence / frame["dynamic_objects_path"]).read_text())
        image = axis.imshow(depth, vmin=0, vmax=20, cmap="turbo")
        for obj in objects:
            if obj["inside_image"]:
                axis.scatter(obj["projected_u"], obj["projected_v"], s=50,
                             facecolors="none", edgecolors="white")
                axis.text(obj["projected_u"] + 2, obj["projected_v"], str(obj["object_id"]),
                          color="white", fontsize=8)
        axis.set_title(f"frame {frame['frame_index']}  t={float(frame['timestamp']):.2f}")
        axis.axis("off")
    figure.colorbar(image, ax=axes, label="depth (m)", shrink=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    print(args.output)


if __name__ == "__main__":
    main()
