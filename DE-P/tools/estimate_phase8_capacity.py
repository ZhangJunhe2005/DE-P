#!/usr/bin/env python3
"""Estimate production recording capacity from the immutable phase-7 pilot."""

import argparse
import json
from pathlib import Path


def tree_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("data/phase7_dynamic_pilot"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/phase8_dataset_capacity_plan.json"))
    parser.add_argument("--train-maps", type=int, default=12)
    parser.add_argument("--valid-maps", type=int, default=3)
    parser.add_argument("--test-maps", type=int, default=3)
    parser.add_argument("--sequences-per-map", type=int, default=12)
    parser.add_argument("--estimated-cache-ratio", type=float, default=0.08)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    sequences = list((pilot / "sequences").iterdir())
    total_bytes = tree_bytes(pilot)
    sequence_bytes = sum(tree_bytes(path) for path in sequences)
    frame_count = sum(1 for path in sequences for _ in (path / "depth").glob("*.npy"))
    sequence_count = len(sequences)
    planned_maps = args.train_maps + args.valid_maps + args.test_maps
    planned_sequences = planned_maps * args.sequences_per_map
    bytes_per_sequence = sequence_bytes / sequence_count
    bytes_per_frame = sequence_bytes / frame_count
    planned_bytes = bytes_per_sequence * planned_sequences
    seconds_per_sequence = frame_count / sequence_count / 10.0
    payload = {
        "status": "AWAITING_USER_APPROVAL",
        "source": str(pilot),
        "pilot": {"bytes": total_bytes, "sequences": sequence_count,
                  "frames": frame_count, "bytes_per_sequence": bytes_per_sequence,
                  "bytes_per_frame": bytes_per_frame},
        "proposed": {
            "train_maps": args.train_maps, "valid_maps": args.valid_maps,
            "test_maps": args.test_maps, "sequences_per_map": args.sequences_per_map,
            "total_sequences": planned_sequences,
            "estimated_dataset_bytes": int(planned_bytes),
            "estimated_dataset_gib": planned_bytes / 2**30,
            "simulated_duration_hours": planned_sequences * seconds_per_sequence / 3600,
            "estimated_context_cache_bytes": int(planned_bytes * args.estimated_cache_ratio),
            "estimated_context_cache_gib": planned_bytes * args.estimated_cache_ratio / 2**30,
            "validation_time_estimate_minutes": planned_sequences * 0.15,
            "recording_wall_time_estimate_hours": planned_sequences * seconds_per_sequence / 3600 * 1.5,
        },
        "assumptions": {
            "record_rate_hz": 10.0, "frames_per_sequence": frame_count / sequence_count,
            "recording_realtime_factor": 1.5,
            "estimated_cache_ratio": args.estimated_cache_ratio,
            "production_recording_started": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
