#!/usr/bin/env python3
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/phase8_authoritative_v2"
CONFIG = yaml.safe_load((
    ROOT/"configs/phase8_authoritative_v2_generation.yaml"
).read_text())


def main():
    complete = {"train": 0, "valid": 0}
    sequences = {"train": 0, "valid": 0}
    for path in (DATA/"manifests/sequences").glob("*.json"):
        value = json.loads(path.read_text())
        split = value["split"]
        complete[split] += int(value["frame_count"])
        sequences[split] += 1
    locks = (
        sorted(str(path.relative_to(ROOT))
               for path in (DATA/"generation_state/locks").glob("*.lock"))
        if DATA.exists() else []
    )
    global_lock = DATA/"generation_state/phase8jqv2_4_state_v2_generation.lock"
    if global_lock.is_file():
        locks.append(str(global_lock.relative_to(ROOT)))
    expected = {
        split: int(CONFIG["formal_splits"][split]["frames"])
        for split in ("train", "valid")
    }
    full = (
        DATA/"generation_state/completion/FULL_GENERATION_COMPLETE"
    ).is_file()
    status = (
        "COMPLETE" if full and complete == expected
        else "RUNNING" if locks
        else "INCOMPLETE"
    )
    print(json.dumps({
        "overall": status,
        "dataset_version": "phase8_authoritative_v2",
        "frames": {
            split: f"{complete[split]}/{expected[split]}"
            for split in ("train", "valid")
        },
        "sequences_complete": sequences,
        "locks": locks,
        "completion_marker": full,
        "old_dataset_untouched": True,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
