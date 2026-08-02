#!/usr/bin/env python3
"""Cross-backend validation of the Q2.3 empty-space sentinel."""

import json
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry_authority.static_v1 import EMPTY_GAP_M, StaticAuthorityMap
from tools.run_phase8jqv2_2_authority_validation import (
    read_cpp, write_queries,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    synthetic = ROOT / "geometry_authority/static_v1/synthetic"
    empty = next(
        path for path in synthetic.iterdir()
        if json.loads((path/"occupancy_metadata.json").read_text())
        ["occupied_voxel_count"] == 0
    )
    authority = StaticAuthorityMap(empty)
    rows = np.asarray([[1., 1., 1., .3]], dtype="<f8")
    offline = authority.query(rows[:, :3], rows[:, 3])
    with tempfile.TemporaryDirectory() as temporary:
        query = Path(temporary)/"query.bin"
        result = Path(temporary)/"result.bin"
        write_queries(query, rows)
        subprocess.run([
            "/home/zjh/YOPO/Simulator/devel/lib/sensor_simulator/"
            "static_authority_contract_test",
            str(empty), str(query), str(result),
        ], check=True, stdout=subprocess.DEVNULL)
        cpu, gpu = read_cpp(result)
    gaps = [
        float(offline.minimum_gap_m[0]),
        float(cpu["minimum_gap_m"][0]),
        float(gpu["minimum_gap_m"][0]),
    ]
    report = {
        "status": "PASS" if gaps == [EMPTY_GAP_M]*3 else "FAIL",
        "minimum_gap_m": EMPTY_GAP_M,
        "minimum_gap_representation":
            "exact_binary32_max_promoted_to_float64",
        "offline_gap_m": gaps[0], "cpu_gap_m": gaps[1],
        "gpu_gap_m": gaps[2],
        "cpu_gpu_offline_exact_equal": gaps == [EMPTY_GAP_M]*3,
        "minimum_gap_mask_required": True,
        "included_in_clearance_mean": False,
        "included_in_risk_normalization": False,
        "collision": False, "out_of_bounds": False,
        "contacted_voxel_index": [-1, -1, -1],
        "contacted_voxel_bounds": "NaN",
        "queried_voxel_count": 0,
    }
    if not args.no_write:
        path = ROOT/"reports/phase8jqv2_3_empty_space_contract.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
