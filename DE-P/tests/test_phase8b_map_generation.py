import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d

from tools.validate_static_map_reachability import validate_map


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = ROOT.parent / "Simulator/src/src/dataset_generator.cpp"


def fixture(directory):
    points = np.asarray([[x, y, 0.0] for x in np.arange(-2, 8.1, .5)
                         for y in np.arange(-2, 2.1, .5)])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    ply = directory / "pointcloud-0.ply"
    o3d.io.write_point_cloud(str(ply), cloud)
    pose = directory / "pose-0.csv"
    with pose.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("px", "py", "pz", "qw", "qx", "qy", "qz"))
        writer.writerow((0, 0, 1.5, 1, 0, 0, 0)); writer.writerow((6, 0, 1.5, 1, 0, 0, 0))
    return ply, pose


class Phase8BMapGenerationTests(unittest.TestCase):
    def test_simulator_generator_has_explicit_seed_and_safe_commit(self):
        source = SIMULATOR_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("std::random_device", source)
        self.assertIn("std::mt19937", source)
        self.assertIn("pose_seed", source)
        self.assertIn("actor_seed", source)
        self.assertIn("--overwrite", source)
        self.assertIn(".staging-", source)
        self.assertIn("fs::rename", source)

    def test_inflated_reachability_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            ply, pose = fixture(Path(temporary))
            first = validate_map(ply, pose, grid_resolution=.5, inflated_radius=.5,
                                 min_path_length=5.0)
            second = validate_map(ply, pose, grid_resolution=.5, inflated_radius=.5,
                                  min_path_length=5.0)
            self.assertEqual(first, second)
            self.assertEqual(first["reachability"], "PASS")
            self.assertGreaterEqual(first["shortest_path_length"], 5.0)
            self.assertGreaterEqual(len(first["path_waypoints_world"]), 2)

    def test_safe_wrapper_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fake = root / "fake_generator.py"; output = root / "dataset"
            fake.write_text("""#!/usr/bin/env python3
import csv, pathlib, sys
import numpy as np, open3d as o3d
p=pathlib.Path(sys.argv[sys.argv.index('--save-path')+1]); p.mkdir(parents=True)
pts=np.asarray([[x,y,0.] for x in np.arange(-2,8.1,.5) for y in np.arange(-2,2.1,.5)])
o3d.io.write_point_cloud(str(p/'pointcloud-0.ply'),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
with (p/'pose-0.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(('px','py','pz','qw','qx','qy','qz')); w.writerow((0,0,1.5,1,0,0,0)); w.writerow((6,0,1.5,1,0,0,0))
(p/'generation_metadata.yaml').write_text('completion_status: complete\\nmap_seed: 1\\npose_seed: 2\\nactor_seed: 3\\n')
""")
            fake.chmod(0o755)
            command = [sys.executable, str(ROOT / "tools/run_safe_map_generation.py"),
                       "--generator", str(fake), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            before = (output / "dataset_manifest.json").read_bytes()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(before, (output / "dataset_manifest.json").read_bytes())
            self.assertEqual(json.loads(before)["completion_status"], "complete")


if __name__ == "__main__":
    unittest.main()
