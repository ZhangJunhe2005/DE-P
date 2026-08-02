import unittest

from tools.finalize_phase8c_static_maps import reachability_by_numeric_id
from tools.generate_phase8c_scenario_matrix import validate_catalog_map_entry


class StaticCatalogTest(unittest.TestCase):
    def test_reachability_is_indexed_by_numeric_map_id(self):
        lexical_order = [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]
        reach = {
            "maps": [{"map_id": value, "marker": value} for value in lexical_order]
        }

        indexed = reachability_by_numeric_id(reach, 12, "train")

        self.assertEqual(
            [indexed[index]["marker"] for index in range(12)], list(range(12))
        )

    def test_reachability_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            reachability_by_numeric_id(
                {"maps": [{"map_id": value} for value in [0, 1, 1]]}, 3, "test"
            )

    def test_reachability_rejects_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            reachability_by_numeric_id(
                {"maps": [{"map_id": value} for value in [0, 1, 3]]}, 3, "test"
            )

    def test_scenario_generation_rejects_crossed_map_identity(self):
        with self.assertRaisesRegex(ValueError, "catalog map identity mismatch"):
            validate_catalog_map_entry({
                "local_map_id": 2,
                "reachability": {"map_id": 10},
                "static_ply": "/maps/pointcloud-2.ply",
            })


if __name__ == "__main__":
    unittest.main()
