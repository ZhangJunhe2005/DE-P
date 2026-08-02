import unittest

from tools.evaluate_dynamic_perception_v2 import active, unmatched_by_index, visible


class OcclusionMetricSemanticsTests(unittest.TestCase):
    def test_visible_requires_active_inside_non_occluded_positive_visibility(self):
        base = {"active": True, "inside_image": True, "occluded": False, "visibility": 1.0}
        self.assertTrue(visible(base))
        for change in (
            {"active": False}, {"inside_image": False}, {"occluded": True},
            {"visibility": 0.0},
        ):
            value = dict(base)
            value.update(change)
            self.assertFalse(visible(value))

    def test_active_is_independent_of_occlusion(self):
        self.assertTrue(active({"active": True, "occluded": True}))
        self.assertFalse(active({"active": False, "occluded": False}))

    def test_never_observed_matching_cannot_reuse_primary_track(self):
        tracks = ["visible_actor_track", "unmatched_false_track"]
        self.assertEqual(unmatched_by_index(tracks, {0}), ["unmatched_false_track"])


if __name__ == "__main__":
    unittest.main()
