from __future__ import annotations

from collections import Counter

from tools.prepare_route_a_v4_maps import load_profiles, selected_profiles


def test_v4_train_plan_is_eighty_twenty_and_type_balanced():
    rows = selected_profiles("train", load_profiles())
    assert len(rows) == 48
    assert sum(row["size_class"] == "large" for row in rows) == 40
    assert sum(row["size_class"] == "narrow" for row in rows) == 8
    large = Counter(row["maze_type"] for row in rows if row["size_class"] == "large")
    assert set(large) == {1, 2, 5, 6, 7}
    assert len(set(large.values())) == 1


def test_v4_validation_large_maps_cover_all_types():
    rows = selected_profiles("valid", load_profiles())
    assert len(rows) == 12
    large = [row for row in rows if row["size_class"] == "large"]
    assert len(large) == 10
    assert {row["maze_type"] for row in large} == {1, 2, 5, 6, 7}
