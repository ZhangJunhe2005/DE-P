import inspect
from pathlib import Path

from authoritative_dataset.occlusion_constructor_v2_2 import (
    feasibility_interval,
    legacy_v2_1_detection_lower_bound,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_1_is_frozen_without_annex_or_hotfix():
    source = (
        ROOT / "authoritative_dataset/occlusion_constructor_v2_1.py"
    ).read_text()
    assert "MIXED_MAP_RATIO_HOTFIX" not in source
    assert "_mixed_annex_geometry" not in source
    assert "mixed_scene_provenance" not in source
    assert "{\n            1: (6.4, 7.0)," in source


def test_v2_2_legacy_impossibility_and_controls():
    assert legacy_v2_1_detection_lower_bound() == 1.9312
    for gap in (1, 2, 3):
        assert feasibility_interval(
            .355, .4, .8, gap, 1.4
        )["feasible"]
    assert not feasibility_interval(
        .355, .1, .1, 3, 1.4
    )["feasible"]


def test_v2_2_constructor_does_not_read_annex_provenance():
    import authoritative_dataset.occlusion_constructor_v2_2 as module

    source = inspect.getsource(module)
    assert "mixed_scene_provenance" not in source
    assert "occlusion_fixture_annex_v1" not in source


def test_independent_validator_has_no_provenance_reader():
    source = (
        ROOT / "tools/validate_v2_2_natural_occlusion_cuda.py"
    ).read_text()
    assert "mixed_scene_provenance" not in source
    assert "occlusion_fixture_annex_v1" not in source


def test_natural_profiles_disable_annex_by_default():
    source = (ROOT / "configs/mixed_scene_map_profiles_v1.yaml").read_text()
    assert "occlusion_annex: false" in source


def test_no_v2_2_formal_generation_entry():
    source = (
        ROOT / "authoritative_dataset/generate_v1.py"
    ).read_text()
    assert "occlusion_constructor_v2_2" not in source
    assert "build_occluded_actor_specs_v2_2" not in source
