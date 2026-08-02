"""SMGSS-TR1 frozen-contract regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
V3 = ROOT / "data/phase8_mixed_scene_static_yopo_derived_v3"


@pytest.fixture(scope="module")
def checklist():
    return json.load(open(
        REPORTS / "phase8jqv2_5smgsstr1_test_coverage_77.json"
    ))


@pytest.mark.parametrize("number", range(1, 78))
def test_frozen_requirement(number, checklist):
    """Every numbered requirement has a frozen, passing audit predicate."""
    assert checklist["checks"][f"{number:02d}"] is True


def test_exact_route_a_terminal_result():
    result = json.load(open(
        REPORTS / "phase8jqv2_5smgsstr1_final_result.json"
    ))
    assert result["status"] == "PASS_STATIC_TRAINING_READY"
    assert result["long_training_started"] is False
    assert result["production_qualified"] is False


def test_validation_has_all_five_map_types():
    manifest = json.load(open(V3 / "manifests/dataset_manifest.json"))
    assert set(manifest["map_type_sample_counts"]["validation"]) == {
        "cave", "forest", "pillar", "room", "wall",
    }


def test_published_catalog_has_no_staging_paths():
    catalog = YAML(typ="safe").load(V3 / "map_catalog.yaml")
    paths = [Path(row["static_ply"]) for row in catalog["maps"]]
    assert paths
    assert all(".staging" not in str(path) for path in paths)
    assert all(path.is_file() for path in paths)


def test_combined_h5_is_entry_only():
    result = json.load(open(
        REPORTS / "phase8jqv2_5smgsstr1_combined_h5_dryrun.json"
    ))
    assert result["status"] == "ENTRY_READY_NOT_RUN_FORMAL"
    assert result["formal_h5_started"] is False
    assert result["formal_h5_pass_claimed"] is False
    assert result["production_qualified"] is False


def test_training_launcher_requires_explicit_user_confirmation():
    text = (
        ROOT / "scripts/phase8jqv2_5_run_mixed_static_yopo_formal_training_v3.sh"
    ).read_text()
    assert "--authorize-training" in text
    assert "TRAIN_MIXED_STATIC_YOPO_V3" in text
    assert "--verify-only" in text
    assert "--resume" in text
