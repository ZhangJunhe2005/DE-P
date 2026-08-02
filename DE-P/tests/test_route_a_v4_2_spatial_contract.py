import hashlib
import json

import numpy as np
import yaml

from authoritative_dataset.generate_v1 import safe_position


class _Map:
    bounds_min = np.asarray([0.0, 0.0, 0.0])
    bounds_max = np.asarray([60.0, 60.0, 15.0])


class _FreeBackend:
    map = _Map()

    @staticmethod
    def query_one(_point, _radius):
        return {"collision": False, "minimum_gap_m": 1.0}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()


def test_v4_2_contract_hash_and_map_types():
    config = yaml.safe_load(
        open("configs/route_a_v4_2_raw_static_resolved.yaml")
    )
    contract = config["scene_spatial_sampling_contract"]
    claimed = contract.pop("contract_hash")
    assert hashlib.sha256(_canonical(contract)).hexdigest() == claimed
    assert set(contract["map_types"]) == {
        "cave", "pillar", "forest", "room", "wall"
    }


def test_forest_sampling_favors_obstacle_bearing_altitudes():
    config = yaml.safe_load(
        open("configs/route_a_v4_2_raw_static_resolved.yaml")
    )
    policy = config["scene_spatial_sampling_contract"]["map_types"]["forest"]
    rng = np.random.default_rng(826000000)
    relative = []
    for _ in range(4000):
        point, _, sample = safe_position(_FreeBackend(), rng, policy)
        assert 0.4 <= point[2] <= 14.6
        relative.append(sample["relative_altitude"])
    relative = np.asarray(relative)
    assert np.mean(relative <= 0.40) >= 0.55
    assert np.mean(relative > 0.68) <= 0.12


def test_legacy_safe_position_return_contract_is_unchanged():
    result = safe_position(_FreeBackend(), np.random.default_rng(1))
    assert len(result) == 2
