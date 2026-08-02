from pathlib import Path

from ruamel.yaml import YAML


def test_v4_2_1_regularization_and_stop_contract():
    path = Path("configs/route_a_v4_2_1_regularized_training.yaml")
    config = YAML(typ="safe").load(path)
    assert config["contract_version"] == "route_a_static_yopo_training_v4_2_1"
    assert config["optimizer"]["backbone_learning_rate"] == 2.0e-6
    assert config["optimizer"]["head_learning_rate"] == 2.0e-5
    assert config["optimizer"]["weight_decay"] == 5.0e-4
    assert config["scheduler"]["patience"] == 5
    assert config["validation"]["minimum_epoch"] == 12
    assert config["validation"]["patience"] == 8
    assert config["model"]["initial_checkpoint"].endswith(
        "saved/DEP_0/epoch10.pth"
    )
