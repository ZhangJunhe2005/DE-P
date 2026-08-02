import json
from pathlib import Path
import tempfile
import unittest

from ruamel.yaml import YAML

from tools.phase8d_production_gate import REQUIRED_TRUE, sha256
from tools.run_managed_dynamic_training import require_production_authorization


class Phase8DProductionGateTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        data = root / "formal"; data.mkdir()
        manifest = data / "dataset_manifest.yaml"
        YAML().dump({
            "completion_status": "complete",
            "formal_pointcloud_training_allowed": False,
        }, manifest)
        dynamic_catalog = data / "map_catalog.yaml"; dynamic_catalog.write_text("dynamic")
        static_catalog = root / "static.yaml"; static_catalog.write_text("static")
        checkpoint = root / "init.pt"; checkpoint.write_bytes(b"checkpoint")
        risk = root / "risk.json"; risk.write_text("{}")
        shakedown = root / "shakedown.json"; shakedown.write_text("{}")
        config = root / "config.yaml"
        YAML().dump({
            "config_version": "phase8d_production_v2",
            "dataset_root": str(data), "dataset_manifest": str(manifest),
            "dynamic_map_catalog": str(dynamic_catalog),
            "static_map_catalog": str(static_catalog),
            "initialization_checkpoint": str(checkpoint),
        }, config)
        gate_payload = {"status": "PASS", **{name: True for name in REQUIRED_TRUE}}
        gate_payload.update({
            "data_root": str(data.resolve()),
            "dataset_manifest_sha256": sha256(manifest),
            "dynamic_map_catalog_sha256": sha256(dynamic_catalog),
            "static_map_catalog_sha256": sha256(static_catalog),
            "production_config_sha256": sha256(config),
            "initialization_checkpoint_sha256": sha256(checkpoint),
            "risk_set_manifest": str(risk), "risk_set_manifest_sha256": sha256(risk),
            "shakedown_report": str(shakedown), "shakedown_report_sha256": sha256(shakedown),
        })
        gate = root / "gate.json"
        gate.write_text(json.dumps(gate_payload))
        return config, gate, gate_payload

    def test_complete_gate_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            config, gate, _ = self.fixture(root)
            require_production_authorization(True, config, gate)
            with self.assertRaisesRegex(RuntimeError, "explicit --yes"):
                require_production_authorization(False, config, gate)

    def test_false_ready_or_missing_artifact_locks(self):
        with tempfile.TemporaryDirectory() as root:
            config, gate, payload = self.fixture(root)
            payload["production_ready"] = False
            gate.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "false/missing"):
                require_production_authorization(True, config, gate)

    def test_every_locked_hash_rejects_mutation(self):
        targets = (
            "dataset_manifest", "dynamic_map_catalog", "static_map_catalog",
            "initialization_checkpoint",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as root:
                config, gate, _ = self.fixture(root)
                values = YAML(typ="safe").load(config)
                Path(values[target]).write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    require_production_authorization(True, config, gate)

    def test_config_risk_shakedown_and_incomplete_staging_lock(self):
        for target in ("config", "risk", "shakedown", "staging"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as root:
                config, gate, payload = self.fixture(root)
                if target == "config":
                    with config.open("a") as stream:
                        stream.write("changed: true\n")
                elif target == "risk":
                    Path(payload["risk_set_manifest"]).unlink()
                elif target == "shakedown":
                    Path(payload["shakedown_report"]).unlink()
                else:
                    data = Path(payload["data_root"])
                    data.parent.joinpath(f".{data.name}.staging-test").mkdir()
                with self.assertRaises(RuntimeError):
                    require_production_authorization(True, config, gate)


if __name__ == "__main__":
    unittest.main()
