"""Fail-closed validation for Phase 8D production authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRUE = {
    "production_ready", "capacity_approved", "formal_dataset_validated",
    "formal_dataset_distribution_passed", "map_catalog_isolation_passed",
    "mixed_epoch_schedule_passed", "lazy_static_dataset_passed",
    "fixed_validation_suites_passed", "estimated_context_quality_passed",
    "risk_sets_frozen", "long_horizon_shakedown_passed", "resume_passed",
    "resource_stability_passed", "real_static_smoke_passed",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_phase8d_gate(config_path, gate_path=None):
    config_path = Path(config_path).expanduser().resolve()
    gate_path = Path(gate_path or ROOT / "reports/phase8d_final_preproduction_result.json").resolve()
    if not config_path.is_file() or not gate_path.is_file():
        raise RuntimeError("production is locked: Phase 8D config/gate is missing")
    config = YAML(typ="safe").load(config_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("production is locked: Phase 8D status must be PASS")
    false_fields = sorted(name for name in REQUIRED_TRUE if gate.get(name) is not True)
    if false_fields:
        raise RuntimeError(f"production is locked: Phase 8D gates false/missing {false_fields}")
    paths = {
        "dataset_manifest_sha256": Path(config["dataset_manifest"]),
        "dynamic_map_catalog_sha256": Path(config["dynamic_map_catalog"]),
        "static_map_catalog_sha256": Path(config["static_map_catalog"]),
        "production_config_sha256": config_path,
        "initialization_checkpoint_sha256": Path(config["initialization_checkpoint"]),
    }
    for field, path in paths.items():
        if not path.is_file() or gate.get(field) != sha256(path):
            raise RuntimeError(f"production is locked: {field} mismatch")
    data_root = Path(config["dataset_root"]).resolve()
    if str(data_root) != gate.get("data_root"):
        raise RuntimeError("production is locked: data root mismatch")
    manifest = YAML(typ="safe").load(paths["dataset_manifest_sha256"])
    if manifest.get("completion_status") != "complete":
        raise RuntimeError("production is locked: dataset is incomplete")
    if manifest.get("formal_pointcloud_training_allowed") is not False:
        raise RuntimeError("production is locked: pointcloud training must remain false")
    incomplete = list(data_root.parent.glob(f".{data_root.name}.staging-*"))
    if incomplete:
        raise RuntimeError(f"production is locked: incomplete staging exists: {incomplete}")
    risk_manifest = Path(gate["risk_set_manifest"])
    shakedown = Path(gate["shakedown_report"])
    for label, path, hash_field in (
        ("risk set", risk_manifest, "risk_set_manifest_sha256"),
        ("shakedown", shakedown, "shakedown_report_sha256"),
    ):
        if not path.is_file() or sha256(path) != gate.get(hash_field):
            raise RuntimeError(f"production is locked: {label} missing or changed")
    if config.get("config_version") != "phase8d_production_v2":
        raise RuntimeError("production is locked: production v2 config is required")
    return {"gate": gate, "hashes": {field: sha256(path) for field, path in paths.items()}}
