"""Proposer-only contract for the controlled annex occlusion fixture.

This module is deliberately not imported by dataset generation or any
validator.  It labels development fixtures; it cannot award natural-map
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


FIXTURE_VERSION = "occlusion_fixture_annex_v1"


@dataclass(frozen=True)
class AnnexFixtureProvenance:
    authority_root: str
    authority_manifest_hash: str
    profile_name: str
    fixture_version: str


def read_proposer_fixture_provenance(authority_root):
    """Read explicit annex provenance for proposal generation only."""
    authority_root = Path(authority_root)
    path = authority_root / "mixed_scene_provenance.json"
    value = json.loads(path.read_text())
    parameters = value.get("resolved_parameters", {})
    if not parameters.get("occlusion_annex", False):
        raise ValueError("authority artifact is not an annex fixture")
    version = parameters.get("occlusion_fixture_version")
    if version != FIXTURE_VERSION:
        raise ValueError(
            "legacy annex is diagnostic-only and is not fixture v1"
        )
    if not value.get("development_only", False):
        raise ValueError("annex fixture v1 is development-only")
    return AnnexFixtureProvenance(
        authority_root=str(authority_root),
        authority_manifest_hash=value["authority_manifest_hash"],
        profile_name=value["profile_name"],
        fixture_version=version,
    )


__all__ = [
    "AnnexFixtureProvenance",
    "FIXTURE_VERSION",
    "read_proposer_fixture_provenance",
]
