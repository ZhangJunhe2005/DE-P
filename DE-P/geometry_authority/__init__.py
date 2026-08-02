"""Versioned static geometry authority backends."""

from .static_v1 import (
    AUTHORITY_VERSION,
    CONTACT_TOLERANCE_M,
    DEFAULT_UAV_RADIUS_M,
    AuthorityArtifactError,
    StaticAuthorityMap,
    build_authority_artifact,
)

__all__ = [
    "AUTHORITY_VERSION",
    "CONTACT_TOLERANCE_M",
    "DEFAULT_UAV_RADIUS_M",
    "AuthorityArtifactError",
    "StaticAuthorityMap",
    "build_authority_artifact",
]
