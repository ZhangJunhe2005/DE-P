"""Single-source validation for the static DEP backbone architecture."""

from __future__ import annotations

BACKBONE_VARIANTS = ("legacy", "corrected")


def resolve_backbone_variant(backbone_variant=None) -> str:
    """Resolve an explicit variant or the shared YAML default, then validate it."""
    if backbone_variant is None:
        from config.config import cfg

        backbone_variant = cfg["backbone_variant"]
    if backbone_variant not in BACKBONE_VARIANTS:
        raise ValueError(
            f"Unsupported backbone_variant={backbone_variant!r}; "
            f"expected one of {BACKBONE_VARIANTS}"
        )
    return backbone_variant
