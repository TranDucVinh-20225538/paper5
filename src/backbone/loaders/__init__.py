"""Backbone-specific extraction loaders."""

from __future__ import annotations

from src.backbone.loaders.medsam_loader import extract_medsam_embeddings
from src.backbone.loaders.timm_loader import extract_timm_embeddings

__all__ = ["extract_medsam_embeddings", "extract_timm_embeddings"]
