"""Backbone-specific extraction loaders."""

from __future__ import annotations

from src.backbone.loaders.hf_loader import create_hf_model, extract_hf_embeddings
from src.backbone.loaders.medsam_loader import create_medsam_model, extract_medsam_embeddings
from src.backbone.loaders.open_clip_loader import create_open_clip_model, extract_open_clip_embeddings
from src.backbone.loaders.timm_loader import create_timm_model, extract_timm_embeddings

__all__ = [
    "create_hf_model",
    "create_medsam_model",
    "create_open_clip_model",
    "create_timm_model",
    "extract_hf_embeddings",
    "extract_medsam_embeddings",
    "extract_open_clip_embeddings",
    "extract_timm_embeddings",
]
