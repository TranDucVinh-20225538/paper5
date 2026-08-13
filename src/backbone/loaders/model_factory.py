"""Unified backbone model construction for verification and extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.backbone.loaders.hf_loader import create_hf_model, forward_hf_features, verify_hf_forward
from src.backbone.loaders.medsam_loader import create_medsam_model, verify_medsam_forward
from src.backbone.loaders.open_clip_loader import (
    create_open_clip_model,
    forward_open_clip_features,
    verify_open_clip_forward,
)
from src.backbone.loaders.timm_loader import create_timm_model
from src.utils.config import BackboneConfig


def create_backbone_model(cfg: BackboneConfig) -> nn.Module:
    loader = cfg.loader.lower()
    if loader == "timm":
        return create_timm_model(cfg)
    if loader == "hf":
        return create_hf_model(cfg)
    if loader == "open_clip":
        return create_open_clip_model(cfg)
    if loader == "medsam":
        return create_medsam_model(cfg)
    raise ValueError(f"{cfg.name}: unsupported loader {loader!r}")


def verify_backbone_forward(cfg: BackboneConfig, model: nn.Module) -> torch.Tensor:
    loader = cfg.loader.lower()
    if loader == "timm":
        size = getattr(model, "pretrained_cfg", {}).get("input_size", (3, 224, 224))
        with torch.no_grad():
            out = model(torch.zeros(1, *size))
        if out.ndim != 2:
            raise ValueError(f"{cfg.name}: expected 2-D output, got {tuple(out.shape)}")
        return out
    if loader == "hf":
        return verify_hf_forward(cfg, model)
    if loader == "open_clip":
        return verify_open_clip_forward(cfg, model)
    if loader == "medsam":
        return verify_medsam_forward(cfg, model)
    raise ValueError(f"{cfg.name}: verify not implemented for loader {loader!r}")
