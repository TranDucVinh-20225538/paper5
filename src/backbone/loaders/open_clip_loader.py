"""OpenCLIP backbone extraction — BiomedCLIP, OpenCLIP."""

from __future__ import annotations

from pathlib import Path

import open_clip
import torch
import torch.nn as nn

from src.backbone.loaders._extract_common import (
    build_standard_dataloaders,
    finalize_extraction_output,
    run_feature_batches,
)
from src.utils.config import BackboneConfig


def create_open_clip_model(cfg: BackboneConfig) -> nn.Module:
    checkpoint = cfg.checkpoint
    if not checkpoint:
        raise ValueError(f"{cfg.name}: backbone.checkpoint is null")

    if checkpoint.startswith("hf-hub:"):
        model, _, _ = open_clip.create_model_and_transforms(checkpoint)
    elif "::" in checkpoint:
        arch, pretrained = checkpoint.split("::", 1)
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    else:
        raise ValueError(f"{cfg.name}: unsupported open_clip checkpoint {checkpoint!r}")

    model.eval()
    return model


def forward_open_clip_features(cfg: BackboneConfig, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    pooling = cfg.pooling or "cls"

    # TimmModel backbones (BiomedCLIP): trunk is pre-projection pooled embedding.
    if hasattr(visual, "trunk") and not hasattr(visual, "transformer"):
        feat = visual.trunk(images)
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            return feat[:, 0] if pooling == "cls" else feat.mean(dim=1)
        raise ValueError(f"{cfg.name}: unexpected trunk output {tuple(feat.shape)}")

    # Native OpenCLIP ViT — CLS token after ln_post, before proj (D-018).
    if not hasattr(visual, "conv1"):
        raise ValueError(f"{cfg.name}: unsupported open_clip visual type {type(visual).__name__}")

    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls_t = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_t, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x)
    x = visual.ln_post(x[:, 0, :])
    return x


def verify_open_clip_forward(cfg: BackboneConfig, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return forward_open_clip_features(cfg, model, torch.zeros(1, 3, 224, 224))


def extract_open_clip_embeddings(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    train_df, eval_df, train_loader, eval_loader, dev = build_standard_dataloaders(
        cfg, batch_size=batch_size, num_workers=num_workers, device=device
    )
    model = create_open_clip_model(cfg).to(dev)

    def _fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return forward_open_clip_features(cfg, m, batch)

    train_z = run_feature_batches(
        model, train_loader, forward_fn=_fn, device=dev, desc=f"{cfg.name} train"
    )
    eval_z = run_feature_batches(
        model, eval_loader, forward_fn=_fn, device=dev, desc=f"{cfg.name} eval"
    )
    return finalize_extraction_output(
        cfg,
        output_dir=output_dir,
        train_df=train_df,
        eval_df=eval_df,
        train_z=train_z,
        eval_z=eval_z,
    )
