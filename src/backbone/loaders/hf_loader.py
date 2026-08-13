"""Hugging Face backbone extraction — MONET, SigLIP, MoCo v3, DINOv3."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from src.backbone.loaders._extract_common import (
    build_standard_dataloaders,
    finalize_extraction_output,
    run_feature_batches,
)
from src.utils.config import BackboneConfig


def _backbone_fields(cfg: BackboneConfig) -> dict:
    return cfg.raw.get("backbone") or {}


def _hub_revision(cfg: BackboneConfig) -> str | None:
    return _backbone_fields(cfg).get("revision")


def _hub_repo(cfg: BackboneConfig) -> str:
    repo = _backbone_fields(cfg).get("hub_repo") or cfg.checkpoint
    if not repo:
        raise ValueError(f"{cfg.name}: backbone.checkpoint is null")
    return str(repo)


def create_hf_model(cfg: BackboneConfig) -> torch.nn.Module:
    """Construct a HF-backed model for forward verification and extraction."""
    if cfg.name == "mocov3":
        return _create_mocov3_timm_model(cfg)

    repo = _hub_repo(cfg)
    revision = _hub_revision(cfg)

    if cfg.name == "monet":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(repo, revision=revision)
        return _MonetVisionWrapper(model)

    if cfg.name == "siglip":
        from transformers import SiglipVisionModel

        return SiglipVisionModel.from_pretrained(repo, revision=revision)

    if cfg.name == "dinov3":
        from transformers import AutoModel

        return AutoModel.from_pretrained(repo, revision=revision)

    raise ValueError(f"{cfg.name}: no HF loader registered")


def _create_mocov3_timm_model(cfg: BackboneConfig) -> torch.nn.Module:
    """MoCo v3 via timm ViT-B + HF safetensors mirror (D-042)."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    bb = _backbone_fields(cfg)
    repo = bb.get("hub_repo") or cfg.checkpoint
    revision = bb.get("revision")
    weights_sha256 = "345a04069364250ef448402363443ae8d2da68678e6a88239a8906d2325b912e"

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    path = Path(hf_hub_download(repo, "model.safetensors", revision=revision))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != weights_sha256:
        raise ValueError(f"mocov3: weights sha256 mismatch (got {digest})")
    model.load_state_dict(load_file(path), strict=False)
    model.eval()
    return model


class _MonetVisionWrapper(nn.Module):
    """Expose MONET vision tower CLS tokens (pre-projection)."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = clip_model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]


def forward_hf_features(cfg: BackboneConfig, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    pooling = cfg.pooling or "cls"

    if cfg.name == "monet":
        return model(images)

    if cfg.name == "siglip":
        out = model(pixel_values=images)
        if pooling == "attn_pool":
            pooled = out.pooler_output
            if pooled is None:
                raise ValueError("siglip: pooler_output is None — check SiglipVisionModel forward")
            return pooled
        return out.last_hidden_state[:, 0, :]

    if cfg.name == "mocov3":
        feat = model.forward_features(images)
        if feat.ndim == 3:
            return feat[:, 0] if pooling == "cls" else feat.mean(dim=1)
        if feat.ndim == 2:
            return feat
        raise ValueError(f"mocov3: unexpected feature shape {tuple(feat.shape)}")

    if cfg.name == "dinov3":
        out = model(pixel_values=images)
        return out.last_hidden_state[:, 0, :]

    raise ValueError(f"{cfg.name}: forward_hf_features not implemented")


def verify_hf_forward(cfg: BackboneConfig, model: nn.Module) -> torch.Tensor:
    """Single-batch forward for verify_checkpoint.py."""
    size = 256 if cfg.name == "siglip" else 224
    if cfg.name == "monet":
        images = torch.zeros(1, 3, size, size)
    else:
        images = torch.zeros(1, 3, size, size)
    model.eval()
    with torch.no_grad():
        return forward_hf_features(cfg, model, images)


def extract_hf_embeddings(
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
    model = create_hf_model(cfg).to(dev)

    def _fn(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return forward_hf_features(cfg, m, batch)

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
