"""Single-image Step 3 smoke extraction (production forward path, no dataset I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.backbone.preprocessing_transform import load_preprocessing_pipeline
from src.utils.config import BackboneConfig


@torch.no_grad()
def extract_single_embedding(
    cfg: BackboneConfig,
    image_path: Path | str,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Load one image through the frozen preprocessing asset and return one embedding.

    Uses the same model construction and forward helpers as full extraction.
    """
    path = Path(image_path)
    dev = device or torch.device("cpu")

    image = Image.open(path).convert("RGB")
    orig_width, orig_height = image.size
    transform = load_preprocessing_pipeline(cfg.preprocessing_asset)
    batch = transform(image).unsqueeze(0).to(dev)

    loader = cfg.loader.lower()

    if loader == "timm":
        from src.backbone.loaders.timm_loader import _pool_forward_features, create_timm_model

        model = create_timm_model(cfg).to(dev)
        feat = model.forward_features(batch)
        out = _pool_forward_features(feat, cfg.pooling or "gap")

    elif loader == "hf":
        from src.backbone.loaders.hf_loader import create_hf_model, forward_hf_features

        model = create_hf_model(cfg).to(dev)
        out = forward_hf_features(cfg, model, batch)

    elif loader == "open_clip":
        from src.backbone.loaders.open_clip_loader import (
            create_open_clip_model,
            forward_open_clip_features,
        )

        model = create_open_clip_model(cfg).to(dev)
        out = forward_open_clip_features(cfg, model, batch)

    elif loader == "medsam":
        from src.backbone.loaders.medsam_loader import create_medsam_model, forward_medsam_pre_neck
        from src.backbone.pooling import masked_global_average_pool, medsam_valid_patch_mask

        model = create_medsam_model(cfg)
        encoder = model.vision_encoder.to(dev)
        tokens = forward_medsam_pre_neck(encoder, batch)
        mask = medsam_valid_patch_mask(orig_height, orig_width)
        vec = masked_global_average_pool(tokens.cpu().numpy()[0], mask)
        if vec.shape[0] != cfg.embed_dim:
            raise ValueError(
                f"{cfg.name}: smoke pooled dim {vec.shape[0]} != config {cfg.embed_dim}"
            )
        return vec.astype(np.float32)

    else:
        raise ValueError(f"{cfg.name}: smoke extract unsupported for loader {loader!r}")

    if out.ndim != 2 or out.shape[0] != 1:
        raise ValueError(f"{cfg.name}: expected [1, D] smoke output, got {tuple(out.shape)}")
    if out.shape[1] != cfg.embed_dim:
        raise ValueError(
            f"{cfg.name}: smoke dim {out.shape[1]} != config embed_dim {cfg.embed_dim}"
        )
    return out.cpu().numpy()[0].astype(np.float32)
