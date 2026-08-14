"""MedSAM extraction — pre-neck 768-d + masked GAP (D-025, D-026)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.backbone.loaders._extract_common import finalize_extraction_output, metadata_frame
from src.backbone.pooling import masked_global_average_pool, medsam_valid_patch_mask
from src.backbone.preprocessing_transform import load_preprocessing_pipeline
from src.datasets.splits import assert_split_counts, build_eval_pool_df, build_isic_train_df
from src.utils.config import BackboneConfig
from src.utils.paths import load_dataset_paths
from src.utils.torch_device import pin_memory_for, resolve_torch_device


class MedSAMImageDataset(Dataset):
    """Return transformed image, label, and original PIL size for masked pooling."""

    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        row = self.frame.iloc[idx]
        image = Image.open(str(row["path"])).convert("RGB")
        width, height = image.size
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_idx"]), height, width


def create_medsam_model(cfg: BackboneConfig) -> nn.Module:
    from huggingface_hub import hf_hub_download
    from transformers import SamConfig, SamModel

    repo = cfg.checkpoint
    revision = (cfg.raw.get("backbone") or {}).get("revision")
    if not repo:
        raise ValueError(f"{cfg.name}: backbone.checkpoint is null")

    config = SamConfig.from_pretrained(repo, revision=revision)
    model = SamModel(config)
    weight_path = hf_hub_download(repo, "pytorch_model.bin", revision=revision)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def forward_medsam_pre_neck(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return pre-neck spatial tokens [B, 64, 64, 768]."""
    hidden = encoder.patch_embed(images)
    if encoder.pos_embed is not None:
        hidden = hidden + encoder.pos_embed
    for layer in encoder.layers:
        hidden = layer(hidden)
    return hidden


def verify_medsam_forward(cfg: BackboneConfig, model: nn.Module) -> torch.Tensor:
    encoder = model.vision_encoder
    images = torch.zeros(1, 3, 1024, 1024)
    with torch.no_grad():
        tokens = forward_medsam_pre_neck(encoder, images)
        pooled = tokens.mean(dim=(1, 2))
    if pooled.shape[1] != cfg.embed_dim:
        raise ValueError(
            f"{cfg.name}: embed_dim {cfg.embed_dim} != forward dim {pooled.shape[1]} (neck may have run)"
        )
    return pooled


@torch.no_grad()
def _extract_medsam_split(
    encoder: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    embed_dim: int,
    desc: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    encoder.eval()
    for images, _labels, heights, widths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        tokens = forward_medsam_pre_neck(encoder, images)
        tokens_np = tokens.cpu().numpy()
        for i in range(tokens_np.shape[0]):
            mask = medsam_valid_patch_mask(int(heights[i]), int(widths[i]))
            vec = masked_global_average_pool(tokens_np[i], mask)
            if vec.shape[0] != embed_dim:
                raise ValueError(f"MedSAM pooled dim {vec.shape[0]} != {embed_dim}")
            chunks.append(vec.astype(np.float32))
    return np.stack(chunks, axis=0)


def extract_medsam_embeddings(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    device: torch.device | None = None,
):
    paths = load_dataset_paths()
    train_df = build_isic_train_df(paths.master_metadata)
    eval_df = build_eval_pool_df(paths.master_metadata)
    assert_split_counts(train_df, eval_df)

    transform = load_preprocessing_pipeline(cfg.preprocessing_asset)
    dev = device or resolve_torch_device()
    if dev.type == "mps" and batch_size > 2:
        batch_size = 2
    pin = pin_memory_for(dev)

    train_loader = DataLoader(
        MedSAMImageDataset(train_df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    eval_loader = DataLoader(
        MedSAMImageDataset(eval_df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model = create_medsam_model(cfg)
    encoder = model.vision_encoder.to(dev)

    train_z = _extract_medsam_split(
        encoder, train_loader, device=dev, embed_dim=cfg.embed_dim, desc=f"{cfg.name} train"
    )
    eval_z = _extract_medsam_split(
        encoder, eval_loader, device=dev, embed_dim=cfg.embed_dim, desc=f"{cfg.name} eval"
    )
    return finalize_extraction_output(
        cfg,
        output_dir=output_dir,
        train_df=train_df,
        eval_df=eval_df,
        train_z=train_z,
        eval_z=eval_z,
    )
