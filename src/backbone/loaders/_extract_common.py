"""Shared helpers for backbone embedding extraction (Step 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbone.extract import ExtractionOutput, _write_fixture_artifact
from src.backbone.preprocessing_transform import load_preprocessing_pipeline
from src.datasets.image_dataset import MetadataImageDataset
from src.datasets.splits import assert_split_counts, build_eval_pool_df, build_isic_train_df
from src.utils.config import BackboneConfig
from src.utils.paths import load_dataset_paths


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[["domain", "label_idx"]].copy()


def build_standard_dataloaders(
    cfg: BackboneConfig,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DataLoader, DataLoader, torch.device]:
    paths = load_dataset_paths()
    train_df = build_isic_train_df(paths.master_metadata)
    eval_df = build_eval_pool_df(paths.master_metadata)
    assert_split_counts(train_df, eval_df)

    transform = load_preprocessing_pipeline(cfg.preprocessing_asset)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = dev.type == "cuda"

    train_loader = DataLoader(
        MetadataImageDataset(train_df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    eval_loader = DataLoader(
        MetadataImageDataset(eval_df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_df, eval_df, train_loader, eval_loader, dev


def finalize_extraction_output(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_z: np.ndarray,
    eval_z: np.ndarray,
) -> ExtractionOutput:
    if train_z.shape[1] != cfg.embed_dim:
        raise ValueError(
            f"{cfg.name}: extracted dim {train_z.shape[1]} != config embed_dim {cfg.embed_dim}"
        )

    train_dir = output_dir / "ReferenceTrainEmbedding"
    eval_dir = output_dir / "ReferenceEmbedding"
    train_hash = _write_fixture_artifact(train_dir, train_z, metadata_frame(train_df))
    eval_hash = _write_fixture_artifact(eval_dir, eval_z, metadata_frame(eval_df))
    return ExtractionOutput(
        backbone=cfg.name,
        train_dir=train_dir,
        eval_dir=eval_dir,
        train_sha256=train_hash,
        eval_sha256=eval_hash,
        train_n=len(train_df),
        eval_n=len(eval_df),
        embed_dim=cfg.embed_dim,
    )


@torch.no_grad()
def run_feature_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    forward_fn,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for batch in tqdm(loader, desc=desc, leave=False):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, _labels = batch
        else:
            raise ValueError(f"Unexpected batch structure for {desc}")
        images = images.to(device, non_blocking=True)
        feat = forward_fn(model, images)
        if feat.ndim != 2:
            raise ValueError(f"{desc}: expected [B, D] features, got {tuple(feat.shape)}")
        chunks.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)
