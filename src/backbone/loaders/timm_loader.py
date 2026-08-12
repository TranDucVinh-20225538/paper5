"""timm backbone extraction — protocol Step 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbone.extract import ExtractionOutput, _sha256_array, _write_fixture_artifact
from src.backbone.preprocessing_transform import load_preprocessing_pipeline
from src.datasets.image_dataset import MetadataImageDataset
from src.datasets.splits import assert_split_counts, build_eval_pool_df, build_isic_train_df
from src.utils.config import BackboneConfig
from src.utils.paths import load_dataset_paths


def _metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[["domain", "label_idx"]].copy()


def _pool_forward_features(features: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "gap":
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        if features.ndim == 3:
            return features.mean(dim=1)
    if pooling in ("cls", "none", "attn_pool"):
        if features.ndim == 3:
            return features[:, 0]
    if features.ndim == 2:
        return features
    raise ValueError(f"Cannot pool features with shape {tuple(features.shape)} pooling={pooling!r}")


def create_timm_model(cfg: BackboneConfig) -> torch.nn.Module:
    import hashlib

    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    checkpoint = cfg.checkpoint
    if not checkpoint:
        raise ValueError(f"{cfg.name}: backbone.checkpoint is null — cannot extract")

    bb = cfg.raw.get("backbone", {})
    kwargs: dict = {"pretrained": True, "num_classes": 0}
    loader_kwargs = bb.get("loader_kwargs")
    safetensors_hub = None
    if isinstance(loader_kwargs, dict):
        safetensors_hub = loader_kwargs.get("safetensors_hub")
        kwargs.update({k: v for k, v in loader_kwargs.items() if k != "safetensors_hub"})

    if cfg.name == "uni":
        kwargs.setdefault("init_values", 1e-5)
        kwargs.setdefault("dynamic_img_size", True)

    model = timm.create_model(checkpoint, **kwargs)

    if safetensors_hub:
        path = hf_hub_download(
            safetensors_hub["repo_id"],
            safetensors_hub["filename"],
            revision=safetensors_hub.get("revision"),
        )
        expected = safetensors_hub.get("weights_sha256")
        if expected:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(
                    f"{cfg.name}: MoCo weights sha256 mismatch "
                    f"(got {digest}, expected {expected})"
                )
        state = load_file(path)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


@torch.no_grad()
def _extract_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    pooling: str,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for images, _labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        feat = model.forward_features(images)
        pooled = _pool_forward_features(feat, pooling)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def extract_timm_embeddings(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> ExtractionOutput:
    paths = load_dataset_paths()
    train_df = build_isic_train_df(paths.master_metadata)
    eval_df = build_eval_pool_df(paths.master_metadata)
    assert_split_counts(train_df, eval_df)

    transform = load_preprocessing_pipeline(cfg.preprocessing_asset)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_timm_model(cfg).to(dev)
    pooling = cfg.pooling or "gap"

    train_ds = MetadataImageDataset(train_df, transform)
    eval_ds = MetadataImageDataset(eval_df, transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )

    train_z = _extract_loader(model, train_loader, pooling=pooling, device=dev, desc=f"{cfg.name} train")
    eval_z = _extract_loader(model, eval_loader, pooling=pooling, device=dev, desc=f"{cfg.name} eval")

    if train_z.shape[1] != cfg.embed_dim:
        raise ValueError(
            f"{cfg.name}: extracted dim {train_z.shape[1]} != config embed_dim {cfg.embed_dim}"
        )

    train_dir = output_dir / "ReferenceTrainEmbedding"
    eval_dir = output_dir / "ReferenceEmbedding"
    train_hash = _write_fixture_artifact(train_dir, train_z, _metadata_frame(train_df))
    eval_hash = _write_fixture_artifact(eval_dir, eval_z, _metadata_frame(eval_df))

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
