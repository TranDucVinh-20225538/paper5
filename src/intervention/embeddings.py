"""Load cached embedding arrays + metadata for intervention steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingArtifacts:
    train_embeddings: np.ndarray
    train_metadata: pd.DataFrame
    eval_embeddings: np.ndarray
    eval_metadata: pd.DataFrame
    train_dir: Path
    eval_dir: Path


def _load_dir(artifact_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    emb_path = artifact_dir / "embeddings.npy"
    meta_path = artifact_dir / "metadata.csv"
    if not emb_path.is_file():
        raise FileNotFoundError(f"embeddings.npy missing: {emb_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.csv missing: {meta_path}")
    embeddings = np.load(emb_path).astype(np.float32)
    metadata = pd.read_csv(meta_path)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"{artifact_dir}: embeddings rows {embeddings.shape[0]} != metadata {len(metadata)}"
        )
    return embeddings, metadata


def load_embedding_artifacts(train_dir: Path, eval_dir: Path) -> EmbeddingArtifacts:
    train_z, train_meta = _load_dir(train_dir)
    eval_z, eval_meta = _load_dir(eval_dir)
    return EmbeddingArtifacts(
        train_embeddings=train_z,
        train_metadata=train_meta,
        eval_embeddings=eval_z,
        eval_metadata=eval_meta,
        train_dir=train_dir,
        eval_dir=eval_dir,
    )


def build_training_population(artifacts: EmbeddingArtifacts) -> tuple[np.ndarray, np.ndarray]:
    """ISIC train + PAD rows with y=-1, matching Paper 4 / CSG-Skin convention."""
    y_isic = artifacts.train_metadata["label_idx"].to_numpy().astype(np.int64)
    z_isic = artifacts.train_embeddings

    pad_mask = (artifacts.eval_metadata["domain"] == "pad_ufes").to_numpy()
    z_pad = artifacts.eval_embeddings[pad_mask]
    y_pad = np.full(len(z_pad), -1, dtype=np.int64)

    z = np.concatenate([z_isic, z_pad], axis=0)
    y = np.concatenate([y_isic, y_pad], axis=0)
    return z, y
