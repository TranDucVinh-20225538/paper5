"""
Backbone-agnostic embedding extraction wrapper (protocol Step 3).

Real checkpoint loaders are dispatched by config; only ``fixture`` is implemented
for integration tests. Production loaders (panderm, timm, medsam, …) raise until
wired in a later milestone — no PanDerm assumptions in the dispatch layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backbone.pooling_dispatch import pool_representation
from src.utils.config import BackboneConfig


@dataclass(frozen=True)
class ExtractionOutput:
    """Paths and checksums for one backbone extraction pass."""

    backbone: str
    train_dir: Path
    eval_dir: Path
    train_sha256: str
    eval_sha256: str
    train_n: int
    eval_n: int
    embed_dim: int
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def train_path(self) -> Path:
        return self.train_dir / "embeddings.npy"

    @property
    def eval_path(self) -> Path:
        return self.eval_dir / "embeddings.npy"


def _sha256_array(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_fixture_artifact(
    artifact_dir: Path,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "embeddings.npy", embeddings.astype(np.float32))
    metadata.to_csv(artifact_dir / "metadata.csv", index=False)
    return _sha256_array(artifact_dir / "embeddings.npy")


def _inject_fixture_label_signal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int = 8,
    scale: float = 1.5,
) -> None:
    """In-place label structure for Gate 0 ID-task probes (fixture loader only)."""
    for k in range(num_classes):
        mask = labels == k
        if not np.any(mask):
            continue
        col = 1 + (k % max(1, embeddings.shape[1] - 1))
        embeddings[mask, col] += scale


def _fixture_metadata(train_n: int, eval_n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_meta = pd.DataFrame(
        {
            "domain": ["isic"] * train_n,
            "label_idx": rng.integers(0, 8, train_n),
        }
    )
    n_pad = max(1, eval_n // 3)
    n_isic = eval_n - n_pad
    eval_meta = pd.DataFrame(
        {
            "domain": ["isic"] * n_isic + ["pad_ufes"] * n_pad,
            "label_idx": np.concatenate(
                [rng.integers(0, 8, n_isic), np.zeros(n_pad, dtype=int)]
            ),
        }
    )
    return train_meta, eval_meta


def extract_embeddings(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
    loader: str | None = None,
    train_n: int = 16_211,
    eval_n: int = 7_365,
    fixture_seed: int = 0,
) -> ExtractionOutput:
    """
    Extract (or skip) train + eval embedding arrays for one backbone.

    PanDerm: skipped when ``PANDERM_EMBEDDINGS_ROOT`` env points at Paper 4 assets.
    Fixture loader: deterministic random arrays for tests (no GPU).
    """
    loader_name = (loader or cfg.raw["backbone"].get("loader", "")).lower()

    if cfg.name == "panderm" and loader_name != "fixture":
        from src.utils.paths import load_panderm_embeddings_root

        root = load_panderm_embeddings_root()
        train_src = root / "ReferenceTrainEmbedding"
        eval_src = root / "ReferenceEmbedding"
        if not train_src.is_dir() or not eval_src.is_dir():
            raise FileNotFoundError(
                f"PanDerm reference embeddings not found under {root}"
            )
        return ExtractionOutput(
            backbone=cfg.name,
            train_dir=train_src,
            eval_dir=eval_src,
            train_sha256="external",
            eval_sha256="external",
            train_n=train_n,
            eval_n=eval_n,
            embed_dim=cfg.embed_dim,
            skipped=True,
            skip_reason="reuse Paper 4 frozen embeddings",
        )

    if loader_name == "timm":
        from src.backbone.loaders.timm_loader import extract_timm_embeddings

        return extract_timm_embeddings(cfg, output_dir=output_dir)

    if loader_name == "medsam":
        from src.backbone.loaders.medsam_loader import extract_medsam_embeddings

        return extract_medsam_embeddings(cfg, output_dir=output_dir)

    if loader_name != "fixture":
        raise NotImplementedError(
            f"Extraction loader {loader_name!r} is not implemented yet. "
            f"Backbone {cfg.name} requires a loader module in src/backbone/loaders/."
        )

    train_dir = output_dir / "ReferenceTrainEmbedding"
    eval_dir = output_dir / "ReferenceEmbedding"
    rng = np.random.default_rng(fixture_seed)
    train = rng.normal(scale=0.1, size=(train_n, cfg.embed_dim)).astype(np.float32)
    train[:, 0] += 2.0
    eval_ = rng.normal(scale=0.1, size=(eval_n, cfg.embed_dim)).astype(np.float32)
    train_meta, eval_meta = _fixture_metadata(train_n, eval_n, fixture_seed)
    _inject_fixture_label_signal(
        train,
        train_meta["label_idx"].to_numpy(),
    )
    n_isic = int((eval_meta["domain"] == "isic").sum())
    eval_[:n_isic, 0] += 2.0
    eval_[n_isic:, 0] -= 2.0
    isic_labels = eval_meta.loc[eval_meta["domain"] == "isic", "label_idx"].to_numpy()
    _inject_fixture_label_signal(eval_[:n_isic], isic_labels)
    train_hash = _write_fixture_artifact(train_dir, train, train_meta)
    eval_hash = _write_fixture_artifact(eval_dir, eval_, eval_meta)
    return ExtractionOutput(
        backbone=cfg.name,
        train_dir=train_dir,
        eval_dir=eval_dir,
        train_sha256=train_hash,
        eval_sha256=eval_hash,
        train_n=train_n,
        eval_n=eval_n,
        embed_dim=cfg.embed_dim,
    )


def pool_spatial_sample(
    cfg: BackboneConfig,
    spatial_tokens: np.ndarray,
    *,
    orig_height: int,
    orig_width: int,
) -> np.ndarray:
    """Apply config pooling to one spatial feature map (used by real loaders)."""
    vec = pool_representation(
        cfg.pooling,
        spatial_tokens,
        embed_dim=cfg.embed_dim,
        orig_height=orig_height,
        orig_width=orig_width,
    )
    if vec.shape[0] != cfg.embed_dim:
        raise ValueError(
            f"{cfg.name}: pooled dim {vec.shape[0]} != config embed_dim {cfg.embed_dim}"
        )
    return vec
