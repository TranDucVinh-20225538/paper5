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

from src.backbone.pooling_dispatch import pool_representation
from src.utils.config import BackboneConfig


@dataclass(frozen=True)
class ExtractionOutput:
    """Paths and checksums for one backbone extraction pass."""

    backbone: str
    train_path: Path
    eval_path: Path
    train_sha256: str
    eval_sha256: str
    train_n: int
    eval_n: int
    embed_dim: int
    skipped: bool = False
    skip_reason: str | None = None


def _sha256_array(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))
    return _sha256_array(path)


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

    out_train = output_dir / f"{cfg.name}_train.npy"
    out_eval = output_dir / f"{cfg.name}_eval.npy"

    if cfg.name == "panderm" and loader_name != "fixture":
        from src.utils.paths import load_panderm_embeddings_root

        root = load_panderm_embeddings_root()
        if not root.is_dir():
            raise FileNotFoundError(
                "PanDerm extraction skipped but PANDERM_EMBEDDINGS_ROOT is unset or missing. "
                "Point it at Paper4/PhaseB/assets/reference_embeddings/ or use loader=fixture."
            )
        train_src = root / "ReferenceTrainEmbedding"
        eval_src = root / "ReferenceEmbedding"
        if not train_src.is_dir() or not eval_src.is_dir():
            raise FileNotFoundError(
                f"PanDerm reference embeddings not found under {root}"
            )
        return ExtractionOutput(
            backbone=cfg.name,
            train_path=train_src,
            eval_path=eval_src,
            train_sha256="external",
            eval_sha256="external",
            train_n=train_n,
            eval_n=eval_n,
            embed_dim=cfg.embed_dim,
            skipped=True,
            skip_reason="reuse Paper 4 frozen embeddings",
        )

    if loader_name != "fixture":
        raise NotImplementedError(
            f"Extraction loader {loader_name!r} is not implemented yet. "
            f"Backbone {cfg.name} requires a loader module in src/backbone/loaders/."
        )

    rng = np.random.default_rng(fixture_seed)
    train = rng.normal(size=(train_n, cfg.embed_dim)).astype(np.float32)
    eval_ = rng.normal(size=(eval_n, cfg.embed_dim)).astype(np.float32)
    train_hash = _write_array(out_train, train)
    eval_hash = _write_array(out_eval, eval_)
    return ExtractionOutput(
        backbone=cfg.name,
        train_path=out_train,
        eval_path=out_eval,
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
