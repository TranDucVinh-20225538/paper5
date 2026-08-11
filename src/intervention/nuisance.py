"""
Nuisance direction w = unit(mu_ISIC - mu_PAD) — protocol Step 4.

Ported from Paper4/PhaseB/analysis/compute_nuisance_direction.py (closed-form, never learned).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.intervention.embeddings import EmbeddingArtifacts


@dataclass(frozen=True)
class NuisanceDirection:
    w: np.ndarray
    w_raw: np.ndarray
    mu_isic: np.ndarray
    mu_pad_ufes: np.ndarray
    w_raw_norm: float
    n_isic: int
    n_pad_ufes: int


def compute_nuisance_direction(artifacts: EmbeddingArtifacts) -> NuisanceDirection:
    pad_mask = (artifacts.eval_metadata["domain"] == "pad_ufes").to_numpy()
    if not np.any(pad_mask):
        raise ValueError("No pad_ufes rows in eval metadata — cannot compute nuisance direction")

    mu_isic = artifacts.train_embeddings.astype(np.float64).mean(axis=0)
    mu_pad = artifacts.eval_embeddings[pad_mask].astype(np.float64).mean(axis=0)
    w_raw = mu_isic - mu_pad
    w_raw_norm = float(np.linalg.norm(w_raw))
    if w_raw_norm == 0.0:
        raise ValueError("mu_ISIC == mu_PAD — nuisance direction is undefined")
    w = (w_raw / w_raw_norm).astype(np.float32)

    return NuisanceDirection(
        w=w,
        w_raw=w_raw.astype(np.float32),
        mu_isic=mu_isic.astype(np.float32),
        mu_pad_ufes=mu_pad.astype(np.float32),
        w_raw_norm=w_raw_norm,
        n_isic=len(artifacts.train_metadata),
        n_pad_ufes=int(pad_mask.sum()),
    )


def save_nuisance_direction(result: NuisanceDirection, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "nuisance_direction.npz"
    np.savez(
        out,
        w=result.w,
        w_raw=result.w_raw,
        mu_isic=result.mu_isic,
        mu_pad_ufes=result.mu_pad_ufes,
    )
    return out
