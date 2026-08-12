"""
Vendored from Paper4/PhaseB/analysis/lid_spectral_decay.py
Source: Paper4 PhaseB (Paper-4-owned geometry diagnostics for Gate 1 selection).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

LID_K = 50


class LidSpectralDiagnostics(NamedTuple):
    lid_mean: float
    lid_k: int
    spectral_slope: float
    composite: float
    n_samples: int
    feat_dim: int
    num_classes: int


def local_intrinsic_dimensionality(features: np.ndarray, k: int = LID_K) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n <= k:
        raise ValueError(f"n_samples={n} must exceed k={k} for LID estimation.")

    tree = cKDTree(features)
    distances, _ = tree.query(features, k=k + 1)
    r = distances[:, 1:]

    r_k = r[:, -1:]
    with np.errstate(divide="ignore"):
        log_ratios = np.log(np.clip(r / r_k, a_min=1e-300, a_max=None))
    mean_log_ratio = np.mean(log_ratios, axis=1)

    lid = np.full(n, np.nan, dtype=np.float64)
    valid = mean_log_ratio < 0
    lid[valid] = -1.0 / mean_log_ratio[valid]
    return lid


def within_class_scatter(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    n, d = features.shape

    s_w = np.zeros((d, d), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError(f"Class {c} has zero samples -- num_classes/labels mismatch.")
        centered = features[mask] - features[mask].mean(axis=0)
        s_w += centered.T @ centered
    return s_w / n


def spectral_decay_slope(s_w: np.ndarray, *, eig_floor: float = 1e-12) -> float:
    """
    Sort eigenvalues of S_w descending, fit log(lambda_i) = a + b*i on positive
    eigenvalues only. When embed_dim exceeds effective rank (common for high-d
    CNN backbones at n~7k), trailing zero eigenvalues are excluded rather than
    treated as fatal — the slope is fit on the support of S_w.
    """
    eigvals = np.linalg.eigvalsh(s_w)
    eigvals = np.sort(eigvals)[::-1]
    pos = eigvals[eigvals > eig_floor]
    if pos.size < 2:
        raise ValueError(
            "S_w has fewer than two positive eigenvalues above the floor; "
            "cannot estimate spectral-decay slope."
        )

    d = pos.shape[0]
    i = np.arange(1, d + 1, dtype=np.float64)
    log_lambda = np.log(pos)
    design = np.vstack([np.ones_like(i), i]).T
    _a, b = np.linalg.lstsq(design, log_lambda, rcond=None)[0]
    return float(abs(b))


def compute_lid_spectral_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int = LID_K,
) -> LidSpectralDiagnostics:
    lid_values = local_intrinsic_dimensionality(features, k=k)
    lid_mean = float(np.nanmean(lid_values))
    s_w = within_class_scatter(features, labels, num_classes)
    slope = spectral_decay_slope(s_w)
    n_samples, feat_dim = features.shape
    return LidSpectralDiagnostics(
        lid_mean=lid_mean,
        lid_k=k,
        spectral_slope=slope,
        composite=lid_mean * slope,
        n_samples=n_samples,
        feat_dim=feat_dim,
        num_classes=num_classes,
    )
