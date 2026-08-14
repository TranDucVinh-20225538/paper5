"""
Vendored from paper-3/scripts/geometry_diagnostics.py
Source commit: 5fedcb3870b1eb17d15c36f79cc11421e7067522
Do not modify without a decision-log entry and a PanDerm regression check (D-027).

REUSE.md §3: κ is computed on the precision matrix derived from pooled within-class
covariance Σ_W (via compute_mahalanobis_params_from_arrays), not the marginal covariance.

D-029 is closed by D-035: κ_primary = λ₁/λ_k over unregularized Σ_W, k=256.
condition_number() remains the vendored κ_paper4 quantity and must not change.
"""

from __future__ import annotations

import numpy as np


def condition_number(precision: np.ndarray) -> float:
    """
    kappa(Sigma) = lambda_max(Sigma) / lambda_min(Sigma).

    Computed directly from the precision matrix (Sigma^-1), not by inverting
    back to Sigma: condition number is invariant under matrix inversion
    (kappa(Sigma) == kappa(Sigma^-1)), since inversion just reciprocates every
    eigenvalue, which flips which one is "max" and which is "min" without
    changing their ratio. This avoids a second, separately-conditioned
    inversion of an already-inverted, already-regularized matrix.
    """
    precision = np.asarray(precision, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(precision)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size < 2:
        raise ValueError("Precision matrix has fewer than 2 positive eigenvalues.")
    return float(eigvals.max() / eigvals.min())


PRIMARY_K = 256
SENSITIVITY_KS = (128, 256, 512)


def unregularized_pooled_within_class_covariance(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 8,
) -> np.ndarray:
    """Pooled within-class covariance Σ_W, same formula as the Mahalanobis scorer, no εI.

    Σ_W = (1/(N-K)) Σ_i (x_i - μ_{y_i})(x_i - μ_{y_i})^T
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    n_samples, feat_dim = features.shape
    if n_samples < num_classes + 1:
        raise ValueError("Not enough samples to estimate a shared covariance.")

    class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError(f"No samples for class {c}; cannot fit Σ_W.")
        class_means[c] = features[mask].mean(axis=0)

    centered = features - class_means[labels]
    denom = max(n_samples - num_classes, 1)
    return (centered.T @ centered) / denom


def kappa_primary(sigma_w: np.ndarray, k: int = PRIMARY_K) -> float:
    """κ_primary = λ₁/λ_k over descending eigenvalues of unregularized Σ_W (D-035).

    No epsilon: this quantity is never inverted. No normalization: κ(cΣ)=κ(Σ).
    ``sigma_w`` must be the pooled within-class covariance, not the marginal
    covariance and not Σ_W + εI.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    sigma_w = np.asarray(sigma_w, dtype=np.float64)
    if sigma_w.ndim != 2 or sigma_w.shape[0] != sigma_w.shape[1]:
        raise ValueError(f"Σ_W must be square, got shape {sigma_w.shape}")
    eigvals = np.linalg.eigvalsh(sigma_w)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals.size < k:
        raise ValueError(f"Σ_W is {eigvals.size}-d; cannot take λ_{k}")
    lam1 = float(eigvals[0])
    lamk = float(eigvals[k - 1])
    if lam1 <= 0 or lamk <= 0:
        raise ValueError(f"Need positive λ₁ and λ_{k}; got {lam1} and {lamk}")
    return float(lam1 / lamk)
