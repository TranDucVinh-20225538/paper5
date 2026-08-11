"""
Vendored from CSG-SKin/src/utils/ood_metrics.py
Source commit: 12338983d87a35b3ad040687ad09f04908fb0c55
Do not modify without a decision-log entry and a PanDerm regression check (D-027).

Only compute_mahalanobis_params_from_arrays is vendored here. Torch-dependent
scoring helpers remain in CSG-SKin until a later milestone ports them.
"""

from __future__ import annotations

import numpy as np


def compute_mahalanobis_params_from_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 8,
    reg_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Class-conditional means + shared precision (inverse pooled within-class covariance).
    Σ_W = (1/(N-K)) sum_i (x_i - μ_{y_i})(x_i - μ_{y_i})^T
    """
    n_samples, feat_dim = features.shape
    if n_samples < num_classes + 1:
        raise ValueError("Not enough samples to estimate a shared covariance.")

    class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError(f"No samples for class {c}; cannot fit Mahalanobis.")
        class_means[c] = features[mask].mean(axis=0)

    centered = features - class_means[labels.astype(np.int64)]
    denom = max(n_samples - num_classes, 1)
    cov = (centered.T @ centered) / denom
    cov = cov + reg_eps * np.eye(feat_dim, dtype=np.float64)
    precision = np.linalg.inv(cov)
    return class_means.astype(np.float32), precision.astype(np.float32)


def mahalanobis_min_squared_distances(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """Min over classes of squared Mahalanobis distance (higher → more OOD-like)."""
    n_samples = features.shape[0]
    n_classes = class_means.shape[0]
    mins = np.empty(n_samples, dtype=np.float64)
    p = precision.astype(np.float64)
    for i in range(n_samples):
        x = features[i].astype(np.float64)
        best = np.inf
        for c in range(n_classes):
            delta = x - class_means[c].astype(np.float64)
            d2 = float(delta @ p @ delta)
            if d2 < best:
                best = d2
        mins[i] = best
    return mins.astype(np.float32)


def fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    reached = np.where(tpr >= 0.95)[0]
    if reached.size == 0:
        return 1.0
    return float(fpr[reached[0]])
