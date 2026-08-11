"""
Mahalanobis-family OOD scorers — ported from Paper4 groupB_mahalanobis_family.py.

Cosine/kNN/KDE reproduce paper-3 formulas verbatim; Mahalanobis scoring uses vendored
CSG-SKin helpers in src/estimators/mahalanobis.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors

from src.estimators.mahalanobis import fpr_at_95_tpr

NUM_CLASSES = 8
REG_EPS = 1e-5
K_VALUES = (1, 10, 50)
PRIMARY_K = 10


def cosine_centroid_scores(z: np.ndarray, means: np.ndarray) -> np.ndarray:
    z_unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    means_unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosine_sim = z_unit @ means_unit.T
    return (1.0 - cosine_sim).min(axis=1)


def compute_knn_scores(
    z_train: np.ndarray,
    z_id: np.ndarray,
    z_ood: np.ndarray,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    k_max = max(k_values)
    index = NearestNeighbors(n_neighbors=k_max).fit(z_train)
    dist_id, _ = index.kneighbors(z_id)
    dist_ood, _ = index.kneighbors(z_ood)
    return {k: (dist_id[:, k - 1], dist_ood[:, k - 1]) for k in k_values}


def density_kde_scores(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    d = z_train.shape[1]
    log_probs = np.empty((z_query.shape[0], num_classes), dtype=np.float64)
    for c in range(num_classes):
        z_c = z_train[y_train == c]
        n_c = len(z_c)
        sigma_bar = z_c.std(axis=0, ddof=1).mean()
        bandwidth = max(sigma_bar * (n_c ** (-1.0 / (d + 4))), 1e-6)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z_c)
        log_probs[:, c] = kde.score_samples(z_query)
    return -log_probs.max(axis=1)


def auroc_fpr95(s_id: np.ndarray, s_ood: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.zeros(len(s_id), dtype=np.int64), np.ones(len(s_ood), dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])
    return float(roc_auc_score(y, scores)), fpr_at_95_tpr(y, scores)
